import torch
import gpytorch
import time

import numpy as np

from src.models import AffineTransformedDistribution, EqualWeightedMixtureDist
from src.random_gp import RandomGPMeta
from src.util import _handle_input_dimensionality, DummyLRScheduler
from src.svgd import SVGD, RBF_Kernel, IMQSteinKernel
from src.abstract import RegressionModelMetaLearned
from config import device

class GPRegressionMetaLearnedSVGD(RegressionModelMetaLearned):

    def __init__(self, meta_train_data, num_iter_fit=10000, feature_dim=1,
                 prior_factor=0.01, weight_prior_std=0.5, bias_prior_std=3.0,
                 covar_module='NN', mean_module='NN', mean_nn_layers=(32, 32), kernel_nn_layers=(32, 32),
                 optimizer='Adam', lr=1e-3, lr_decay=1.0, kernel='RBF', bandwidth=None, num_particles=10,
                 task_batch_size=-1, normalize_data=True, random_seed=None):
        """
        Variational GP classification model (https://arxiv.org/abs/1411.2005) that supports prior learning with
        neural network mean and covariance functions

        Args:
            meta_train_data: list of tuples of ndarrays[(train_x_1, train_t_1), ..., (train_x_n, train_t_n)]
            num_iter_fit: (int) number of gradient steps for fitting the parameters
            prior_factor: (float) weighting of the hyper-prior (--> meta-regularization parameter)
            feature_dim: (int) output dimensionality of NN feature map for kernel function
            weight_prior_std (float): std of Gaussian hyper-prior on weights
            bias_prior_std (float): std of Gaussian hyper-prior on biases
            covar_module: (gpytorch.mean.Kernel) optional kernel module, default: RBF kernel
            mean_module: (gpytorch.mean.Mean) optional mean module, default: ZeroMean
            mean_nn_layers: (tuple) hidden layer sizes of mean NN
            kernel_nn_layers: (tuple) hidden layer sizes of kernel NN
            optimizer: (str) type of optimizer to use - must be either 'Adam' or 'SGD'
            lr: (float) learning rate for prior parameters
            lr_decay: (float) lr rate decay multiplier applied after every 1000 steps
            kernel (std): SVGD kernel, either 'RBF' or 'IMQ'
            bandwidth (float): bandwidth of kernel, if None the bandwidth is chosen via heuristic
            num_particles: (int) number particles to approximate the hyper-posterior
            normalize_data: (bool) whether the data should be normalized
            random_seed: (int) seed for pytorch
        """
        super().__init__(normalize_data, random_seed)

        assert mean_module in ['NN', 'constant', 'zero'] or isinstance(mean_module, gpytorch.means.Mean)
        assert covar_module in ['NN', 'SE'] or isinstance(covar_module, gpytorch.kernels.Kernel)
        assert optimizer in ['Adam', 'SGD']

        self.num_iter_fit, self.prior_factor, self.feature_dim = num_iter_fit, prior_factor, feature_dim
        self.weight_prior_std, self.bias_prior_std = weight_prior_std, bias_prior_std
        self.num_particles = num_particles
        if task_batch_size < 1:
            self.task_batch_size = len(meta_train_data)
        else:
            self.task_batch_size = min(task_batch_size, len(meta_train_data))

        # Check that data all has the same size
        self._check_meta_data_shapes(meta_train_data)

        """ --- Setup model & inference --- """
        self._setup_model_inference(mean_module, covar_module, mean_nn_layers, kernel_nn_layers,
                                    kernel, bandwidth, optimizer, lr, lr_decay)

        # Setup components that are different across tasks
        self.task_dicts = []

        for train_x, train_y in meta_train_data:
            task_dict = {}

            # a) prepare data
            x_tensor, y_tensor, task_dict = self._prepare_data_per_task(train_x, train_y, stats_dict=task_dict)
            task_dict['train_x'], task_dict['train_y'] = x_tensor, y_tensor
            self.task_dicts.append(task_dict)

        self.fitted = False


    def meta_fit(self, valid_tuples=None, verbose=True, log_period=500, n_iter=None):

        """
        fits the hyper-posterior particles with SVGD

        Args:
            valid_tuples: list of valid tuples, i.e. [(test_context_x_1, test_context_t_1, test_x_1, test_t_1), ...]
            verbose: (boolean) whether to print training progress
            log_period (int) number of steps after which to print stats
            n_iter: (int) number of gradient descent iterations
        """

        assert (valid_tuples is None) or (all([len(valid_tuple) == 4 for valid_tuple in valid_tuples]))

        t = time.time()

        if n_iter is None:
            n_iter = self.num_iter_fit

        for itr in range(1, n_iter + 1):

            task_dict_batch = self.rds_numpy.choice(self.task_dicts, size=self.task_batch_size)
            self.svgd_step(task_dict_batch)
            self.lr_scheduler.step()

            # print training stats stats
            if itr == 1 or itr % log_period == 0:
                duration = time.time() - t
                t = time.time()

                message = 'Iter %d/%d - Time %.2f sec' % (itr, self.num_iter_fit, duration)

                # if validation data is provided  -> compute the valid log-likelihood
                if valid_tuples is not None:
                    valid_ll, valid_rmse, calibr_err = self.eval_datasets(valid_tuples)
                    message += ' - Valid-LL: %.3f - Valid-RMSE: %.3f - Calib-Err %.3f' % (valid_ll, valid_rmse, calibr_err)

                if verbose:
                    self.logger.info(message)

        self.fitted = True

    def predict(self, context_x, context_y, test_x, return_density=False):
        """
        computes the predictive distribution of the targets p(t|test_x, test_context_x, context_y)

        Args:
            context_x: (ndarray) context input data for which to compute the posterior
            context_y: (ndarray) context targets for which to compute the posterior
            test_x: (ndarray) query input data of shape (n_samples, ndim_x)
            return_density: (bool) whether to return result as mean and std ndarray or as MultivariateNormal pytorch object

        Returns:
            (pred_mean, pred_std) predicted mean and standard deviation corresponding to p(t|test_x, test_context_x, context_y)
        """

        context_x, context_y = _handle_input_dimensionality(context_x, context_y)
        test_x = _handle_input_dimensionality(test_x)
        assert test_x.shape[1] == context_x.shape[1]

        # normalize data and convert to tensor
        context_x, context_y, data_stats = self._prepare_data_per_task(context_x, context_y, stats_dict={})

        test_x = self._normalize_data(X=test_x, Y=None, stats_dict=data_stats)
        test_x = torch.from_numpy(test_x).float().to(device)

        with torch.no_grad():
            pred_dist = self.get_pred_dist(context_x, context_y, test_x)
            pred_dist = AffineTransformedDistribution(pred_dist, normalization_mean=data_stats['y_mean'],
                                                      normalization_std=data_stats['y_std'])
            pred_dist = EqualWeightedMixtureDist(pred_dist, batched=True)

            if return_density:
                return pred_dist
            else:
                pred_mean = pred_dist.mean.cpu().numpy()
                pred_std = pred_dist.stddev.cpu().numpy()
                return pred_mean, pred_std

    def _setup_model_inference(self, mean_module_str, covar_module_str, mean_nn_layers, kernel_nn_layers,
                               kernel, bandwidth, optimizer, lr, lr_decay):
        assert mean_module_str in ['NN', 'constant']
        assert covar_module_str in ['NN', 'SE']

        """ random gp model """
        self.random_gp = RandomGPMeta(size_in=self.input_dim, prior_factor=self.prior_factor,
                                  weight_prior_std=self.weight_prior_std, bias_prior_std=self.bias_prior_std,
                                  covar_module_str=covar_module_str, mean_module_str=mean_module_str,
                                  mean_nn_layers=mean_nn_layers, kernel_nn_layers=kernel_nn_layers)

        """ Setup SVGD inference"""

        if kernel == 'RBF':
            kernel = RBF_Kernel(bandwidth=bandwidth)
        elif kernel == 'IMQ':
            kernel = IMQSteinKernel(bandwidth=bandwidth)
        else:
            raise NotImplemented

        # sample initial particle locations from prior
        self.particles = self.random_gp.sample_params_from_prior(shape=(self.num_particles,))

        self._setup_optimizer(optimizer, lr, lr_decay)

        self.svgd = SVGD(self.random_gp, kernel, optimizer=self.optimizer)

        """ define svgd step """

        def svgd_step(tasks_dicts):
            # tile data to svi_batch_shape
            train_data_tuples_tiled = []
            for task_dict in tasks_dicts:
                x_data, y_data = task_dict['train_x'], task_dict['train_y']
                x_data = x_data.view(torch.Size((1,)) + x_data.shape).repeat(self.num_particles, 1, 1)
                y_data = y_data.view(torch.Size((1,)) + y_data.shape).repeat(self.num_particles, 1)
                train_data_tuples_tiled.append((x_data, y_data))

            self.svgd.step(self.particles, train_data_tuples_tiled)

        """ define predictive dist """

        def get_pred_dist(x_context, y_context, x_valid):
            with torch.no_grad():
                x_context = x_context.view(torch.Size((1,)) + x_context.shape).repeat(self.num_particles, 1, 1)
                y_context = y_context.view(torch.Size((1,)) + y_context.shape).repeat(self.num_particles, 1)
                x_valid = x_valid.view(torch.Size((1,)) + x_valid.shape).repeat(self.num_particles, 1, 1)

                gp_fn = self.random_gp.get_forward_fn(self.particles)
                gp, likelihood = gp_fn(x_context, y_context, train=False)
                pred_dist = likelihood(gp(x_valid))
            return pred_dist

        self.svgd_step = svgd_step
        self.get_pred_dist = get_pred_dist

    def _setup_optimizer(self, optimizer, lr, lr_decay):
        assert hasattr(self, 'particles'), "SVGD must be initialized before setting up optimizer"

        if optimizer == 'Adam':
            self.optimizer = torch.optim.Adam([self.particles], lr=lr)
        elif optimizer == 'SGD':
            self.optimizer = torch.optim.SGD([self.particles], lr=lr)
        else:
            raise NotImplementedError('Optimizer must be Adam or SGD')

        if lr_decay < 1.0:
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1000, gamma=lr_decay)
        else:
            self.lr_scheduler = DummyLRScheduler()

    def _vectorize_pred_dist(self, pred_dist):
        multiv_normal_batched = pred_dist.dists
        normal_batched = torch.distributions.Normal(multiv_normal_batched.mean, multiv_normal_batched.stddev)
        return EqualWeightedMixtureDist(normal_batched, batched=True, num_dists=multiv_normal_batched.batch_shape[0])

if __name__ == "__main__":
    """ 1) Generate some training data from GP prior """
    from experiments.data_sim import GPFunctionsDataset

    data_sim = GPFunctionsDataset(random_state=np.random.RandomState(26))

    # meta_train_data = data_sim.generate_meta_train_data(n_tasks=5, n_samples=40)
    meta_test_data = data_sim.generate_meta_test_data(n_tasks=5, n_samples_context=40, n_samples_test=160)
    meta_train_data = [(context_x, context_y) for context_x, context_y, _, _ in meta_test_data]

    NN_LAYERS = (16, 16)

    plot = False
    from matplotlib import pyplot as plt

    if plot:
        for x_train, y_train in meta_train_data:
            plt.scatter(x_train, y_train)
        plt.title('sample from the GP prior')
        plt.show()

    for prior_factor in [1e-3]:
        gp_model = GPRegressionMetaLearnedSVGD(meta_train_data, num_iter_fit=2000, prior_factor=prior_factor, num_particles=10,
                                             covar_module='SE', mean_module='NN', mean_nn_layers=NN_LAYERS, kernel_nn_layers=NN_LAYERS,
                                             bandwidth=0.5, task_batch_size=2)

        for i in range(10):
            itrs = 0
            gp_model.meta_fit(valid_tuples=meta_test_data, log_period=500, n_iter=1000)
            itrs += 10

            x_test = np.linspace(-5, 5, num=150)
            x_context, t_context, _, _ = meta_test_data[0]
            pred_mean, pred_std = gp_model.predict(x_context, t_context, x_test)
            ucb, lcb = gp_model.confidence_intervals(x_context, t_context, x_test, confidence=0.9)

            plt.scatter(x_context, t_context)
            plt.plot(x_test, pred_mean)
            plt.fill_between(x_test, lcb, ucb, alpha=0.2)
            plt.title('GPR meta SVGD (prior-factor =  %.4f) itrs = %i'%(prior_factor, itrs))
            plt.show()