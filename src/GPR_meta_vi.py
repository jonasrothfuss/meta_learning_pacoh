import torch
import gpytorch
import time
import numpy as np

from torch.distributions.multivariate_normal import MultivariateNormal

from src.models import AffineTransformedDistribution, EqualWeightedMixtureDist
from src.random_gp import RandomGPMeta, RandomGPPosterior
from src.util import _handle_input_dimensionality, DummyLRScheduler
from src.abstract import RegressionModelMetaLearned
from config import device

class GPRegressionMetaLearnedVI(RegressionModelMetaLearned):

    def __init__(self, meta_train_data, num_iter_fit=10000, feature_dim=1,
                 prior_factor=0.01, weight_prior_std=0.5, bias_prior_std=3.0,
                 covar_module='NN', mean_module='NN', mean_nn_layers=(32, 32), kernel_nn_layers=(32, 32),
                 optimizer='Adam', lr=1e-3, lr_scheduler=False, svi_batch_size=10, cov_type='full',
                 task_batch_size=-1, normalize_data=True, random_seed=None):
        """
        Variational GP classification model (https://arxiv.org/abs/1411.2005) that supports prior learning with
        neural network mean and covariance functions

        Args:
            meta_train_data: list of tuples of ndarrays[(train_x_1, train_t_1), ..., (train_x_n, train_t_n)]
            learning_mode: (str) specifying which of the GP prior parameters to optimize. Either one of
                    ['learned_mean', 'learned_kernel', 'both', 'vanilla']
            lr: (float) learning rate for prior parameters
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
            kernel (std): SVGD kernel, either 'RBF' or 'IMQ'
            bandwidth (float): bandwidth of kernel, if None the bandwidth is chosen via heuristic
            num_particles: (int) number particles to approximate the hyper-posterior
            normalize_data: (bool) whether the data should be normalized
            lr_scheduler: (str) whether to use a lr scheduler
            random_seed: (int) seed for pytorch
        """
        super().__init__(normalize_data, random_seed)

        assert mean_module in ['NN', 'constant', 'zero'] or isinstance(mean_module, gpytorch.means.Mean)
        assert covar_module in ['NN', 'SE'] or isinstance(covar_module, gpytorch.kernels.Kernel)
        assert optimizer in ['Adam', 'SGD']

        self.num_iter_fit, self.prior_factor, self.feature_dim = num_iter_fit, prior_factor, feature_dim
        self.weight_prior_std, self.bias_prior_std = weight_prior_std, bias_prior_std
        self.svi_batch_size = svi_batch_size
        if task_batch_size < 1:
            self.task_batch_size = len(meta_train_data)
        else:
            self.task_batch_size = min(task_batch_size, len(meta_train_data))

        # Check that data all has the same size
        self._check_meta_data_shapes(meta_train_data)

        """ --- Setup model & inference --- """
        self._setup_model_inference(mean_module, covar_module, mean_nn_layers, kernel_nn_layers,
                                    cov_type)

        self._setup_optimizer(optimizer, lr, lr_scheduler)

        # Setup components that are different across tasks
        self.task_dicts = []

        for train_x, train_y in meta_train_data:
            task_dict = {}

            x_tensor, y_tensor, task_dict = self._prepare_data_per_task(train_x, train_y, stats_dict=task_dict)
            task_dict['train_x'], task_dict['train_y'] = x_tensor, y_tensor
            self.task_dicts.append(task_dict)

        self.fitted = False


    def meta_fit(self, valid_tuples=None, verbose=True, log_period=500):

        """
        fits the variational hyper-posterior by minimizing the negative ELBO

        Args:
            valid_tuples: list of valid tuples, i.e. [(test_context_x_1, test_context_t_1, test_x_1, test_t_1), ...]
            verbose: (boolean) whether to print training progress
            log_period (int) number of steps after which to print stats

        """

        assert (valid_tuples is None) or (all([len(valid_tuple) == 4 for valid_tuple in valid_tuples]))


        t = time.time()

        for itr in range(1, self.num_iter_fit + 1):

            task_dict_batch = self.rds_numpy.choice(self.task_dicts, size=self.task_batch_size)
            self.optimizer.zero_grad()
            loss = self.get_neg_elbo(task_dict_batch)
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step(loss)

            # print training stats stats
            if itr == 1 or itr % log_period == 0:
                duration = time.time() - t
                t = time.time()

                message = 'Iter %d/%d - Loss: %.6f - Time %.2f sec' % (itr, self.num_iter_fit, loss.item(), duration)

                # if validation data is provided  -> compute the valid log-likelihood
                if valid_tuples is not None:
                    valid_ll, valid_rmse = self.eval_datasets(valid_tuples)
                    message += ' - Valid-LL: %.3f - Valid-RMSE: %.3f' % (np.mean(valid_ll), np.mean(valid_rmse))

                if verbose:
                    self.logger.info(message)

        self.fitted = True

    def predict(self, context_x, context_y, test_x, n_posterior_samples=100, mode='Bayes', return_density=False):
        """
        computes the predictive distribution of the targets p(t|test_x, test_context_x, context_y)

        Args:
            context_x: (ndarray) context input data for which to compute the posterior
            context_y: (ndarray) context targets for which to compute the posterior
            test_x: (ndarray) query input data of shape (n_samples, ndim_x)
                        n_posterior_samples: (int) number of samples from posterior to average over
            mode: (std) either of ['Bayes' , 'MAP']
            return_density: (bool) whether to return result as mean and std ndarray or as MultivariateNormal pytorch object

        Returns:
            (pred_mean, pred_std) predicted mean and standard deviation corresponding to p(t|test_x, test_context_x, context_y)
        """
        assert mode in ['bayes', 'Bayes', 'MAP', 'map']

        context_x, context_y = _handle_input_dimensionality(context_x, context_y)
        test_x = _handle_input_dimensionality(test_x)
        assert test_x.shape[1] == context_x.shape[1]

        # normalize data and convert to tensor
        context_x, context_y, data_stats = self._prepare_data_per_task(context_x, context_y, stats_dict={})

        test_x = self._normalize_data(X=test_x, Y=None, stats_dict=data_stats)
        test_x = torch.from_numpy(test_x).float().to(device)

        with torch.no_grad():

            if mode == 'Bayes' or mode == 'bayes':
                pred_dist = self.get_pred_dist(context_x, context_y, test_x, n_post_samples=n_posterior_samples)
                pred_dist = AffineTransformedDistribution(pred_dist, normalization_mean=data_stats['y_mean'],
                                                      normalization_std=data_stats['y_std'])

                pred_dist = EqualWeightedMixtureDist(pred_dist, batched=True)
            else:
                pred_dist = self.get_pred_dist_map(context_x, context_y, test_x)
                pred_dist = AffineTransformedDistribution(pred_dist, normalization_mean=data_stats['y_mean'],
                                                      normalization_std=data_stats['y_std'])
            if return_density:
                return pred_dist
            else:
                pred_mean = pred_dist.mean.cpu().numpy()
                pred_std = pred_dist.stddev.cpu().numpy()
                return pred_mean, pred_std

    def eval(self, context_x, context_y, test_x, test_t, n_posterior_samples=100, mode='Bayes'):
        """
              Computes the average test log likelihood and the rmse on test data

              Args:
                  context_x: (ndarray) context input data for which to compute the posterior
                  context_y: (ndarray) context targets for which to compute the posterior
                  test_x: (ndarray) test input data of shape (n_samples, ndim_x)
                  test_t: (ndarray) test target data of shape (n_samples, 1)

              Returns: (avg_log_likelihood, rmse)

              """

        context_x, context_y = _handle_input_dimensionality(context_x, context_y)
        test_x, test_t = _handle_input_dimensionality(test_x, test_t)
        test_t_tensor = torch.from_numpy(test_t).float().flatten().to(device)

        with torch.no_grad():
            pred_dist = self.predict(context_x, context_y, test_x, n_posterior_samples=n_posterior_samples,
                                     mode=mode, return_density=True)
            avg_log_likelihood = pred_dist.log_prob(test_t_tensor) / test_t_tensor.shape[0]
            rmse = torch.mean(torch.pow(pred_dist.mean - test_t_tensor, 2)).sqrt()

            return avg_log_likelihood.cpu().item(), rmse.cpu().item()

    def eval_datasets(self, test_tuples, **kwargs):
        """
        Computes the average test log likelihood and the rmse over multiple test datasets

        Args:
            test_tuples: list of test set tuples, i.e. [(test_context_x_1, test_context_y_1, test_x_1, test_y_1), ...]

        Returns: (avg_log_likelihood, rmse)

        """

        assert (all([len(valid_tuple) == 4 for valid_tuple in test_tuples]))

        ll_list, rmse_list = list(zip(*[self.eval(*test_data_tuple, **kwargs) for test_data_tuple in test_tuples]))

        return np.mean(ll_list), np.mean(rmse_list)

    def state_dict(self):
        state_dict = {
            'optimizer': self.optimizer.state_dict(),
            'model': self.task_dicts[0]['model'].state_dict()
        }
        for task_dict in self.task_dicts:
            for key, tensor in task_dict['model'].state_dict().items():
                assert torch.all(state_dict['model'][key] == tensor).item()
        return state_dict

    def load_state_dict(self, state_dict):
        for task_dict in self.task_dicts:
            task_dict['model'].load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])

    def _setup_model_inference(self, mean_module_str, covar_module_str, mean_nn_layers, kernel_nn_layers, cov_type):
        assert mean_module_str in ['NN', 'constant']
        assert covar_module_str in ['NN', 'SE']

        """ random gp model """
        self.random_gp = RandomGPMeta(size_in=self.input_dim, prior_factor=self.prior_factor,
                                  weight_prior_std=self.weight_prior_std, bias_prior_std=self.bias_prior_std,
                                  covar_module_str=covar_module_str, mean_module_str=mean_module_str,
                                  mean_nn_layers=mean_nn_layers, kernel_nn_layers=kernel_nn_layers)

        param_shapes_dict = self.random_gp.parameter_shapes()

        """ variational posterior """
        self.posterior = RandomGPPosterior(param_shapes_dict, cov_type=cov_type)

        def _tile_data_tuples(tasks_dicts, tile_size):
            train_data_tuples_tiled = []
            for task_dict in tasks_dicts:
                x_data, y_data = task_dict['train_x'], task_dict['train_y']
                x_data = x_data.view(torch.Size((1,)) + x_data.shape).repeat(tile_size, 1, 1)
                y_data = y_data.view(torch.Size((1,)) + y_data.shape).repeat(tile_size, 1)
                train_data_tuples_tiled.append((x_data, y_data))
            return train_data_tuples_tiled

        """ define negative ELBO """
        def get_neg_elbo(tasks_dicts):
            # tile data to svi_batch_shape
            data_tuples_tiled = _tile_data_tuples(tasks_dicts, self.svi_batch_size)

            param_sample = self.posterior.rsample(sample_shape=(self.svi_batch_size,))
            elbo = self.random_gp.log_prob(param_sample, data_tuples_tiled) - self.prior_factor * self.posterior.log_prob(param_sample)

            assert elbo.ndim == 1 and elbo.shape[0] == self.svi_batch_size
            return - torch.mean(elbo)

        self.get_neg_elbo = get_neg_elbo

        """ define predictive dist """
        def get_pred_dist(x_context, y_context, x_valid, n_post_samples=100):
            with torch.no_grad():
                x_context = x_context.view(torch.Size((1,)) + x_context.shape).repeat(n_post_samples, 1, 1)
                y_context = y_context.view(torch.Size((1,)) + y_context.shape).repeat(n_post_samples, 1)
                x_valid = x_valid.view(torch.Size((1,)) + x_valid.shape).repeat(n_post_samples, 1, 1)

                param_sample = self.posterior.sample(sample_shape=(n_post_samples,))
                gp_fn = self.random_gp.get_forward_fn(param_sample)
                gp, likelihood = gp_fn(x_context, y_context, train=False)
                pred_dist = likelihood(gp(x_valid))
            return pred_dist

        def get_pred_dist_map(x_context, y_context, x_valid):
            with torch.no_grad():
                x_context = x_context.view(torch.Size((1,)) + x_context.shape).repeat(1, 1, 1)
                y_context = y_context.view(torch.Size((1,)) + y_context.shape).repeat(1, 1)
                x_valid = x_valid.view(torch.Size((1,)) + x_valid.shape).repeat(1, 1, 1)
                param = self.posterior.mode
                param = param.view(torch.Size((1,)) + param.shape).repeat(1, 1)

                gp_fn = self.random_gp.get_forward_fn(param)
                gp, likelihood = gp_fn(x_context, y_context, train=False)
                pred_dist = likelihood(gp(x_valid))
            return MultivariateNormal(pred_dist.loc, pred_dist.covariance_matrix[0])


        self.get_pred_dist = get_pred_dist
        self.get_pred_dist_map = get_pred_dist_map

    def _setup_optimizer(self, optimizer, lr, lr_scheduler):
        if optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.posterior.parameters(), lr=lr)
        elif optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.posterior.parameters(), lr=lr)
        else:
            raise NotImplementedError('Optimizer must be Adam or SGD')

        if lr_scheduler:
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',
                                                                           factor=0.2, patience=100, threshold=1e-3)
        else:
            self.lr_scheduler = DummyLRScheduler()

if __name__ == "__main__":
    """ 1) Generate some training data from GP prior """
    from experiments.data_sim import GPFunctionsDataset

    data_sim = GPFunctionsDataset(random_state=np.random.RandomState(26))

    meta_train_data = data_sim.generate_meta_train_data(n_tasks=10, n_samples=40)
    meta_test_data = data_sim.generate_meta_test_data(n_tasks=10, n_samples_context=40, n_samples_test=160)

    NN_LAYERS = (32, 32)

    plot = False
    from matplotlib import pyplot as plt

    if plot:
        for x_train, y_train in meta_train_data:
            plt.scatter(x_train, y_train)
        plt.title('sample from the GP prior')
        plt.show()

    """ 2) Classical mean learning based on mll """

    from src.GPR_meta_mll import GPRegressionMetaLearned

    # print(' ---- GPR mll meta-learning ---- ')
    #
    # gp_model = GPRegressionMetaLearned(meta_train_data, num_iter_fit=10000, covar_module='SE', mean_module='NN', mean_nn_layers=NN_LAYERS,
    #                                      kernel_nn_layers=NN_LAYERS)
    #
    # gp_model.meta_fit(valid_tuples=meta_test_data, log_period=1000)

    print('\n ---- GPR VI meta-learning ---- ')

    torch.set_num_threads(2)

    for prior_factor in [1e-3 / 40.]:
        gp_model = GPRegressionMetaLearnedVI(meta_train_data, num_iter_fit=10000, prior_factor=prior_factor, svi_batch_size=10, task_batch_size=2,
                                             covar_module='NN', mean_module='NN', mean_nn_layers=NN_LAYERS, kernel_nn_layers=NN_LAYERS, cov_type='diag')

        gp_model.meta_fit(valid_tuples=meta_test_data, log_period=100)


        # x_test = np.linspace(-5, 5, num=150)
        # x_context, t_context, _, _ = meta_test_data[0]
        # pred_mean, pred_std = gp_model.predict(x_context, t_context, x_test)
        #
        # plt.scatter(x_context, t_context)
        # plt.plot(x_test, pred_mean)
        # plt.fill_between(x_test, pred_mean-pred_std, pred_mean+pred_std, alpha=0.2)
        # plt.title('[test_meta_vi.py] GPR meta VI (prior-factor =  %i)'%prior_factor)
        # plt.show()