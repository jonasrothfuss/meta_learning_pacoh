import torch
import gpytorch
import time
import numpy as np
import math
from collections import OrderedDict

from torch.distributions.multivariate_normal import MultivariateNormal

from meta_learn.models import AffineTransformedDistribution, EqualWeightedMixtureDist, LearnedGPRegressionModelApproximate
from meta_learn.random_gp import RandomGPMeta, RandomGPPosterior
from meta_learn.util import _handle_input_dimensionality, DummyLRScheduler
from meta_learn.abstract import RegressionModelMetaLearned
from config import device

class GPRegressionMetaLearnedPAC(RegressionModelMetaLearned):

    def __init__(self, meta_train_data, num_iter_fit=40000, feature_dim=1, weight_prior_std=0.5, bias_prior_std=3.0,
                 delta=0.1, task_kl_weight=1.0, meta_kl_weight=1.0, posterior_lr_multiplier=1.0,
                 covar_module='SE', mean_module='zero', mean_nn_layers=(32, 32), kernel_nn_layers=(32, 32),
                 optimizer='Adam', lr=1e-3, lr_decay=1.0, svi_batch_size=5, cov_type='diag',
                 task_batch_size=-1, likelihood_noise_init=0.01, normalize_data=True, random_seed=None):
        """
        PACOH-VI: Variational Inference on the PAC-optimal hyper-posterior with Gaussian family.
        Meta-Learns a distribution over GP-priors.

        Args:
            meta_train_data: list of tuples of ndarrays[(train_x_1, train_t_1), ..., (train_x_n, train_t_n)]
            num_iter_fit: (int) number of gradient steps for fitting the parameters
            feature_dim: (int) output dimensionality of NN feature map for kernel function
            prior_factor: (float) weighting of the hyper-prior (--> meta-regularization parameter)
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
            task_batch_size: (int) mini-batch size of tasks for estimating gradients
            normalize_data: (bool) whether the data should be normalized
            random_seed: (int) seed for pytorch
        """
        super().__init__(normalize_data, random_seed)

        assert mean_module in ['NN', 'constant', 'zero'] or isinstance(mean_module, gpytorch.means.Mean)
        assert covar_module in ['NN', 'SE'] or isinstance(covar_module, gpytorch.kernels.Kernel)
        assert optimizer in ['Adam', 'SGD']

        self.num_iter_fit, self.feature_dim = num_iter_fit, feature_dim
        self.task_kl_weight, self.meta_kl_weight = task_kl_weight, meta_kl_weight
        self.weight_prior_std, self.bias_prior_std = weight_prior_std, bias_prior_std
        self.svi_batch_size = svi_batch_size
        self.lr = lr
        self.n_tasks = len(meta_train_data)
        self.delta = torch.tensor(delta, dtype=torch.float32)
        if task_batch_size < 1:
            self.task_batch_size = len(meta_train_data)
        else:
            self.task_batch_size = min(task_batch_size, len(meta_train_data))

        # Check that data all has the same size
        self._check_meta_data_shapes(meta_train_data)
        self._compute_normalization_stats(meta_train_data)

        """ --- Setup model & inference --- """
        self.meta_train_params = []

        self._setup_meta_train_step(mean_module, covar_module, mean_nn_layers, kernel_nn_layers,
                                    cov_type)
        self.meta_train_params.append({'params': self.hyper_posterior.parameters(), 'lr': lr})

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.likelihood.noise = likelihood_noise_init * torch.ones((1,))
        self.meta_train_params.append({'params': self.likelihood.parameters(), 'lr': lr})

        # Setup components that are different across tasks
        self.task_dicts, posterior_params = self._setup_task_dicts(meta_train_data)
        self.meta_train_params.append({'params': posterior_params, 'lr': posterior_lr_multiplier * lr})

        self._setup_optimizer(optimizer, lr, lr_decay)

        self.fitted = False


    def meta_fit(self, valid_tuples=None, verbose=True, log_period=500, eval_period=5000, n_iter=None):
        """
        fits the variational hyper-posterior by minimizing the negative ELBO

        Args:
            valid_tuples: list of valid tuples, i.e. [(test_context_x_1, test_context_t_1, test_x_1, test_t_1), ...]
            verbose: (boolean) whether to print training progress
            log_period (int) number of steps after which to print stats
            n_iter: (int) number of gradient descent iterations
        """
        assert eval_period % log_period == 0, "eval_period should be multiple of log_period"
        assert (valid_tuples is None) or (all([len(valid_tuple) == 4 for valid_tuple in valid_tuples]))

        t = time.time()

        if n_iter is None:
            n_iter = self.num_iter_fit

        for itr in range(1, n_iter + 1):

            task_dict_batch = self.rds_numpy.choice(self.task_dicts, size=self.task_batch_size)
            self.optimizer.zero_grad()
            loss, diagnostics_dict = self._meta_train_pac_bound(task_dict_batch)
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            # print training stats stats
            if verbose and (itr == 1 or itr % log_period == 0):
                duration = time.time() - t
                t = time.time()

                message = 'Iter %d/%d - Loss: %.6f - Time %.2f sec - ' % (itr, self.num_iter_fit, loss.item(), duration)

                # if validation data is provided  -> compute the valid log-likelihood
                if valid_tuples is not None and itr % eval_period == 0 and itr > 0:
                    valid_ll, valid_rmse, calibr_err = self.eval_datasets(valid_tuples)
                    message += ' - Valid-LL: %.3f - Valid-RMSE: %.3f - Calib-Err %.3f' % (valid_ll, valid_rmse, calibr_err)

                # add diagnostics
                message += ' - '.join(['%s: %.4f'%(key, value) for key, value in diagnostics_dict.items()])
                self.logger.info(message)

        self.fitted = True
        return loss.item(), diagnostics_dict

    def predict(self, context_x, context_y, test_x, n_iter_meta_test=3000, return_density=False):
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
        context_x, context_y = _handle_input_dimensionality(context_x, context_y)
        test_x = _handle_input_dimensionality(test_x)
        assert test_x.shape[1] == context_x.shape[1]

        # meta-test training / inference
        task_dict = self._meta_test_inference([(context_x, context_y)], verbose=True, log_period=500,
                                              n_iter=n_iter_meta_test)[0]

        with torch.no_grad():
            # meta-test evaluation
            test_x = self._normalize_data(X=test_x, Y=None)
            test_x = torch.from_numpy(test_x).float().to(device)

            gp_model = task_dict["gp_model"]
            gp_model.eval()
            pred_dist = self.likelihood(gp_model(test_x))
            pred_dist = AffineTransformedDistribution(pred_dist, normalization_mean=self.y_mean,
                                                  normalization_std=self.y_std)
            if return_density:
                return pred_dist
            else:
                pred_mean = pred_dist.mean.cpu().numpy()
                pred_std = pred_dist.stddev.cpu().numpy()
                return pred_mean, pred_std

    def eval_datasets(self, test_tuples, n_iter_meta_test=3000, **kwargs):
        """
        Performs meta-testing on multiple tasks / datasets.
        Computes the average test log likelihood, the rmse and the calibration error over multiple test datasets

        Args:
            test_tuples: list of test set tuples, i.e. [(test_context_x_1, test_context_y_1, test_x_1, test_y_1), ...]

        Returns: (avg_log_likelihood, rmse, calibr_error)

        """
        assert (all([len(valid_tuple) == 4 for valid_tuple in test_tuples]))

        # meta-test training / inference
        context_tuples = [test_tuple[:2] for test_tuple in test_tuples]
        task_dicts = self._meta_test_inference(context_tuples, verbose=True, log_period=500,
                                              n_iter=n_iter_meta_test)

        # meta-test evaluation
        ll_list, rmse_list, calibr_err_list = [], [], []
        for task_dict, test_tuple in zip(task_dicts, test_tuples):
            # data prep
            _, _, test_x, test_y = test_tuple
            test_x, test_y = _handle_input_dimensionality(test_x, test_y)
            test_x_tensor = torch.from_numpy(self._normalize_data(X=test_x, Y=None)).float().to(device)
            test_y_tensor = torch.from_numpy(test_y).float().flatten().to(device)

            # get predictive dist
            gp_model = task_dict["gp_model"]
            gp_model.eval()
            self.likelihood.eval()
            pred_dist = self.likelihood(gp_model(test_x_tensor))
            pred_dist = AffineTransformedDistribution(pred_dist, normalization_mean=self.y_mean,
                                                      normalization_std=self.y_std)

            # compute eval metrics
            ll_list.append(torch.mean(pred_dist.log_prob(test_y_tensor) / test_y_tensor.shape[0]).cpu().item())
            rmse_list.append(torch.mean(torch.pow(pred_dist.mean - test_y_tensor, 2)).sqrt().cpu().item())
            pred_dist_vect = self._vectorize_pred_dist(pred_dist)
            calibr_err_list.append(self._calib_error(pred_dist_vect, test_y_tensor).cpu().item())

        return np.mean(ll_list), np.mean(rmse_list), np.mean(calibr_err_list)

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

    def _setup_task_dicts(self, train_data_tuples):
        task_dicts, parameters = [], []

        for train_x, train_y in train_data_tuples:
            task_dict = OrderedDict()

            # a) prepare data
            x_tensor, y_tensor = self._prepare_data_per_task(train_x, train_y)
            task_dict['train_x'], task_dict['train_y'] = x_tensor, y_tensor

            prior_param_sample = self.hyper_posterior.sample(sample_shape=(20,))
            mean_module, covar_module = self._aggregate_gp_priors(prior_param_sample)

            task_dict['gp_model'] = LearnedGPRegressionModelApproximate(x_tensor, y_tensor, self.likelihood,
                                                                        mean_module=mean_module,
                                                                        covar_module=covar_module)
            init_mean = mean_module(x_tensor)
            init_mean += 1e-3 * torch.randn_like(init_mean)
            task_dict['gp_model'].variational_distribution.variational_mean.data.copy_(init_mean)

            prior_covar = covar_module(x_tensor)
            init_chol_covar = torch.cholesky(prior_covar + 1e-3 * torch.eye(prior_covar.shape[0]))
            task_dict['gp_model'].variational_distribution.chol_variational_covar.data.copy_(init_chol_covar)

            parameters.extend(task_dict['gp_model'].variational_parameters())
            task_dicts.append(task_dict)

        return task_dicts, parameters

    def _meta_test_inference(self, context_tuples, n_iter=5000, lr=1e-2, log_period=100, verbose=False):
        n_tasks = len(context_tuples)
        task_dicts, posterior_params = self._setup_task_dicts(context_tuples)

        optimizer = torch.optim.Adam(posterior_params, lr=lr)

        t = time.time()
        for itr in range(n_iter):
            optimizer.zero_grad()
            param_sample = self.hyper_posterior.rsample(sample_shape=(self.svi_batch_size,))
            task_pac_bounds, diagnostics_dict = self._task_pac_bounds(task_dicts, param_sample,
                                                               task_kl_weight=self.task_kl_weight,
                                                               meta_kl_weight=self.meta_kl_weight,
                                                               meta_test=True)
            loss = torch.sum(torch.stack(task_pac_bounds))
            loss.backward()
            optimizer.step()

            if itr % log_period == 0 and verbose:
                duration = time.time() - t
                t = time.time()
                message = '\t Meta-Test Iter %d/%d - Loss: %.6f - Time %.2f sec - ' % (itr, n_iter,
                                                                                           loss.item() / n_tasks,
                                                                                           duration)
                # add diagnostics
                message += ' - '.join(['%s: %.4f' % (key, value) for key, value in diagnostics_dict.items()])
                self.logger.info(message)

        return task_dicts

    def _setup_meta_train_step(self, mean_module_str, covar_module_str, mean_nn_layers, kernel_nn_layers, cov_type):
        assert mean_module_str in ['NN', 'constant']
        assert covar_module_str in ['NN', 'SE']

        """ random gp model """
        self.random_gp = RandomGPMeta(size_in=self.input_dim, prior_factor=1.0,
                                  weight_prior_std=self.weight_prior_std, bias_prior_std=self.bias_prior_std,
                                  covar_module_str=covar_module_str, mean_module_str=mean_module_str,
                                  mean_nn_layers=mean_nn_layers, kernel_nn_layers=kernel_nn_layers)

        param_shapes_dict = self.random_gp.parameter_shapes()

        """ variational posterior """
        self.hyper_posterior = RandomGPPosterior(param_shapes_dict, cov_type=cov_type)

        def _tile_data_tuple(task_dict, tile_size):
            x_data, y_data = task_dict['train_x'], task_dict['train_y']
            x_data = x_data.view(torch.Size((1,)) + x_data.shape).repeat(tile_size, 1, 1)
            y_data = y_data.view(torch.Size((1,)) + y_data.shape).repeat(tile_size, 1)
            return x_data, y_data

        def _hyper_kl(prior_param_sample):
            return torch.mean(self.hyper_posterior.log_prob(prior_param_sample) -
                                  self.random_gp.hyper_prior.log_prob(prior_param_sample))

        def _task_pac_bounds(task_dicts, prior_param_sample, task_kl_weight=1.0, meta_kl_weight=1.0, meta_test=False):

            fn = self.random_gp.get_forward_fn(prior_param_sample)

            kl_outer = meta_kl_weight * _hyper_kl(prior_param_sample)

            task_pac_bounds = []
            for task_dict in task_dicts:
                if meta_test:
                    posterior = task_dict["gp_model"](task_dict["train_x"])
                else:
                    posterior = task_dict["gp_model"].variational_distribution() #

                # likelihood
                avg_ll = torch.mean(self.likelihood.expected_log_prob(task_dict["train_y"], posterior))

                # task complexity
                x_data_tiled, y_data_tiled = _tile_data_tuple(task_dict, self.svi_batch_size)
                gp, _ = fn(x_data_tiled, None, prior=True)
                prior = gp.forward(x_data_tiled)

                kl_inner = task_kl_weight * torch.mean(_kl_divergence_safe(posterior.expand(
                    (self.svi_batch_size,)), prior))

                m = torch.tensor(task_dict["train_y"].shape[0], dtype=torch.float32)
                n = torch.tensor(self.n_tasks, dtype=torch.float32)
                task_complexity = torch.sqrt((kl_outer + kl_inner + math.log(2.) +
                                              torch.log(m) + torch.log(n) - torch.log(self.delta)) / (2*(m - 1)))

                diagnostics_dict = {
                    'avg_ll': avg_ll.item(),
                    'kl_outer_weighted': kl_outer.item(),
                    'kl_inner_weighted': kl_inner.item(),
                }

                task_pac_bound = - avg_ll + task_complexity
                task_pac_bounds.append(task_pac_bound)
            return task_pac_bounds, diagnostics_dict

        def _meta_complexity(prior_param_sample, meta_kl_weight=1.0):
            outer_kl = _hyper_kl(prior_param_sample)
            n = torch.tensor(self.n_tasks, dtype=torch.float32)
            return torch.sqrt(meta_kl_weight * outer_kl + math.log(2.) + torch.log(n) - torch.log(self.delta) / (2*(n-1)))

        def _meta_train_pac_bound(task_dicts):
            param_sample = self.hyper_posterior.rsample(sample_shape=(self.svi_batch_size,))

            task_pac_bounds, diagnostics_dict = _task_pac_bounds(task_dicts, param_sample, task_kl_weight=self.task_kl_weight,
                                               meta_kl_weight=self.meta_kl_weight)
            meta_complexity = _meta_complexity(param_sample, meta_kl_weight=self.meta_kl_weight)

            pac_bound = torch.mean(torch.stack(task_pac_bounds)) + meta_complexity
            return pac_bound, diagnostics_dict

        self._task_pac_bounds = _task_pac_bounds
        self._meta_train_pac_bound = _meta_train_pac_bound

    def _setup_optimizer(self, optimizer, lr, lr_decay):
        if optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.meta_train_params, lr=lr)
        elif optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.meta_train_params, lr=lr)
        else:
            raise NotImplementedError('Optimizer must be Adam or SGD')

        if lr_decay < 1.0:
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1000, gamma=lr_decay)
        else:
            self.lr_scheduler = DummyLRScheduler()

    def _vectorize_pred_dist(self, pred_dist):
        # converts a multivariate gaussian into a vectorized univariate gaussian
        return torch.distributions.Normal(pred_dist.mean, pred_dist.stddev)

    def prior_mean(self, x, n_hyperposterior_samples=1000):
        x = (x - self.x_mean) / self.x_std
        assert x.ndim == 1 or (x.ndim == 2 and x.shape[-1] == 1)
        x_data_tiled = np.tile(x.reshape(1, x.shape[0], 1), (n_hyperposterior_samples, 1, 1))
        x_data_tiled = torch.tensor(x_data_tiled, dtype=torch.float32)

        with torch.no_grad():
            param_sample = self.hyper_posterior.rsample(sample_shape=(n_hyperposterior_samples,))
            fn = self.random_gp.get_forward_fn(param_sample)
            gp, _ = fn(x_data_tiled, None, prior=True)
            prior = gp(x_data_tiled)
            mean = torch.mean(prior.mean, axis=0).numpy() * self.y_std + self.y_mean
            # torch.mean(gp.learned_mean(x_data_tiled), axis=0).numpy() * self.y_std + self.y_mean
        return mean

    def _aggregate_gp_priors(self, prior_param_sample, jitter=1e-5):
        assert prior_param_sample.ndim == 2
        n_samples = prior_param_sample.shape[0]
        forward_fn = self.random_gp.get_forward_fn(prior_param_sample)

        def _tile(x_data):
            assert x_data.ndim == 2
            return x_data.view(torch.Size((1,)) + x_data.shape).repeat(n_samples, 1, 1)

        def mean_module(x):
            gp, _ = forward_fn(_tile(x), None, prior=True)
            mean = torch.mean(gp.forward(x).mean, axis=0)
            return mean

        def covar_module(x):
            gp, _ = forward_fn(_tile(x), None, prior=True)
            dist = gp.forward(x)
            mean = torch.mean(dist.mean, axis=0)

            # cov due to difference in individual means
            residual = dist.mean - mean
            # batched outer product of residuals followed by mean over the
            cov_loc = torch.mean(torch.bmm(residual.unsqueeze(2), residual.unsqueeze(1)), axis=0)

            # average of individual covs
            cov_var = torch.mean(dist.covariance_matrix, axis=0)
            return cov_loc + cov_var + 1e-5*torch.eye(cov_var.shape[0])

        return mean_module, covar_module

""" helper functions """

def _kl_divergence_safe(posterior, prior):
    for jitter_eps in [1e-6, 1e-5, 1e-4]:
        try:
            return torch.distributions.kl.kl_divergence(posterior, prior)
        except RuntimeError:
            posterior = _add_jitter(prior, eps=jitter_eps)
            prior = _add_jitter(prior, eps=jitter_eps)
            import warnings
            warnings.warn("added jitter of %s to the diagonal posterior and prior covariance" % str(jitter_eps))
    return torch.distributions.kl.kl_divergence(posterior, prior)

def _add_jitter(distr, eps=1e-6):
    from meta_learn.models import GaussianLikelihoodLight
    jitter = GaussianLikelihoodLight(noise_var=eps * torch.ones((1,)))
    return jitter(distr)

if __name__ == "__main__":
    """ 1) Generate some training data from GP prior """
    from experiments.data_sim import SwissfelDataset

    # data_sim = SwissfelDataset(random_state=np.random.RandomState(26))
    #
    # meta_train_data = data_sim.generate_meta_train_data(n_tasks=50, n_samples=20)
    # meta_test_data = data_sim.generate_meta_test_data(n_tasks=50, n_samples_context=20, n_samples_test=160)

    from experiments.data_sim import provide_data
    meta_train_data, meta_test_data, _ = provide_data(dataset='sin_20')


    NN_LAYERS = (32, 32, 32, 32)

    plot = False
    from matplotlib import pyplot as plt

    if plot:
        for x_train, y_train in meta_train_data:
            plt.scatter(x_train, y_train)
        plt.title('sample from the GP prior')
        plt.show()

    """ 2) Classical mean learning based on mll """

    print('\n ---- GPR PAC meta-learning ---- ')

    torch.set_num_threads(4)

    gp_model = GPRegressionMetaLearnedPAC(meta_train_data, num_iter_fit=20000, task_kl_weight=1.0,
                                          meta_kl_weight=1e-5, lr=1e-3, lr_decay=0.97, posterior_lr_multiplier=5.0,
                                          svi_batch_size=5, task_batch_size=5,
                                          covar_module='NN', mean_module='NN', mean_nn_layers=NN_LAYERS,
                                          kernel_nn_layers=NN_LAYERS, cov_type='diag', normalize_data=True)


    for i in range(2):
        itrs = 0
        gp_model.meta_fit(valid_tuples=meta_test_data[:10], log_period=500, eval_period=10000, n_iter=10000)
        itrs += 10000

        for j in range(1):
            x_plot = np.linspace(-10, 10, num=150)
            prior_mean = gp_model.prior_mean(x_plot)
            plt.plot(x_plot, prior_mean, color='green')

            x_context, t_context, x_test, y_test = meta_test_data[j]
            pred_mean, pred_std = gp_model.predict(x_context, t_context, x_plot, n_iter_meta_test=3000)

            plt.scatter(x_test[:20], y_test[:20], alpha=0.4)
            plt.scatter(x_context, t_context)
            plt.plot(x_plot, pred_mean, color='blue')
            plt.fill_between(x_plot, (pred_mean - pred_std).flatten(), (pred_mean + pred_std).flatten(), alpha=0.2, color='blue')
            plt.title('GPR meta PAC meta-test Nr%i (itrs = %i)' % (j, itrs))
            plt.show()

    ll, rmse, calib_err = gp_model.eval_datasets(meta_test_data, n_iter_meta_test=3000)
    print('ll', ll)
    print('rmse', rmse)
    print('calib_err', calib_err)