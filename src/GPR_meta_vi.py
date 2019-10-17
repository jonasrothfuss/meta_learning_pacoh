import torch
import gpytorch
import time
import pyro

import numpy as np
from pyro.distributions import Normal, InverseGamma, Delta
from pyro.infer import SVI, Trace_ELBO
import torch.nn.functional as F


from src.models import LearnedGPRegressionModel, NeuralNetwork, UnnormalizedExpDist, AffineTransformedDistribution, \
    SEKernelLight, ConstantMeanLight, EqualWeightedMixtureDist
from src.util import _handle_input_dimensionality, get_logger

from pyro.distributions import Normal, Delta, Gamma, LogNormal



class GPRegressionMetaLearnedVI:

    def __init__(self, meta_train_data, lr_params=1e-3, weight_prior_scale=0.1, feature_dim=2,
                 num_iter_fit=50000, covar_module='NN', mean_module='NN', mean_nn_layers=(32, 32), kernel_nn_layers=(32, 32),
                 prior_factor=0.1, optimizer='Adam', svi_batch_size=1000, normalize_data=True, random_seed=None):
        """
        Variational GP classification model (https://arxiv.org/abs/1411.2005) that supports prior learning with
        neural network mean and covariance functions

        Args:
            meta_train_data: list of tuples of ndarrays[(train_x_1, train_t_1), ..., (train_x_n, train_t_n)]
            lr: (float) learning rate for prior parameters
            weight_prior_scale: (float) scale of hyper-prior distribution on NN weights
            feature_dim: (int) output dimensionality of NN feature map for kernel function
            num_iter_fit: (int) number of gradient steps for fitting the parameters
            covar_module: (gpytorch.mean.Kernel) optional kernel module, default: RBF kernel
            mean_module: (gpytorch.mean.Mean) optional mean module, default: ZeroMean
            mean_nn_layers: (tuple) hidden layer sizes of mean NN
            kernel_nn_layers: (tuple) hidden layer sizes of kernel NN
            learning_rate: (float) learning rate for AdamW optimizer
            task_batch_size: (int) batch size for meta training, i.e. number of tasks for computing grads
            optimizer: (str) type of optimizer to use - must be either 'Adam' or 'SGD'
            svi_batch_size (int): number of posterior samples to estimate grads
            normalize_data (bool): whether to normalize the training / context data
            random_seed: (int) seed for pytorch
        """
        self.logger = get_logger()

        assert mean_module in ['NN', 'constant', 'zero'] or isinstance(mean_module, gpytorch.means.Mean)
        assert covar_module in ['NN', 'SE'] or isinstance(covar_module, gpytorch.kernels.Kernel)
        assert optimizer in ['Adam', 'SGD']

        self.lr_params, self.weight_prior_scale, self.feature_dim = lr_params, weight_prior_scale, feature_dim
        self.num_iter_fit, self.prior_factor, self.normalize_data = num_iter_fit, prior_factor, normalize_data

        if random_seed is not None:
            pyro.set_rng_seed(random_seed)

        # normalize data, convert to tensors and store them in self.task_dicts
        self._data_handling(meta_train_data)

        # Setup shared modules
        self._setup_model_guide(mean_module, covar_module, mean_nn_layers, kernel_nn_layers)

        self._setup_optimizer(optimizer, lr_params)

        self.svi = SVI(self.model, self.guide, self.optimizer, num_samples=svi_batch_size, loss=Trace_ELBO())

        self.fitted = False


    def meta_fit(self, valid_tuples=None, verbose=True, log_period=500):
        """
        fits the VI and prior parameters of the  GPC model

        Args:
            valid_tuples: list of valid tuples, i.e. [(test_context_x_1, test_context_t_1, test_x_1, test_t_1), ...]
            verbose: (boolean) whether to print training progress
            log_period (int) number of steps after which to print stats
        """

        assert (valid_tuples is None) or (all([len(valid_tuple) == 4 for valid_tuple in valid_tuples]))


        t = time.time()

        for itr in range(1, self.num_iter_fit + 1):
            loss = self.svi.step(self.meta_train_data_tensors)

            # print training stats stats
            if verbose and (itr == 1 or itr % log_period == 0):
                duration = time.time() - t
                t = time.time()

                message = 'Iter %d/%d - Loss: %.6f - Time %.2f sec' % (itr, self.num_iter_fit, loss, duration)

                # if validation data is provided  -> compute the valid log-likelihood
                if valid_tuples is not None:
                    valid_ll, valid_rmse = self.eval_datasets(valid_tuples, mode='MAP')
                    message += ' - Valid-LL: %.3f - Valid-RMSE: %.3f' % (np.mean(valid_ll), np.mean(valid_rmse))

                self.logger.info(message)

        print("-------- Parameter summary --------")
        with torch.no_grad():
            for param_name, param_value in pyro.get_param_store().named_parameters():
                try:
                    if 'lengthscale' in param_name:
                        print("{:<50}{:<30}{:<30}".format(param_name, param_value[0].item(),
                                                          param_value[1].item()))
                    else:
                        print("{:<50}{:<30}".format(param_name, param_value.item()))
                except:
                    pass

        self.fitted = True


    def predict(self, test_context_x, test_context_t, test_x, n_hyperposterior_samples=100, mode='Bayes',
                return_density=False):
        """
        computes the predictive distribution of the targets p(t|test_x, test_context_x, test_context_t)

        Args:
            test_context_x: (ndarray) context input data for which to compute the posterior
            test_context_x: (ndarray) context targets for which to compute the posterior
            test_x: (ndarray) query input data of shape (n_samples, ndim_x)
            return_density: (bool) whether to return result as mean and std ndarray or as MultivariateNormal pytorch object

        Returns:
            (pred_mean, pred_std) predicted mean and standard deviation corresponding to p(t|test_x, test_context_x, test_context_t)
        """
        assert mode in ['Bayes', 'MAP']

        # 1) data handling and normalization
        test_context_x, test_context_t = _handle_input_dimensionality(test_context_x, test_context_t)
        if test_x.ndim == 1:
            test_x = np.expand_dims(test_x, axis=-1)

        stats_dict = self._compute_normalization_stats(test_context_x, test_context_t)
        test_context_x, test_context_t = self._normalize_data(X=test_context_x, Y=test_context_t, stats_dict=stats_dict)
        test_x = self._normalize_data(X=test_x, stats_dict=stats_dict)

        # 2) convert data to tensors
        test_context_x_tensor = torch.from_numpy(test_context_x).contiguous().float()
        test_context_t_tensor = torch.from_numpy(test_context_t).contiguous().float().flatten()
        test_x_tensor = torch.from_numpy(test_x).float().contiguous()
        context_data = [(test_context_x_tensor, test_context_t_tensor)]

        with torch.no_grad():
            if mode == 'Bayes':
                pred_dists = []
                for _ in range(n_hyperposterior_samples):
                    gps, likelihood = self.guide(context_data, mode_delta=False)
                    gp_model = gps[0]
                    gp_model.eval()
                    likelihood.eval()
                    pred_dist = likelihood(gp_model(test_x_tensor))
                    pred_dist = AffineTransformedDistribution(pred_dist, normalization_mean=stats_dict['y_mean'],
                                                              normalization_std=stats_dict['y_std'])
                    pred_dists.append(pred_dist)

                pred_dist = EqualWeightedMixtureDist(pred_dists)
            else:
                gps, likelihood = self.guide(context_data, mode_delta=True)
                gp_model = gps[0]
                pred_dist = likelihood(gp_model(test_x_tensor))
                pred_dist = AffineTransformedDistribution(pred_dist, normalization_mean=stats_dict['y_mean'],
                                                          normalization_std=stats_dict['y_std'])

        if return_density:
            return pred_dist
        else:
            return pred_dist.mean.numpy(), pred_dist.stddev.numpy()


    def eval(self, test_context_x, test_context_t, test_x, test_t, n_hyperposterior_samples=100, mode='Bayes'):
        """
          computes the predictive distribution of the targets p(t|test_x, test_context_x, test_context_t)

          Args:
              test_context_x: (ndarray) context input data for which to compute the posterior
              test_context_x: (ndarray) context targets for which to compute the posterior
              test_x: (ndarray) query input data of shape (n_samples, ndim_x)
              return_density: (bool) whether to return result as mean and std ndarray or as MultivariateNormal pytorch object

          Returns:
              (pred_mean, pred_std) predicted mean and standard deviation corresponding to p(t|test_x, test_context_x, test_context_t)
        """

        test_context_x, test_context_t = _handle_input_dimensionality(test_context_x, test_context_t)
        test_x, test_t = _handle_input_dimensionality(test_x, test_t)

        with torch.no_grad():
            pred_dist = self.predict(test_context_x, test_context_t, test_x, return_density=True,
                                     n_hyperposterior_samples=n_hyperposterior_samples, mode=mode)

            test_t_tensor = torch.from_numpy(test_t).contiguous().float().flatten()
            avg_log_likelihood = pred_dist.log_prob(test_t_tensor).item() / test_t_tensor.shape[0]
            rmse = torch.mean(torch.pow(pred_dist.mean - test_t_tensor, 2)).sqrt().item()

            return avg_log_likelihood, rmse

    def eval_datasets(self, test_tuples, mode='Bayes'):
        """
        Computes the average test log likelihood and the rmse over multiple test datasets

        Args:
            test_tuples: list of test set tuples, i.e. [(test_context_x_1, test_context_t_1, test_x_1, test_t_1), ...]

        Returns: (avg_log_likelihood, rmse)

        """

        assert (all([len(valid_tuple) == 4 for valid_tuple in test_tuples]))

        ll_rmse_tuples = []
        for test_data_tuple in test_tuples:
            try:
                ll_rmse_tuples.append(self.eval(*test_data_tuple, mode=mode))
            except RuntimeError as e:
                self.logger.warn('skipped one eval dataset due the following error: %s'%str(e))

        ll_list, rmse_list = list(zip(* ll_rmse_tuples ))

        return np.mean(ll_list), np.mean(rmse_list)


    def state_dict(self):
        raise NotImplementedError

    def load_state_dict(self, state_dict):
        raise NotImplementedError

    def _compute_normalization_stats(self, X, Y, stats_dict=None):
        if stats_dict is None:
            stats_dict = {}

        if self.normalize_data:
            # compute mean and std of data for normalization
            stats_dict['x_mean'], stats_dict['y_mean'] = np.mean(X, axis=0), np.mean(Y, axis=0)
            stats_dict['x_std'], stats_dict['y_std'] = np.std(X, axis=0), np.std(Y, axis=0)
        else:
            stats_dict['x_mean'], stats_dict['y_mean'] = np.zeros(X.shape[1]), np.zeros(Y.shape[1])
            stats_dict['x_std'], stats_dict['y_std'] = np.ones(X.shape[1]), np.ones(Y.shape[1])

        return stats_dict


    def _normalize_data(self, X, Y=None, stats_dict=None):
        assert "x_mean" in stats_dict and "x_std" in stats_dict, "requires computing normalization stats beforehand"
        assert "y_mean" in stats_dict and "y_std" in stats_dict, "requires computing normalization stats beforehand"

        X_normalized = (X - stats_dict["x_mean"]) / stats_dict["x_std"]

        if Y is None:
            return X_normalized
        else:
            Y_normalized = (Y - stats_dict["y_mean"]) / stats_dict["y_std"]
            return X_normalized, Y_normalized


    def _data_handling(self, meta_train_data):
        # Check that data all has the same size
        for i in range(len(meta_train_data)):
            meta_train_data[i] = _handle_input_dimensionality(*meta_train_data[i])
        self.input_dim = meta_train_data[0][0].shape[-1]
        assert all([self.input_dim == train_x.shape[-1] for train_x, _ in meta_train_data])

        self.meta_train_data_tensors = []
        for train_x, train_t in meta_train_data:
            # add data stats to task dict
            stats_dict = self._compute_normalization_stats(train_x, train_t, stats_dict={})
            train_x, train_t = self._normalize_data(train_x, train_t, stats_dict=stats_dict)

            # Convert the data into arrays of torch tensors
            train_x_tensor = torch.from_numpy(train_x).contiguous().float()
            train_t_tensor = torch.from_numpy(train_t).contiguous().float().flatten()
            self.meta_train_data_tensors.append((train_x_tensor, train_t_tensor))
        return self.meta_train_data_tensors

    def _setup_model_guide(self, mean_module_str, covar_module_str, mean_nn_layers, kernel_nn_layers):
        _ones = torch.ones(self.input_dim)
        assert mean_module_str in ['NN', 'constant', 'zero']
        assert covar_module_str in ['NN', 'SE']

        prefix = str(self.__hash__())[-6:] + '_'

        if mean_module_str == 'NN':
            mean_nn = NeuralNetwork(input_dim=self.input_dim, output_dim=1,
                                    layer_sizes=mean_nn_layers, prefix=prefix + 'mean_nn_')
        if covar_module_str == 'NN':
            kernel_nn = NeuralNetwork(input_dim=self.input_dim, output_dim=self.feature_dim,
                                      layer_sizes=kernel_nn_layers, prefix=prefix + 'kernel_nn_')

        def model(meta_data_tensors):

            # mean function prior
            if mean_module_str == 'NN':
                lifted_mean_nn = _lift_nn_prior(mean_nn, "mean_nn")
                nn_mean_fn = lifted_mean_nn()
                mean_module = None
            elif covar_module_str == 'constant':
                constant_mean = pyro.sample("constant_mean", Normal(0.0 * _ones, 1.0 * _ones))
                mean_module = ConstantMeanLight(constant_mean)
                nn_mean_fn = None
            else:
                mean_module = gpytorch.means.ZeroMean()
                nn_mean_fn = None

            # kernel function prior

            if covar_module_str == 'NN':
                lifted_kernel_nn = _lift_nn_prior(kernel_nn, "kernel_nn")
                nn_kernel_fn = lifted_kernel_nn()
                kernel_dim = self.feature_dim

                lengthscale = torch.ones(kernel_dim)
                outputscale = 1.0
            else:
                nn_kernel_fn = None
                kernel_dim = self.input_dim

                lengthscale = pyro.sample("lengthscale",
                                          LogNormal(0.0 * torch.ones(kernel_dim), 2.0 * torch.ones(kernel_dim)).to_event(1))
                outputscale = pyro.sample("outputscale", LogNormal(torch.tensor(0.0), torch.tensor(20.0)))

            covar_module = SEKernelLight(lengthscale, outputscale)

            # noise variance prior
            noise_prior = {'noise_covar.raw_noise': Normal(torch.tensor(0.0), torch.tensor(2.0))}
            lifted_likelihood = pyro.random_module('likelihood', gpytorch.likelihoods.GaussianLikelihood(), noise_prior)
            likelihood = lifted_likelihood()

            n_datasets = len(meta_data_tensors)
            mlls_normalized = torch.zeros(n_datasets)
            dataset_sizes = torch.zeros(n_datasets)

            for i, (x_tensor, t_tensor) in enumerate(meta_data_tensors):
                gp_model = LearnedGPRegressionModel(x_tensor, t_tensor, likelihood,
                                                    learned_kernel=nn_kernel_fn, learned_mean=nn_mean_fn,
                                                    covar_module=covar_module, mean_module=mean_module)
                mll_fn = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)
                gp_model.train()
                likelihood.train()

                n_samples = x_tensor.shape[0]
                pred = gp_model(x_tensor)
                mlls_normalized[i] = mll_fn(pred, t_tensor) / n_samples
                dataset_sizes[i] = n_samples

            harmonic_mean_dataset_size = n_datasets / torch.sum(1/dataset_sizes)

            pre_factor = harmonic_mean_dataset_size / (harmonic_mean_dataset_size + n_datasets)
            exponent_fn = lambda t_tensor: (1.0 / self.prior_factor) * pre_factor * torch.sum(mlls_normalized)


            pyro.sample("obs", UnnormalizedExpDist(exponent_fn=exponent_fn), obs=t_tensor)

            return UnnormalizedExpDist(exponent_fn=exponent_fn).log_prob(t_tensor).item()

        def guide(meta_data_tensors, mode_delta=False):
            _delta_wrap = _get_delta_wrap(mode_delta)

            # mean function posterior

            if mean_module_str == 'NN':
                lifted_mean_nn = _lift_nn_posterior(mean_nn, "mean_nn", mode_delta=mode_delta)
                nn_mean_fn = lifted_mean_nn()
                mean_module = None
            elif covar_module_str == 'constant':
                constant_mean_q_loc = pyro.param(prefix + "constant_mean_q_loc", 0.0 * _ones)
                constant_mean_q_scale = pyro.param(prefix + "constant_mean_q_scale", 1.0 * _ones)
                constant_mean = pyro.sample("constant_mean", _delta_wrap(Normal(constant_mean_q_loc, constant_mean_q_scale)))
                mean_module = ConstantMeanLight(constant_mean)
                nn_mean_fn = None
            else:
                mean_module = gpytorch.means.ZeroMean()
                nn_mean_fn = None

            # kernel function posterior
            if covar_module_str == 'NN':
                lifted_kernel_nn = _lift_nn_posterior(kernel_nn, "kernel_nn", mode_delta=mode_delta)
                nn_kernel_fn = lifted_kernel_nn()
                kernel_dim = self.feature_dim

                lengthscale = torch.ones(kernel_dim)
                outputscale = 1.0

            else:
                nn_kernel_fn = None
                kernel_dim = self.input_dim

                lengthscale_q_loc = pyro.param(prefix + "lengthscale_q_loc", 0.0 * torch.ones(kernel_dim))
                lengthscale_q_scale = pyro.param(prefix + "lengthscale_q_scale", 1.0 * torch.ones(kernel_dim))
                lengthscale = pyro.sample("lengthscale", _delta_wrap(LogNormal(lengthscale_q_loc, lengthscale_q_scale)).to_event(1))

                outputscale_q_loc = pyro.param(prefix + "outputscale_q_loc", torch.tensor(0.0))
                outputscale_q_scale = pyro.param(prefix + "outputscale_q_scale", torch.tensor(1.0))
                outputscale = pyro.sample("outputscale", _delta_wrap(LogNormal(outputscale_q_loc, outputscale_q_scale)))

            covar_module = SEKernelLight(lengthscale, outputscale)

            # noise variance posterior
            noise_var_q_loc = pyro.param(prefix + "noise_var_q_loc", torch.tensor([0.0]))
            noise_var_q_scale = pyro.param(prefix + "noise_var_q_scale", torch.tensor([1.0]))
            noise_prior = {'noise_covar.raw_noise': _delta_wrap(Normal(noise_var_q_loc, noise_var_q_scale))}
            lifted_likelihood = pyro.random_module('likelihood', gpytorch.likelihoods.GaussianLikelihood(), noise_prior)
            likelihood = lifted_likelihood()

            gp_models = []
            for i, (x_tensor, t_tensor) in enumerate(meta_data_tensors):
                gp_model = LearnedGPRegressionModel(x_tensor, t_tensor, likelihood,
                                                    learned_kernel=nn_kernel_fn, learned_mean=nn_mean_fn,
                                                    covar_module=covar_module, mean_module=mean_module)
                gp_models.append(gp_model)
            gp_model.eval()
            likelihood.eval()

            return gp_models, likelihood

        self.model = model
        self.guide = guide


    def _setup_optimizer(self, optimizer, lr):
        if optimizer == 'Adam':
            self.optimizer = pyro.optim.Adam({'lr': lr})
        elif optimizer == 'SGD':
            self.optimizer = pyro.optim.SGD({'lr': lr})
        else:
            raise NotImplementedError('Optimizer must be Adam or SGD')


def _lift_nn_prior(net, name, weight_prior_scale=1.0, bias_prior_scale=1000.0):
    priors = {}
    for name, param in net.named_parameters():
        if 'bias' in name:
            priors[name] = Normal(loc=torch.zeros_like(param), scale=bias_prior_scale * torch.ones_like(param)).to_event(1)
        if 'weight' in name:
            priors[name] = Normal(loc=torch.zeros_like(param), scale=weight_prior_scale * torch.ones_like(param)).to_event(2)

    # lift module parameters to random variables sampled from the priors
    return pyro.random_module(name, net, priors)

def _lift_nn_posterior(net, name, mode_delta=False):
    _delta_wrap = _get_delta_wrap(mode_delta)

    vi_posteriors = {}
    for name, param in net.named_parameters():
        assert 'bias' in name or 'weight' in name
        vi_mu_param = pyro.param("%s_mu" % name, torch.randn_like(param))
        vi_sigma_param = F.softplus(pyro.param("%s_sigma_raw" % name, torch.randn_like(param)))
        if 'bias' in name:
            vi_posteriors[name] = _delta_wrap(Normal(loc=vi_mu_param, scale=vi_sigma_param)).to_event(1)
        if 'weight' in name:
            vi_posteriors[name] = _delta_wrap(Normal(loc=vi_mu_param, scale=vi_sigma_param)).to_event(2)

    return pyro.random_module(name, nn_module=net, prior=vi_posteriors)

def _get_delta_wrap(mode_delta=False):
    if mode_delta:
        def _delta_wrap(dist):
            if isinstance(dist, Normal):
                return Delta(dist.mean)
            elif isinstance(dist, LogNormal):
                mode = torch.exp(dist.loc - dist.scale ** 2)
                return Delta(mode)
            else:
                raise NotImplementedError
    else:
        def _delta_wrap(dist):
            return dist
    return _delta_wrap
