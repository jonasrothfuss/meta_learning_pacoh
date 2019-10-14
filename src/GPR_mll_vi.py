import torch
import gpytorch
import time
import numpy as np

import torch.nn.functional as F

from src.models import LearnedGPRegressionModel, NeuralNetwork, UnnormalizedExpDist, AffineTransformedDistribution, \
    SEKernelLight, ConstantMeanLight, EqualWeightedMixtureDist
from src.util import _handle_input_dimensionality, get_logger

import pyro
from pyro.distributions import Normal, InverseGamma, Delta, Gamma, LogNormal
from pyro.infer import SVI, Trace_ELBO


class GPRegressionLearnedVI:

    def __init__(self, train_x, train_t, lr_params=1e-3, num_iter_fit=10000, prior_factor=0.01, feature_dim=2,
                 covar_module='NN', mean_module='NN', mean_nn_layers=(32, 32), kernel_nn_layers=(32, 32),
                 optimizer='Adam', svi_batch_size=100, normalize_data=True, random_seed=None):
        """
        Variational GP classification model (https://arxiv.org/abs/1411.2005) that supports prior learning with
        neural network mean and covariance functions

        Args:
            train_x: (ndarray) train inputs - shape: (n_sampls, ndim_x)
            train_t: (ndarray) train targets - shape: (n_sampls, 1)
            learning_mode: (str) specifying which of the GP prior parameters to optimize. Either one of
                    ['learned_mean', 'learned_kernel', 'both', 'vanilla']
            lr_params: (float) learning rate for prior parameters
            weight_decay: (float) weight decay penalty
            feature_dim: (int) output dimensionality of NN feature map for kernel function
            num_iter_fit: (int) number of gradient steps for fitting the parameters
            covar_module: (gpytorch.mean.Kernel) optional kernel module, default: RBF kernel
            mean_module: (gpytorch.mean.Mean) optional mean module, default: ZeroMean
            mean_nn_layers: (tuple) hidden layer sizes of mean NN
            kernel_nn_layers: (tuple) hidden layer sizes of kernel NN
            optimizer: (str) type of optimizer to use - must be either 'Adam' or 'SGD'
            random_seed: (int) seed for pytorch
        """
        self.logger = get_logger()
        self.num_iter_fit, self.prior_factor, self.feature_dim = num_iter_fit, prior_factor, feature_dim
        self.normalize_data = normalize_data

        if random_seed is not None:
            pyro.set_rng_seed(random_seed)

        # normalize data and convert to tensors
        self._data_handling(train_x, train_t)

        # setup model and guide
        self._setup_model_guide(mean_module, covar_module, mean_nn_layers, kernel_nn_layers)

        # setup inference procedure
        self._setup_optimizer(optimizer, lr_params)

        self.svi = SVI(self.model, self.guide, self.optimizer, num_samples=svi_batch_size, loss=Trace_ELBO())

        self.fitted = False


    def fit(self, verbose=True, valid_x=None, valid_t=None, log_period=1000):
        """
        fits prior parameters of the  GPC model by maximizing the mll of the training data

        Args:
            verbose: (boolean) whether to print training progress
            valid_x: (np.ndarray) validation inputs - shape: (n_samples, ndim_x)
            valid_y: (np.ndarray) validation targets - shape: (n_samples, 1)
            log_period: (int) number of steps after which to print stats
        """

        assert (valid_x is None and valid_t is None) or (isinstance(valid_x, np.ndarray) and isinstance(valid_x, np.ndarray))

        t = time.time()

        for itr in range(1, self.num_iter_fit + 1):

            loss = self.svi.step(self.train_x_tensor, self.train_t_tensor)

            # print training stats stats
            if verbose and (itr == 1 or itr % log_period == 0):
                duration = time.time() - t
                t = time.time()

                message = 'Iter %d/%d - Loss: %.3f - Time %.3f sec' % (itr, self.num_iter_fit, loss, duration)

                # if validation data is provided  -> compute the valid log-likelihood
                if valid_x is not None:
                    valid_ll, rmse = self.eval(valid_x, valid_t)
                    message += ' - Valid-LL: %.3f - Valid-RMSE: %.3f' % (valid_ll, rmse)

                self.logger.info(message)

        print("-------- Parameter summary --------")
        for param_name, param_value in pyro.get_param_store().named_parameters():
            try:
                print("{:<50}{:<30}".format(param_name, param_value.item()))
            except:
                pass

        self.fitted = True


    def predict(self, test_x, n_posterior_samples=100, mode='Bayes', return_density=False):
        """
        computes the predictive distribution of the targets p(t|test_x, train_x, train_y)

        Args:
            test_x: (ndarray) query input data of shape (n_samples, ndim_x)
            n_posterior_samples: (int) number of samples from posterior to average over
        Returns:
            (pred_mean, pred_std) predicted mean and standard deviation corresponding to p(y_test|X_test, X_train, y_train)
        """
        assert mode in ['Bayes', 'MAP']

        with torch.no_grad():
            test_x_normalized = self._normalize_data(test_x)
            test_x_tensor = torch.from_numpy(test_x_normalized).contiguous().float()

            if mode == 'Bayes':
                pred_dists = []
                for i in range(n_posterior_samples):
                    gp_model, likelihood = self.guide(self.train_x_tensor, self.train_t_tensor)
                    pred_dist = likelihood(gp_model(test_x_tensor))
                    pred_dists.append(AffineTransformedDistribution(pred_dist, normalization_mean=self.y_mean,
                                                                          normalization_std=self.y_std))

                pred_dist = EqualWeightedMixtureDist(pred_dists)
            else:
                gp_model, likelihood = self.guide(self.train_x_tensor, self.train_t_tensor, mode_delta=True)
                pred_dist = likelihood(gp_model(test_x_tensor))
                pred_dist = AffineTransformedDistribution(pred_dist, normalization_mean=self.y_mean, normalization_std=self.y_std)


            if return_density:
                return pred_dist
            else:
                pred_mean = pred_dist.mean.numpy()
                pred_std = pred_dist.stddev.numpy()
                return pred_mean, pred_std


    def eval(self, test_x, test_t, n_posterior_samples=100, mode='Bayes'):
        """
        Computes the average test log likelihood and the rmse on test data

        Args:
            test_x: (ndarray) test input data of shape (n_samples, ndim_x)
            test_t: (ndarray) test target data of shape (n_samples, 1)

        Returns: (avg_log_likelihood, rmse)

        """

        # convert to tensors
        test_x, test_t = _handle_input_dimensionality(test_x, test_t)
        test_t_tensor = torch.from_numpy(test_t).contiguous().float().flatten()

        with torch.no_grad():
            pred_dist = self.predict(test_x, n_posterior_samples=n_posterior_samples, mode=mode, return_density=True)
            avg_log_likelihood = pred_dist.log_prob(test_t_tensor) / test_t_tensor.shape[0]
            rmse = torch.mean(torch.pow(pred_dist.mean - test_t_tensor, 2)).sqrt()

            return avg_log_likelihood.item(), rmse.item()


    def state_dict(self): #TODO
        state_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        return state_dict

    def load_state_dict(self, state_dict): #TODO
        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])

    def _compute_normalization_stats(self, X, Y):
        # save mean and variance of data for normalization
        if self.normalize_data:
            self.x_mean, self.y_mean = np.mean(X, axis=0), np.mean(Y, axis=0)
            self.x_std, self.y_std = np.std(X, axis=0), np.std(Y, axis=0)
        else:
            self.x_mean, self.y_mean = np.zeros(X.shape[1]), np.zeros(Y.shape[1])
            self.x_std, self.y_std = np.ones(X.shape[1]), np.ones(Y.shape[1])

    def _normalize_data(self, X, Y=None):
        assert hasattr(self, "x_mean") and hasattr(self, "x_std"), "requires computing normalization stats beforehand"
        assert hasattr(self, "y_mean") and hasattr(self, "y_std"), "requires computing normalization stats beforehand"

        X_normalized = (X - self.x_mean) / self.x_std

        if Y is None:
            return X_normalized
        else:
            Y_normalized = (Y - self.y_mean) / self.y_std
            return X_normalized, Y_normalized

    def _data_handling(self, train_x, train_t):
        # a) Check shape and bring data in 2d-tensor formal
        train_x, train_t = _handle_input_dimensionality(train_x, train_t)
        self.input_dim = train_x.shape[-1]

        # b) normalize data to exhibit zero mean and variance
        self._compute_normalization_stats(train_x, train_t)
        train_x_normalized, train_t_normalized = self._normalize_data(train_x, train_t)

        # c) Convert the data into pytorch tensors
        self.train_x_tensor = torch.from_numpy(train_x_normalized).contiguous().float()
        self.train_t_tensor = torch.from_numpy(train_t_normalized).contiguous().float().flatten()

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

        def model(x_tensor, t_tensor):

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

            gp_model = LearnedGPRegressionModel(x_tensor, t_tensor, likelihood,
                                                learned_kernel=nn_kernel_fn, learned_mean=nn_mean_fn,
                                                covar_module=covar_module, mean_module=mean_module)
            mll_fn = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)
            gp_model.train()
            likelihood.train()

            pred = gp_model(x_tensor)
            exponent_fn = lambda t_tensor: (1.0 / self.prior_factor) * mll_fn(pred, t_tensor)

            pyro.sample("obs", UnnormalizedExpDist(exponent_fn=exponent_fn), obs=t_tensor)

            return UnnormalizedExpDist(exponent_fn=exponent_fn).log_prob(t_tensor).item()

        def guide(x_tensor, t_tensor, mode_delta=False):
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

            gp_model = LearnedGPRegressionModel(x_tensor, t_tensor, likelihood,
                                                learned_kernel=nn_kernel_fn, learned_mean=nn_mean_fn,
                                                covar_module=covar_module, mean_module=mean_module)
            gp_model.eval()
            likelihood.eval()

            return gp_model, likelihood

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
