import copy
import gpytorch
import torch
import math
import torch.nn.functional as F
from pyro.distributions import Normal, LogNormal, Independent
from collections import OrderedDict

from src.models import LearnedGPRegressionModel, ConstantMeanLight, SEKernelLight, GaussianLikelihoodLight, \
    VectorizedModel, CatDist, NeuralNetworkVectorized
from config import device


def _filter(dict, str):
    result = OrderedDict()
    for key, val in dict.items():
        if str in key:
            result[key] = val
    return result


class VectorizedGP(VectorizedModel):

    def __init__(self, input_dim, feature_dim=2, covar_module_str='SE', mean_module_str='constant',
                 mean_nn_layers=(32, 32), kernel_nn_layers=(32, 32), nonlinearlity=torch.tanh):
        super().__init__(input_dim, 1)


        self._params = OrderedDict()
        self.mean_module_str = mean_module_str
        self.covar_module_str = covar_module_str

        if mean_module_str == 'NN':
            self.mean_nn = self._param_module('mean_nn', NeuralNetworkVectorized(input_dim, 1,
                                                         layer_sizes=mean_nn_layers, nonlinearlity=nonlinearlity))
        elif mean_module_str == 'constant':
            self.constant_mean = self._param('constant_mean', torch.zeros(1, 1))
        else:
            raise NotImplementedError


        if covar_module_str == "NN":
            self.kernel_nn = self._param_module('kernel_nn', NeuralNetworkVectorized(input_dim, feature_dim,
                                                        layer_sizes=kernel_nn_layers, nonlinearlity=nonlinearlity))
            self.lengthscale_raw = self._param('lengthscale_raw', torch.zeros(1, feature_dim))
        elif covar_module_str == 'SE':
            self.lengthscale_raw = self._param('lengthscale_raw', torch.zeros(1, input_dim))
        else:
            raise NotImplementedError

        self.noise_raw = self._param('noise_raw', torch.zeros(1, 1))


    def forward(self, x_data, y_data, train=True):
        assert x_data.ndim == 3

        if self.mean_module_str == 'NN':
            learned_mean = self.mean_nn
            mean_module = None
        else:
            learned_mean = None
            mean_module = ConstantMeanLight(self.constant_mean)

        if self.covar_module_str == "NN":
            learned_kernel = self.kernel_nn
        else:
            learned_kernel = None

        lengthscale = F.softplus(self.lengthscale_raw)
        lengthscale = lengthscale.view(lengthscale.shape[0], 1, lengthscale.shape[1])
        covar_module = SEKernelLight(lengthscale)

        noise = F.softplus(self.noise_raw)
        likelihood = GaussianLikelihoodLight(noise)
        gp = LearnedGPRegressionModel(x_data, y_data, likelihood, mean_module=mean_module, covar_module=covar_module,
                                      learned_mean=learned_mean, learned_kernel=learned_kernel)
        if train:
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp)
            output = gp(x_data)
            return likelihood(output), mll(output, y_data)
        else: # --> eval
            gp.eval()
            likelihood.eval()
            return gp, likelihood

    def parameter_shapes(self):
        return OrderedDict([(name, param.shape) for name, param in self.named_parameters().items()])

    def named_parameters(self):
        return self._params

    def _param_module(self, name, module):
        assert type(name) == str
        assert hasattr(module, 'named_parameters')
        for param_name, param in module.named_parameters().items():
            self._param(name + '.' + param_name, param)
        return module

    def _param(self, name, tensor):
        assert type(name) == str
        assert isinstance(tensor, torch.Tensor)
        assert name not in list(self._params.keys())
        if not device.type == tensor.device.type:
            tensor = tensor.to(device)
        self._params[name] = tensor
        return tensor

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class _RandomGPBase:

    def __init__(self, size_in, prior_factor=1.0, weight_prior_std=1.0, bias_prior_std=3.0, **kwargs):

        self._params = OrderedDict()
        self._param_dists = OrderedDict()

        self.prior_factor = prior_factor
        self.gp = VectorizedGP(size_in, **kwargs)

        for name, shape in self.gp.parameter_shapes().items():

            if name == 'constant_mean':
                mean_p_loc = torch.zeros(1).to(device)
                mean_p_scale = torch.ones(1).to(device)
                self._param_dist(name, Normal(mean_p_loc, mean_p_scale).to_event(1))

            if name == 'lengthscale_raw':
                lengthscale_p_loc = torch.zeros(shape[-1]).to(device)
                lengthscale_p_scale = torch.ones(shape[-1]).to(device)
                self._param_dist(name, Normal(lengthscale_p_loc, lengthscale_p_scale).to_event(1))

            if name == 'noise_raw':
                noise_p_loc = -1. * torch.ones(1).to(device)
                noise_p_scale = torch.ones(1).to(device)
                self._param_dist(name, Normal(noise_p_loc, noise_p_scale).to_event(1))

            if 'mean_nn' in name or 'kernel_nn' in name:
                mean = torch.zeros(shape).to(device)
                if "weight" in name:
                    std = weight_prior_std * torch.ones(shape).to(device)
                elif "bias" in name:
                    std = bias_prior_std * torch.ones(shape).to(device)
                else:
                    raise NotImplementedError
                self._param_dist(name, Normal(mean, std).to_event(1))

        # check that parameters in prior and gp modules are aligned
        for param_name_gp, param_name_prior in zip(self.gp.named_parameters().keys(), self._param_dists.keys()):
            assert param_name_gp == param_name_prior

        self.hyper_prior = CatDist(self._param_dists.values())

    def sample_params_from_prior(self, shape=torch.Size()):
        return self.hyper_prior.sample(shape)

    def sample_fn_from_prior(self, shape=torch.Size()):
        params = self.sample_params_from_prior(shape=shape)
        return self.get_forward_fn(params)

    def get_forward_fn(self, params):
        gp_model = copy.deepcopy(self.gp)
        gp_model.set_parameters_as_vector(params)
        return gp_model

    def _param_dist(self, name, dist):
        assert type(name) == str
        assert isinstance(dist, torch.distributions.Distribution)
        assert name not in list(self._param_dists.keys())
        assert hasattr(dist, 'rsample')
        self._param_dists[name] = dist
        return dist

    def _log_prob_prior(self, params):
        return self.hyper_prior.log_prob(params)

    def _log_prob_likelihood(self, *args):
        raise NotImplementedError

    def log_prob(self, *args):
        raise NotImplementedError

    def parameter_shapes(self):
        param_shapes_dict = OrderedDict()
        for name, dist in self._param_dists.items():
            param_shapes_dict[name] = dist.event_shape
        return param_shapes_dict

class RandomGP(_RandomGPBase):

    def _log_prob_likelihood(self, params, x_data, y_data):
        fn = self.get_forward_fn(params)
        _, mll = fn(x_data, y_data)
        return mll

    def log_prob(self, params, x_data, y_data):
        return self.prior_factor * self._log_prob_prior(params) + self._log_prob_likelihood(params, x_data, y_data)

class RandomGPMeta(_RandomGPBase):

    def _log_prob_likelihood(self, params, train_data_tuples):
        fn = self.get_forward_fn(params)

        num_datasets = len(train_data_tuples)
        dataset_sizes = torch.tensor([train_x.shape[-2] for train_x, _ in train_data_tuples]).float().to(device)
        harmonic_mean_dataset_size = 1. / (torch.mean(1. / dataset_sizes))
        pre_factor = harmonic_mean_dataset_size / (harmonic_mean_dataset_size + num_datasets)

        mlls_normalized = []
        for i, (x_data, y_data) in enumerate(train_data_tuples):
            _, mll = fn(x_data, y_data)
            mlls_normalized.append(mll / dataset_sizes[i])
        mlls_normalized = torch.stack(mlls_normalized, dim=-1)
        return pre_factor * torch.sum(mlls_normalized, dim=-1)

    def log_prob(self, params, train_data_tuples):
        return self.prior_factor * self._log_prob_prior(params) + self._log_prob_likelihood(params, train_data_tuples)

class RandomGPPosterior(torch.nn.Module):
    """
    Gaussian VI posterior on the GP-Prior parameters
    """

    def __init__(self, named_param_shapes, init_std=0.1, cov_type='full'):
        super().__init__()

        assert cov_type in ['diag', 'full']

        self.param_idx_ranges = OrderedDict()

        idx_start = 0
        for name, shape in named_param_shapes.items():
            assert len(shape) == 1
            idx_end = idx_start + shape[0]
            self.param_idx_ranges[name] = (idx_start, idx_end)
            idx_start = idx_end

        param_shape = torch.Size((idx_start,))
        self.loc = torch.nn.Parameter(torch.normal(0.0, init_std, size=param_shape, device=device))

        if cov_type == 'diag':
            self.scale = torch.nn.Parameter(torch.normal(math.log(0.1), init_std, size=param_shape, device=device))
            self.dist_fn = lambda: Normal(self.loc, self.scale.exp()).to_event(1)
        if cov_type == 'full':
            self.tril_cov = torch.nn.Parameter(torch.diag(torch.ones(param_shape, device=device).uniform_(0.05, 0.1)))
            self.dist_fn = lambda: torch.distributions.MultivariateNormal(loc=self.loc, scale_tril=torch.tril(self.tril_cov))

    def forward(self):
        return self.dist_fn()

    def rsample(self, sample_shape=torch.Size()):
        return self.forward().rsample(sample_shape)

    def sample(self, sample_shape=torch.Size()):
        return self.forward().sample(sample_shape)

    def log_prob(self, value):
        return self.forward().log_prob(value)

    @property
    def mode(self):
        return self.mean

    @property
    def mean(self):
        return self.forward().mean

    @property
    def stddev(self):
        return self.forward().stddev

    def entropy(self):
        return self.forward().entropy()

    @property
    def mean_stddev_dict(self):
        mean = self.mean
        stddev = self.stddev
        with torch.no_grad():
            return OrderedDict(
                [(name, (mean[idx_start:idx_end], stddev[idx_start:idx_end])) for name, (idx_start, idx_end) in self.param_idx_ranges.items()])


def _get_base_dist(dist):
    if isinstance(dist, Independent):
        return _get_base_dist(dist.base_dist)
    else:
        return dist