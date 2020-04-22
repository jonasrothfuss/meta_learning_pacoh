import torch
import gpytorch
import math
from collections import OrderedDict
from config import device
from meta_learn.util import find_root_by_bounding

""" ----------------------------------------------------"""
""" ------------ Probability Distributions ------------ """
""" ----------------------------------------------------"""

from torch.distributions import Distribution
from torch.distributions import TransformedDistribution, AffineTransform

class AffineTransformedDistribution(TransformedDistribution):
    r"""
    Implements an affine transformation of a probability distribution p(x)

    x_transformed = mean + std * x , x \sim p(x)

    Args:
        base_dist: (torch.distributions.Distribution) probability distribution to transform
        normalization_mean: (np.ndarray) additive factor to add to x
        normalization_std: (np.ndarray) multiplicative factor for scaling x
    """

    def __init__(self, base_dist, normalization_mean, normalization_std):
        self.loc_tensor = torch.tensor(normalization_mean).float().reshape((1,)).to(device)
        self.scale_tensor = torch.tensor(normalization_std).float().reshape((1,)).to(device)
        normalization_transform = AffineTransform(loc=self.loc_tensor, scale=self.scale_tensor)
        super().__init__(base_dist, normalization_transform)

    @property
    def mean(self):
        return self.transforms[0](self.base_dist.mean)

    @property
    def stddev(self):
        return torch.exp(torch.log(self.base_dist.stddev) + torch.log(self.scale_tensor))

    @property
    def variance(self):
        return torch.exp(torch.log(self.base_dist.variance) + 2 * torch.log(self.scale_tensor))

class UnnormalizedExpDist(Distribution):
    r"""
    Creates a an unnormalized distribution with density function with
    density proportional to exp(exponent_fn(value))

    Args:
      exponent_fn: callable that outputs the exponent
    """

    def __init__(self, exponent_fn):
        self.exponent_fn = exponent_fn
        super().__init__()

    @property
    def arg_constraints(self):
        return {}

    def log_prob(self, value):
        return self.exponent_fn(value)

class FactorizedNormal(Distribution):

    def __init__(self, loc, scale, summation_axis=-1):
        self.normal_dist = torch.distributions.Normal(loc, scale)
        self.summation_axis = summation_axis

    def log_prob(self, value):
        return torch.sum(self.normal_dist.log_prob(value), dim=self.summation_axis)

class EqualWeightedMixtureDist(Distribution):

    def __init__(self, dists, batched=False, num_dists=None):
        self.batched = batched
        if batched:
            assert isinstance(dists, torch.distributions.Distribution)
            self.num_dists = dists.batch_shape if num_dists is None else num_dists
            event_shape = dists.event_shape
        else:
            assert all([isinstance(d, torch.distributions.Distribution) for d in dists])
            event_shape = dists[0].event_shape
            self.num_dists = len(dists)
        self.dists = dists

        super().__init__(event_shape=event_shape)

    @property
    def mean(self):
        if self.batched:
            return torch.mean(self.dists.mean, dim=0)
        else:
            return torch.mean(torch.stack([dist.mean for dist in self.dists], dim=0), dim=0)

    @property
    def stddev(self):
        return torch.sqrt(self.variance)

    @property
    def variance(self):
        if self.batched:
            means = self.dists.mean
            vars = self.dists.variance
        else:
            means = torch.stack([dist.mean for dist in self.dists], dim=0)
            vars = torch.stack([dist.variance for dist in self.dists], dim=0)

        var1 = torch.mean((means - torch.mean(means, dim=0))**2, dim=0)
        var2 = torch.mean(vars, dim=0)

        # check shape
        assert var1.shape == var2.shape
        return var1 + var2

    @property
    def arg_constraints(self):
        return {}

    def log_prob(self, value):
        if self.batched:
            log_probs_dists = self.dists.log_prob(value)
        else:
            log_probs_dists = torch.stack([dist.log_prob(value) for dist in self.dists])
        return torch.logsumexp(log_probs_dists, dim=0) - torch.log(torch.tensor(self.num_dists).float())

    def cdf(self, value):
        if self.batched:
            cum_p = self.dists.cdf(value)
        else:
            cum_p = torch.stack([dist.cdf(value) for dist in self.dists])
        assert cum_p.shape[0] == self.num_dists
        return torch.mean(cum_p, dim=0)

    def icdf(self, quantile):
        left = - 1e8 * torch.ones(quantile.shape)
        right = + 1e8 * torch.ones(quantile.shape)
        fun = lambda x: self.cdf(x) - quantile
        return find_root_by_bounding(fun, left, right)



class CatDist(Distribution):

    def __init__(self, dists, reduce_event_dim=True):
        assert all([len(dist.event_shape) == 1 for dist in dists])
        assert all([len(dist.batch_shape) == 0 for dist in dists])
        self.reduce_event_dim = reduce_event_dim
        self.dists = dists
        self._event_shape = torch.Size((sum([dist.event_shape[0] for dist in self.dists]),))

    def sample(self, sample_shape=torch.Size()):
        return self._sample(sample_shape, sample_fn='sample')

    def rsample(self, sample_shape=torch.Size()):
        return self._sample(sample_shape, sample_fn='rsample')

    def log_prob(self, value):
        idx = 0
        log_probs = []
        for dist in self.dists:
            n = dist.event_shape[0]
            if value.ndim == 1:
                val = value[idx:idx+n]
            elif value.ndim == 2:
                val = value[:, idx:idx + n]
            elif value.ndim == 2:
                val = value[:, :, idx:idx + n]
            else:
                raise NotImplementedError('Can only handle values up to 3 dimensions')
            log_probs.append(dist.log_prob(val))
            idx += n

        for i in range(len(log_probs)):
            if log_probs[i].ndim == 0:
                log_probs[i] = log_probs[i].reshape((1,))

        if self.reduce_event_dim:
            return torch.sum(torch.stack(log_probs, dim=0), dim=0)
        return torch.stack(log_probs, dim=0)

    def _sample(self, sample_shape, sample_fn='sample'):
        return torch.cat([getattr(d, sample_fn)(sample_shape) for d in self.dists], dim=-1)

""" ----------------------------------------------------"""
""" ------------------ Neural Network ------------------"""
""" ----------------------------------------------------"""

class NeuralNetwork(torch.nn.Sequential):
    """Trainable neural network kernel function for GPs."""
    def __init__(self, input_dim=2, output_dim=2, layer_sizes=(64, 64), nonlinearlity=torch.tanh,
                 weight_norm=False, prefix='',):
        super(NeuralNetwork, self).__init__()
        self.nonlinearlity = nonlinearlity
        self.n_layers = len(layer_sizes)
        self.prefix = prefix

        if weight_norm:
            _normalize = torch.nn.utils.weight_norm
        else:
            _normalize = lambda x: x

        self.layers = []
        prev_size = input_dim
        for i, size in enumerate(layer_sizes):
            setattr(self, self.prefix + 'fc_%i'%(i+1), _normalize(torch.nn.Linear(prev_size, size)))
            prev_size = size
        setattr(self, self.prefix + 'out', _normalize(torch.nn.Linear(prev_size, output_dim)))

    def forward(self, x):
        output = x
        for i in range(1, self.n_layers+1):
            output = getattr(self, self.prefix + 'fc_%i'%i)(output)
            output = self.nonlinearlity(output)
        output = getattr(self, self.prefix + 'out')(output)
        return output

    def forward_parametrized(self, x, params):
        output = x
        param_idx = 0
        for i in range(1, self.n_layers + 1):
            output = F.linear(output, params[param_idx], params[param_idx+1])
            output = self.nonlinearlity(output)
            param_idx += 2
        output = F.linear(output, params[param_idx], params[param_idx+1])
        return output

""" ----------------------------------------------------"""
""" ------------ Vectorized Neural Network -------------"""
""" ----------------------------------------------------"""

import torch.nn as nn
import torch.nn.functional as F


class VectorizedModel:

    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

    def parameter_shapes(self):
        raise NotImplementedError

    def named_parameters(self):
        raise NotImplementedError

    def parameters(self):
        return list(self.named_parameters().values())

    def set_parameter(self, name, value):
        if len(name.split('.')) == 1:
            setattr(self, name, value)
        else:
            remaining_name = ".".join(name.split('.')[1:])
            getattr(self, name.split('.')[0]).set_parameter(remaining_name, value)

    def set_parameters(self, param_dict):
        for name, value in param_dict.items():
            self.set_parameter(name, value)

    def parameters_as_vector(self):
        return torch.cat(self.parameters(), dim=-1)

    def set_parameters_as_vector(self, value):
        idx = 0
        for name, shape in self.parameter_shapes().items():
            idx_next = idx + shape[-1]
            if value.ndim == 1:
                self.set_parameter(name, value[idx:idx_next])
            elif value.ndim == 2:
                self.set_parameter(name, value[:, idx:idx_next])
            else:
                raise AssertionError
            idx = idx_next
        assert idx_next == value.shape[-1]

class LinearVectorized(VectorizedModel):
    def __init__(self, input_dim, output_dim):
        super().__init__(input_dim, output_dim)

        self.weight = torch.normal(0, 1, size=(input_dim * output_dim,), device=device, requires_grad=True)
        self.bias = torch.zeros(output_dim, device=device, requires_grad=True)

        self.reset_parameters()

    def reset_parameters(self):
        self.weight = _kaiming_uniform_batched(self.weight, fan=self.input_dim, a=math.sqrt(5), nonlinearity='tanh')
        if self.bias is not None:
            fan_in = self.output_dim
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        if self.weight.ndim == 2 or self.weight.ndim == 3:
            model_batch_size = self.weight.shape[0]
            # batched computation
            if self.weight.ndim == 3:
                assert self.weight.shape[-2] == 1 and self.bias.shape[-2] == 1

            W = self.weight.view(model_batch_size, self.output_dim, self.input_dim)
            b = self.bias.view(model_batch_size, self.output_dim)

            if x.ndim == 2:
                # introduce new dimension 0
                x = torch.reshape(x, (1, x.shape[0], x.shape[1]))
                # tile dimension 0 to model_batch size
                x = x.repeat(model_batch_size, 1, 1)
            else:
                assert x.ndim == 3 and x.shape[0] == model_batch_size
            # out dimensions correspond to [nn_batch_size, data_batch_size, out_features)
            return torch.bmm(x, W.permute(0, 2, 1)) + b[:, None, :]
        elif self.weight.ndim == 1:
            return F.linear(x, self.weight.view(self.output_dim, self.input_dim), self.bias)
        else:
            raise NotImplementedError

    def parameter_shapes(self):
        return OrderedDict(bias=self.bias.shape, weight=self.weight.shape)

    def named_parameters(self):
        return OrderedDict(bias=self.bias, weight=self.weight)

    def __call__(self, *args, **kwargs):
        return self.forward( *args, **kwargs)

class NeuralNetworkVectorized(VectorizedModel):
    """Trainable neural network that batches multiple sets of parameters. That is, each
    """
    def __init__(self, input_dim, output_dim, layer_sizes=(64, 64), nonlinearlity=torch.tanh):
        super().__init__(input_dim, output_dim)

        self.nonlinearlity = nonlinearlity
        self.n_layers = len(layer_sizes)

        prev_size = input_dim
        for i, size in enumerate(layer_sizes):
            setattr(self, 'fc_%i'%(i+1), LinearVectorized(prev_size, size))
            prev_size = size
        setattr(self, 'out', LinearVectorized(prev_size, output_dim))

    def forward(self, x):
        output = x
        for i in range(1, self.n_layers + 1):
            output = getattr(self, 'fc_%i' % i)(output)
            output = self.nonlinearlity(output)
        output = getattr(self, 'out')(output)
        return output

    def parameter_shapes(self):
        param_dict = OrderedDict()

        # hidden layers
        for i in range(1, self.n_layers + 1):
            layer_name = 'fc_%i' % i
            for name, param in getattr(self, layer_name).parameter_shapes().items():
                param_dict[layer_name + '.' + name] = param

        # last layer
        layer_name = 'out'
        for name, param in getattr(self, layer_name).parameter_shapes().items():
            param_dict[layer_name + '.' + name] = param

        return param_dict

    def named_parameters(self):
        param_dict = OrderedDict()

        # hidden layers
        for i in range(1, self.n_layers + 1):
            layer_name = 'fc_%i' % i
            for name, param in getattr(self, layer_name).named_parameters().items():
                param_dict[layer_name + '.' + name] = param

        # last layer
        layer_name = 'out'
        for name, param in getattr(self, layer_name).named_parameters().items():
            param_dict[layer_name + '.' + name] = param

        return param_dict

    def __call__(self, *args, **kwargs):
        return self.forward( *args, **kwargs)

""" Initialization Helpers """

def _kaiming_uniform_batched(tensor, fan, a=0.0, nonlinearity='tanh'):
    gain = nn.init.calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


""" ----------------------------------------------------"""
""" ------------------ GP components -------------------"""
""" ----------------------------------------------------"""

from gpytorch.means import Mean
from gpytorch.kernels import Kernel
from gpytorch.functions import RBFCovariance
from gpytorch.utils.broadcasting import _mul_broadcast_shape


class ConstantMeanLight(gpytorch.means.Mean):
    def __init__(self, constant=torch.ones(1), batch_shape=torch.Size()):
        super(ConstantMeanLight, self).__init__()
        self.batch_shape = batch_shape
        self.constant = constant

    def forward(self, input):
        if input.shape[:-2] == self.batch_shape:
            return self.constant.expand(input.shape[:-1])
        else:
            return self.constant.expand(_mul_broadcast_shape(input.shape[:-1], self.constant.shape))

class SEKernelLight(gpytorch.kernels.Kernel):

    def __init__(self, lengthscale=torch.tensor([1.0]), output_scale=torch.tensor(1.0)):
        super(SEKernelLight, self).__init__(batch_shape=(lengthscale.shape[0], ))
        self.length_scale = lengthscale
        self.ard_num_dims = lengthscale.shape[-1]
        self.output_scale = output_scale
        self.postprocess_rbf = lambda dist_mat: self.output_scale * dist_mat.div_(-2).exp_()


    def forward(self, x1, x2, diag=False, **params):
        if (
                x1.requires_grad
                or x2.requires_grad
                or (self.ard_num_dims is not None and self.ard_num_dims > 1)
                or diag
        ):
            x1_ = x1.div(self.length_scale)
            x2_ = x2.div(self.length_scale)
            return self.covar_dist(x1_, x2_, square_dist=True, diag=diag,
                                   dist_postprocess_func=self.postprocess_rbf,
                                   postprocess=True, **params)
        return self.output_scale * RBFCovariance().apply(x1, x2, self.length_scale,
                                     lambda x1, x2: self.covar_dist(x1, x2,
                                                                    square_dist=True,
                                                                    diag=False,
                                                                    dist_postprocess_func=self.postprocess_rbf,
                                                                    postprocess=False,
                                                                    **params))

class HomoskedasticNoiseLight(gpytorch.likelihoods.noise_models._HomoskedasticNoiseBase):

    def __init__(self, noise_var, *params, **kwargs):
        self.noise_var = noise_var
        self._modules = {}
        self._parameters = {}

    @property
    def noise(self):
        return self.noise_var

    @noise.setter
    def noise(self, value):
        self.noise_var = value

class GaussianLikelihoodLight(gpytorch.likelihoods._GaussianLikelihoodBase):


    def __init__(self, noise_var, batch_shape=torch.Size()):
        self.batch_shape = batch_shape
        self._modules = {}
        self._parameters = {}

        noise_covar = HomoskedasticNoiseLight(noise_var)
        super().__init__(noise_covar=noise_covar)

    @property
    def noise(self):
        return self.noise_covar.noise

    @noise.setter
    def noise(self, value):
        self.noise_covar.noise = value

    def expected_log_prob(self, target, input, *params, **kwargs):
        mean, variance = input.mean, input.variance
        noise = self.noise_covar.noise

        res = ((target - mean) ** 2 + variance) / noise + noise.log() + math.log(2 * math.pi)
        return res.mul(-0.5).sum(-1)

class LearnedGPRegressionModel(gpytorch.models.ExactGP):
    """GP model which can take a learned mean and learned kernel function."""
    def __init__(self, train_x, train_y, likelihood, learned_kernel=None, learned_mean=None, mean_module=None, covar_module=None):
        super(LearnedGPRegressionModel, self).__init__(train_x, train_y, likelihood)

        if mean_module is None:
            self.mean_module = gpytorch.means.ZeroMean()
        else:
            self.mean_module = mean_module

        self.covar_module = covar_module

        self.learned_kernel = learned_kernel
        self.learned_mean = learned_mean
        self.likelihood = likelihood

    def forward(self, x):
        # feed through kernel NN
        if self.learned_kernel is not None:
            projected_x = self.learned_kernel(x)
        else:
            projected_x = x

        # feed through mean module
        if self.learned_mean is not None:
            mean_x = self.learned_mean(x).squeeze()
        else:
            mean_x = self.mean_module(projected_x).squeeze()

        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def prior(self, x):
        self.train()
        return self.__call__(x)

    def posterior(self, x):
        self.eval()
        return self.__call__(x)

    def kl(self, x):
        return torch.distributions.kl.kl_divergence(self.posterior(x), self.prior(x))

    def pred_dist(self, x):
        self.eval()
        return self.likelihood(self.__call__(x))

    def pred_ll(self, x, y):
        pred_dist = self.pred_dist(x)
        return pred_dist.log_prob(y)


from gpytorch.models.approximate_gp import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy

class LearnedGPRegressionModelApproximate(ApproximateGP):
    """GP model which can take a learned mean and learned kernel function."""
    def __init__(self, train_x, train_y, likelihood, learned_kernel=None, learned_mean=None, mean_module=None,
                 covar_module=None, beta=1.0):

        self.beta = beta
        self.n_train_samples = train_x.shape[0]

        variational_distribution = CholeskyVariationalDistribution(self.n_train_samples)
        variational_strategy = VariationalStrategy(self, train_x, variational_distribution,
                                                   learn_inducing_locations=False)
        super().__init__(variational_strategy)

        if mean_module is None:
            self.mean_module = gpytorch.means.ZeroMean()
        else:
            self.mean_module = mean_module

        self.covar_module = covar_module

        self.learned_kernel = learned_kernel
        self.learned_mean = learned_mean
        self.likelihood = likelihood

    def forward(self, x):
        # feed through kernel NN
        if self.learned_kernel is not None:
            projected_x = self.learned_kernel(x)
        else:
            projected_x = x

        # feed through mean module
        if self.learned_mean is not None:
            mean_x = self.learned_mean(x).squeeze()
        else:
            mean_x = self.mean_module(projected_x).squeeze()

        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def kl(self):
        return self.variational_strategy.kl_divergence()

    def pred_dist(self, x):
        self.eval()
        return self.likelihood(self.__call__(x))

    def pred_ll(self, x, y):
        variational_dist_f = self.__call__(x)
        return self.likelihood.expected_log_prob(y, variational_dist_f).sum(-1)