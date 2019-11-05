import torch
import gpytorch
import math
from collections import OrderedDict

from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy

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
        self.loc_tensor = torch.tensor(normalization_mean).float().reshape((1,))
        self.scale_tensor = torch.tensor(normalization_std).float().reshape((1,))
        normalization_tranform = AffineTransform(loc=self.loc_tensor, scale=self.scale_tensor)
        super().__init__(base_dist, normalization_tranform)

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

    def __init__(self, dists):
        self.dists = dists
        self.n_dists = len(dists)
        super().__init__()

    @property
    def mean(self):
        return torch.mean(torch.stack([dist.mean for dist in self.dists], dim=0), dim=0)

    @property
    def stddev(self):
        return torch.sqrt(self.variance)

    @property
    def variance(self):
        means = torch.stack([dist.mean for dist in self.dists], dim=0)
        var1 = torch.mean((means - torch.mean(means, dim=0))**2, dim=0)
        var2 = torch.mean(torch.stack([dist.variance for dist in self.dists], dim=0), dim=0)

        # check shape
        original_shape = self.dists[0].mean.shape
        assert var1.shape == var2.shape == original_shape

        return var1 + var2

    @property
    def arg_constraints(self):
        return {}

    def log_prob(self, value):
        log_probs_dists = torch.stack([dist.log_prob(value) for dist in self.dists])
        return torch.logsumexp(log_probs_dists, dim=0) - torch.log(torch.tensor(self.n_dists).float())


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

""" ----------------------------------------------------"""
""" ------------ Vectorized Neural Network -------------"""
""" ----------------------------------------------------"""

import torch.nn as nn
import torch.nn.functional as F

class LinearVectorized:
    def __init__(self, size_in, size_out):
        self.size_in = size_in
        self.size_out = size_out

        self.weight = torch.normal(0, 1, size=(size_in * size_out,), requires_grad=True)
        self.bias = torch.zeros(size_out, requires_grad=True)

        self.reset_parameters()

    def reset_parameters(self):
        self.weight = _kaiming_uniform_batched(self.weight, fan=self.size_in, a=math.sqrt(5), nonlinearity='tanh')
        if self.bias is not None:
            fan_in = self.size_out
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        if self.weight.ndim == 2 or self.weight.ndim == 3:
            model_batch_size = self.weight.shape[0]
            # batched computation
            if self.weight.ndim == 3:
                assert self.weight.shape[-2] == 1 and self.bias.shape[-2] == 1

            W = self.weight.view(model_batch_size, self.size_out, self.size_in)
            b = self.bias.view(model_batch_size, self.size_out)

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
            return F.linear(x, self.weight.view(self.size_out, self.size_in), self.bias)
        else:
            raise NotImplementedError

    def named_parameters(self):
        return {'weight': self.weight, 'bias': self.bias}

    def parameters(self):
        return list(self.named_parameters().values())

    def set_parameter(self, name, value):
        if len(name.split('.')) == 1:
            setattr(self, name, value)
        else:
            remaining_name = ".".join(name.split('.')[1:])
            getattr(self, name.split('.')[0]).set_parameter(remaining_name, value)

    def __call__(self, *args, **kwargs):
        return self.forward( *args, **kwargs)

class NeuralNetworkVectorized:
    """Trainable neural network that batches multiple sets of parameters. That is, each
    """
    def __init__(self, input_dim=2, output_dim=2, layer_sizes=(64, 64), nonlinearlity=torch.tanh):
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

    def parameters(self):
        return list(self.named_parameters().values())

    def set_parameter(self, name, value):
        if len(name.split('.')) == 1:
            setattr(self, name, value)
        else:
            remaining_name = ".".join(name.split('.')[1:])
            getattr(self, name.split('.')[0]).set_parameter(remaining_name, value)

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


class ConstantMeanLight(Mean):
    def __init__(self, constant=torch.ones(1), batch_shape=torch.Size()):
        super(ConstantMeanLight, self).__init__()
        self.batch_shape = batch_shape
        self.constant = constant

    def forward(self, input):
        if input.shape[:-2] == self.batch_shape:
            return self.constant.expand(input.shape[:-1])
        else:
            return self.constant.expand(_mul_broadcast_shape(input.shape[:-1], self.constant.shape))

class SEKernelLight(Kernel):

    def __init__(self, lengthscale=torch.tensor([1.0]), output_scale=torch.tensor(1.0)):
        super(SEKernelLight, self).__init__()
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



# TODO: merge functionality in an abstract base class
class LearnedGPClassificationModel(gpytorch.models.AbstractVariationalGP):

    def __init__(self, train_x, learned_kernel=None, learned_mean=None, mean_module=None, covar_module=None,
                 feature_dim=2):

        variational_distribution = CholeskyVariationalDistribution(train_x.size(0))
        variational_strategy = VariationalStrategy(self, train_x, variational_distribution)
        super(LearnedGPClassificationModel, self).__init__(variational_strategy)

        if mean_module is None:
            self.mean_module = gpytorch.means.ZeroMean()
        else:
            self.mean_module = mean_module

        if covar_module is None:
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=feature_dim))
        else:
            self.covar_module = covar_module

        self.learned_kernel = learned_kernel
        self.learned_mean = learned_mean

    def forward(self, x):
        # feed through kernel NN
        if self.learned_kernel is not None:
            projected_x = self.learned_kernel(x)
        else:
            projected_x = x

        # feed through mean NN
        if self.learned_mean is not None:
            mean_x = self.learned_mean(x).squeeze()
        else:
            mean_x = self.mean_module(projected_x)

        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
