"""
Models for meta learning Gaussian Processes in GPyTorch.
Author: Vincent Fortuin
Copyright 2018 ETH Zurich
"""

import torch
import gpytorch

from gpytorch.models import AbstractVariationalGP, ExactGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy


class LearnedMean(torch.nn.Sequential):
    """Trainable neural network mean function for GPs."""
    def __init__(self, input_dim=2, num_layers=2, layer_width=64):
        super(LearnedMean, self).__init__()
        self.add_module('linear_1', torch.nn.Linear(input_dim, layer_width))
        self.add_module('Tanh_1', torch.nn.Tanh())
        for i in range(2,num_layers):
            self.add_module("linear_{}".format(i), torch.nn.Linear(layer_width, layer_width))
            self.add_module("Tanh_{}".format(i).format(i), torch.nn.Tanh())
        self.add_module("linear_{}".format(num_layers), torch.nn.Linear(layer_width, int(layer_width/2)))
        self.add_module("Tanh_{}".format(num_layers), torch.nn.Tanh())
        self.add_module("linear_{}".format(num_layers+1), torch.nn.Linear(int(layer_width/2), 1))


class LearnedKernel(torch.nn.Sequential):
    """Trainable neural network kernel function for GPs."""
    def __init__(self, input_dim=2, num_layers=2, layer_width=64):
        super(LearnedKernel, self).__init__()
        self.add_module('linear_1', torch.nn.Linear(input_dim, layer_width))
        self.add_module('Tanh_1', torch.nn.Tanh())
        for i in range(2, num_layers):
            self.add_module("linear_{}".format(i), torch.nn.Linear(layer_width, layer_width))
            self.add_module("Tanh_{}".format(i).format(i), torch.nn.Tanh())
        self.add_module("linear_{}".format(num_layers), torch.nn.Linear(layer_width, int(layer_width/2)))
        self.add_module("Tanh_{}".format(num_layers), torch.nn.Tanh())
        self.add_module("linear_{}".format(num_layers+1), torch.nn.Linear(int(layer_width/2), 2))


class LearnedGPRegressionModel(ExactGP):
    """GP model which can take a learned mean and learned kernel function."""
    def __init__(self, train_x, train_y, likelihood, learned_kernel=None, learned_mean=None, input_dim=2):
        super(LearnedGPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=input_dim))
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

# TODO: merge functionality in an abstract base class
class LearnedGPClassificationModel(gpytorch.models.AbstractVariationalGP):

    def __init__(self, train_x, learned_kernel=None, learned_mean=None, input_dim=2):
        variational_distribution = CholeskyVariationalDistribution(train_x.size(0))
        variational_strategy = VariationalStrategy(self, train_x, variational_distribution)
        super(LearnedGPClassificationModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=input_dim))
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
