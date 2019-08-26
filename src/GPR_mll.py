import torch
import gpytorch
import time

import numpy as np

from src.models import LearnedGPRegressionModel, NeuralNetwork


class GPRegressionLearned:

    def __init__(self, train_x, train_t, learning_mode='both', lr_params=1e-3, lr_vi=1e-2, weight_decay=1e-3, feature_dim=2,
                 num_iter_fit=5000, covar_module='NN', mean_module='NN', mean_nn_layers=(64, 64), kernel_nn_layers=(64, 64)):
        """
        Variational GP classification model (https://arxiv.org/abs/1411.2005) that supports prior learning with
        neural network mean and covariance functions

        Args:
            train_x: (ndarray) train inputs - shape: (n_sampls, ndim_x)
            train_t: (ndarray) train targets - shape: (n_sampls, 1)
            learning_mode: (str) specifying how to parametrize the prior. Either one of
                    ['learned_mean', 'learned_kernel', 'both', 'vanilla']
            lr_params: (float) learning rate for prior parameters
            lr_vi: (float) learning rate for variational parameters, i.e. params of Gaussian q(u)
            weight_decay: (float) weight decay penalty
            feature_dim: (int) output dimensionality of NN feature map for kernel function
            num_iter_fit: (int) number of gradient steps for fitting the parameters
            covar_module: (gpytorch.mean.Kernel) optional kernel module, default: RBF kernel
            mean_module: (gpytorch.mean.Mean) optional mean module, default: ZeroMean
            mean_nn_layers: (tuple) hidden layer sizes of mean NN
            kernel_nn_layers: (tuple) hidden layer sizes of kernel NN
        """

        assert learning_mode in ['learn_mean', 'learn_kernel', 'both', 'vanilla']
        assert mean_module in ['NN', 'constant', 'zero'] or isinstance(mean_module, gpytorch.means.Mean)
        assert covar_module in ['NN', 'SE'] or isinstance(covar_module, gpytorch.kernels.Kernel)
        assert lr_params <= lr_vi, "parameter learning rate should be smaller than VI learning rate"

        self.lr_params, self.lr_vi, self.weight_decay, self.num_iter_fit = lr_params, lr_vi, weight_decay, num_iter_fit

        # Convert the data into pytorch tensors
        train_x, train_t = _handle_input_dimensionality(train_x, train_t)

        input_dim = train_x.shape[-1]
        self.train_x_tensor = torch.from_numpy(train_x).contiguous().float()
        self.train_t_tensor = torch.from_numpy(train_t).contiguous().float().flatten()

        ## --- Setup model --- ##
        self.parameters = []

        # a) determine kernel map & module

        if covar_module is 'NN':
            assert learning_mode in ['learn_kernel', 'both'], 'neural network parameters must be learned'
            nn_kernel_map = NeuralNetwork(input_dim=input_dim, output_dim=feature_dim, layer_sizes=kernel_nn_layers)
            self.parameters.append({'params': nn_kernel_map.parameters(), 'lr': self.lr_params, 'weight_decay': self.weight_decay})
            covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=feature_dim))
        else:
            nn_kernel_map = None

        if covar_module is 'SE':
            covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=feature_dim))


        # b) determine mean map & module

        if mean_module is 'NN':
            assert learning_mode in ['learn_mean', 'both'], 'neural network parameters must be learned'
            nn_mean_fn = NeuralNetwork(input_dim=input_dim, output_dim=1, layer_sizes=mean_nn_layers)
            self.parameters.append({'params': nn_mean_fn.parameters(), 'lr': self.lr_params, 'weight_decay': self.weight_decay})
            mean_module = None
        else:
            nn_mean_fn = None

        if mean_module is 'constant':
            mean_module = gpytorch.means.ConstantMean()
        elif mean_module is 'zero':
            mean_module = gpytorch.means.ZeroMean()

        # C) setup GP model

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

        self.model = LearnedGPRegressionModel(self.train_x_tensor, self.train_t_tensor, self.likelihood,
                                              learned_kernel=nn_kernel_map, learned_mean=nn_mean_fn,
                                              covar_module=covar_module, mean_module=mean_module,
                                              feature_dim=feature_dim)



        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)


        # D) determine which parameters and setup optimizer

        if learning_mode in ["learn_kernel", "both"]:
            self.parameters.append({'params': self.model.covar_module.hyperparameters(), 'lr': self.lr_params})

        if learning_mode in ["learn_mean", "both"] and mean_module is not None:
            self.parameters.append({'params': self.model.mean_module.hyperparameters(), 'lr': self.lr_params})

        if len(self.parameters) > 0:
            self.optimizer = torch.optim.Adam(self.parameters)

        self.fitted = False

    def fit(self, verbose=True, valid_x=None, valid_t=None):
        """
        fits the VI and prior parameters of the  GPC model

        Args:
            verbose: (boolean) whether to print training progress
            valid_x: (np.ndarray) validation inputs - shape: (n_sampls, ndim_x)
            valid_y: (np.ndarray) validation targets - shape: (n_sampls, 1)
        """
        self.model.train()
        self.likelihood.train()

        assert (valid_x is None and valid_t is None) or (isinstance(valid_x, np.ndarray) and isinstance(valid_x, np.ndarray))

        if len(self.parameters) > 0:
            t = time.time()

            for itr in range(1, self.num_iter_fit + 1):

                self.optimizer.zero_grad()
                output = self.model(self.train_x_tensor)
                loss = -self.mll(output, self.train_t_tensor)
                loss.backward()
                self.optimizer.step()

                # print training stats stats
                if verbose and (itr == 1 or itr % 100 == 0):
                    duration = time.time() - t
                    t = time.time()

                    message = 'Iter %d/%d - Loss: %.3f - Time %.3f sec' % (itr, self.num_iter_fit, loss.item(), duration)

                    # if validation data is provided  -> compute the valid log-likelihood
                    if valid_x is not None:
                        self.model.eval()
                        self.likelihood.eval()
                        valid_ll, _ = self.eval(valid_x, valid_t)
                        self.model.train()
                        self.likelihood.train()

                        message += ' - Valid-LL: %.3f' % valid_ll

                    print(message)

        else:
            print('Vanilla mode - nothing to fit')

        self.fitted = True

        self.model.eval()
        self.likelihood.eval()

    def predict(self, test_x):
        """
        computes class probabilities and predictions

        Args:
            test_x: (ndarray) query input data of shape (n_samples, ndim_x)

        Returns:
            (pred_mean, pred_std) predicted mean and standard deviation corresponding to p(y_test|X_test, X_train, y_train)
        """
        with torch.no_grad():
            test_x_tensor = torch.from_numpy(test_x).contiguous().float().flatten()
            pred = self.likelihood(self.model(test_x_tensor))
            pred_mean = pred.mean
            pred_std = pred.stddev

        return pred_mean.numpy(), pred_std.numpy()


    def eval(self, test_x, test_t):
        """
        Computes the average test log likelihood and the RSME on test data

        Args:
            test_x: (ndarray) test input data of shape (n_samples, ndim_x)
            test_t: (ndarray) test target data of shape (n_samples, 1)

        Returns: (avg_log_likelihood, rmse)

        """
        with torch.no_grad():
            test_x_tensor = torch.from_numpy(test_x).contiguous().float()
            test_t_tensor = torch.from_numpy(test_t).contiguous().float().flatten()

            pred = self.likelihood(self.model(test_x_tensor))

            pred_dist = torch.distributions.normal.Normal(loc=pred.mean, scale=pred.stddev)
            avg_log_likelihood = torch.mean(pred_dist.log_prob(test_t_tensor))
            rmse = torch.mean(torch.pow(pred.mean - test_t_tensor, 2)).sqrt()

            return avg_log_likelihood.item(), rmse.item()


""" helper functions """

def _handle_input_dimensionality(x: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray):

    if x.ndim == 1:
        x = np.expand_dims(x, -1)
    if y.ndim == 1:
        y = np.expand_dims(y, -1)

    assert x.shape[0] == y.shape[0]
    assert x.ndim == 2
    assert y.ndim == 2

    return x, y


