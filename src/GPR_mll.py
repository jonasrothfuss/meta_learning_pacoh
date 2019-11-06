import torch
import gpytorch
import time
import numpy as np

from src.models import LearnedGPRegressionModel, NeuralNetwork, AffineTransformedDistribution
from src.abstract import RegressionModel
from src.util import _handle_input_dimensionality


class GPRegressionLearned(RegressionModel):

    def __init__(self, train_x, train_t, learning_mode='both', lr=1e-3, weight_decay=0.0, feature_dim=2,
                 num_iter_fit=1000, covar_module='NN', mean_module='NN', mean_nn_layers=(32, 32), kernel_nn_layers=(32, 32),
                 optimizer='Adam', normalize_data=True, lr_scheduler=True, random_seed=None):
        """
        Variational GP classification model (https://arxiv.org/abs/1411.2005) that supports prior learning with
        neural network mean and covariance functions

        Args:
            train_x: (ndarray) train inputs - shape: (n_sampls, ndim_x)
            train_t: (ndarray) train targets - shape: (n_sampls, 1)
            learning_mode: (str) specifying which of the GP prior parameters to optimize. Either one of
                    ['learned_mean', 'learned_kernel', 'both', 'vanilla']
            lr: (float) learning rate for prior parameters
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
        super().__init__(normalize_data=normalize_data, random_seed=random_seed)

        assert learning_mode in ['learn_mean', 'learn_kernel', 'both', 'vanilla']
        assert mean_module in ['NN', 'constant', 'zero'] or isinstance(mean_module, gpytorch.means.Mean)
        assert covar_module in ['NN', 'SE'] or isinstance(covar_module, gpytorch.kernels.Kernel)
        assert optimizer in ['Adam', 'SGD']

        self.lr, self.weight_decay, self.num_iter_fit, self.lr_scheduler = lr, weight_decay, num_iter_fit, lr_scheduler

        """ ------ Data handling ------ """
        self.train_x_tensor, self.train_t_tensor = self.initial_data_handling(train_x, train_t)

        """  ------ Setup model ------ """
        self.parameters = []

        # A) determine kernel map & module

        if covar_module == 'NN':
            assert learning_mode in ['learn_kernel', 'both'], 'neural network parameters must be learned'
            nn_kernel_map = NeuralNetwork(input_dim=self.input_dim, output_dim=feature_dim, layer_sizes=kernel_nn_layers)
            self.parameters.append({'params': nn_kernel_map.parameters(), 'lr': self.lr, 'weight_decay': self.weight_decay})
            covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=feature_dim))
        else:
            nn_kernel_map = None

        if covar_module == 'SE':
            covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=feature_dim))

        # B) determine mean map & module

        if mean_module == 'NN':
            assert learning_mode in ['learn_mean', 'both'], 'neural network parameters must be learned'
            nn_mean_fn = NeuralNetwork(input_dim=self.input_dim, output_dim=1, layer_sizes=mean_nn_layers)
            self.parameters.append({'params': nn_mean_fn.parameters(), 'lr': self.lr, 'weight_decay': self.weight_decay})
            mean_module = None
        else:
            nn_mean_fn = None

        if mean_module == 'constant':
            mean_module = gpytorch.means.ConstantMean()
        elif mean_module == 'zero':
            mean_module = gpytorch.means.ZeroMean()

        # C) setup GP model

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.parameters.append({'params': self.likelihood.parameters(), 'lr': self.lr})

        self.model = LearnedGPRegressionModel(self.train_x_tensor, self.train_t_tensor, self.likelihood,
                                              learned_kernel=nn_kernel_map, learned_mean=nn_mean_fn,
                                              covar_module=covar_module, mean_module=mean_module)

        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)


        # D) determine which parameters are trained and setup optimizer

        if learning_mode in ["learn_kernel", "both"]:
            self.parameters.append({'params': self.model.covar_module.hyperparameters(), 'lr': self.lr})

        if learning_mode in ["learn_mean", "both"] and mean_module is not None:
            self.parameters.append({'params': self.model.mean_module.hyperparameters(), 'lr': self.lr})

        if optimizer == 'Adam':
            self.optimizer = torch.optim.AdamW(self.parameters)
        elif optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.parameters)
        else:
            raise NotImplementedError('Optimizer must be Adam or SGD')

        if self.lr_scheduler:
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.2)
        else:  # factor 1.0 --> no lr decay
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=1.0)

        self.fitted = False

    def fit(self, verbose=True, valid_x=None, valid_t=None, log_period=500):
        """
        fits prior parameters of the  GPC model by maximizing the mll of the training data

        Args:
            verbose: (boolean) whether to print training progress
            valid_x: (np.ndarray) validation inputs - shape: (n_sampls, ndim_x)
            valid_y: (np.ndarray) validation targets - shape: (n_sampls, 1)
            log_period: (int) number of steps after which to print stats
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
                if itr == 1 or itr % log_period == 0:
                    duration = time.time() - t
                    t = time.time()

                    message = 'Iter %d/%d - Loss: %.3f - Time %.3f sec' % (itr, self.num_iter_fit, loss.item(), duration)

                    # if validation data is provided  -> compute the valid log-likelihood
                    if valid_x is not None:
                        self.model.eval()
                        self.likelihood.eval()
                        valid_ll, valid_rmse = self.eval(valid_x, valid_t)
                        self.lr_scheduler.step(valid_ll)
                        self.model.train()
                        self.likelihood.train()

                        message += ' - Valid-LL: %.3f - Valid-RMSE %.3f' %(valid_ll, valid_rmse)

                    if verbose:
                        self.logger.info(message)

        else:
            self.logger.info('Vanilla mode - nothing to fit')

        self.fitted = True

        self.model.eval()
        self.likelihood.eval()

    def predict(self, test_x, return_density=False):
        """
        computes the predictive distribution of the targets p(t|test_x, train_x, train_y)

        Args:
            test_x: (ndarray) query input data of shape (n_samples, ndim_x)
            return_density (bool) whether to return a density object or

        Returns:
            (pred_mean, pred_std) predicted mean and standard deviation corresponding to p(y_test|X_test, X_train, y_train)
        """
        with torch.no_grad():
            test_x_normalized = self._normalize_data(test_x)
            test_x_tensor = torch.from_numpy(test_x_normalized).contiguous().float()

            pred_dist = self.likelihood(self.model(test_x_tensor))
            pred_dist_transformed = AffineTransformedDistribution(pred_dist, normalization_mean=self.y_mean,
                                                                  normalization_std=self.y_std)
            if return_density:
                return pred_dist_transformed
            else:
                pred_mean = pred_dist_transformed.mean.numpy()
                pred_std = pred_dist_transformed.stddev.numpy()
                return pred_mean, pred_std


    def eval(self, test_x, test_t):
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
            pred_dist = self.predict(test_x, return_density=True)
            avg_log_likelihood = pred_dist.log_prob(test_t_tensor) / test_t_tensor.shape[0]
            rmse = torch.mean(torch.pow(pred_dist.mean - test_t_tensor, 2)).sqrt()

            return avg_log_likelihood.item(), rmse.item()

    def state_dict(self):
        state_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        return state_dict

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])

