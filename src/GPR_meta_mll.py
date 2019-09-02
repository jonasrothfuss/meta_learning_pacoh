import torch
import gpytorch
import time

import numpy as np

from src.models import LearnedGPRegressionModel, NeuralNetwork
from src.util import _handle_input_dimensionality, get_logger


class GPRegressionMetaLearned:

    def __init__(self, meta_train_data, learning_mode='both', lr_params=1e-3, weight_decay=0.0, feature_dim=2,
                 num_iter_fit=1000, covar_module='NN', mean_module='NN', mean_nn_layers=(32, 32), kernel_nn_layers=(32, 32),
                 task_batch_size=5, random_seed=None):
        """
        Variational GP classification model (https://arxiv.org/abs/1411.2005) that supports prior learning with
        neural network mean and covariance functions

        Args:
            meta_train_data: list of tuples of ndarrays[(train_x_1, train_t_1), ..., (train_x_n, train_t_n)]
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
            learning_rate: (float) learning rate for AdamW optimizer
            task_batch_size: (int) batch size for meta training, i.e. number of tasks for computing grads
            random_seed: (int) seed for pytorch
        """
        self.logger = get_logger()

        assert learning_mode in ['learn_mean', 'learn_kernel', 'both', 'vanilla']
        assert mean_module in ['NN', 'constant', 'zero'] or isinstance(mean_module, gpytorch.means.Mean)
        assert covar_module in ['NN', 'SE'] or isinstance(covar_module, gpytorch.kernels.Kernel)

        self.lr_params, self.weight_decay, self.feature_dim = lr_params, weight_decay, feature_dim
        self.num_iter_fit, self.task_batch_size = num_iter_fit, task_batch_size

        if random_seed is not None:
            torch.manual_seed(random_seed)
        self.rds_numpy = np.random.RandomState(random_seed)

        # Check that data all has the same size
        for i in range(len(meta_train_data)):
            meta_train_data[i] = _handle_input_dimensionality(*meta_train_data[i])
        self.input_dim = meta_train_data[0][0].shape[-1]
        assert all([self.input_dim == train_x.shape[-1] for train_x, _ in meta_train_data])


        # Setup shared modules

        self._setup_gp_prior(mean_module, covar_module, learning_mode, feature_dim, mean_nn_layers, kernel_nn_layers)

        # setup tasks models

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

        self.task_dicts = []

        for train_x, train_t in meta_train_data:
            # Convert the data into arrays of torch tensors
            train_x_tensor = torch.from_numpy(train_x).contiguous().float()
            train_t_tensor = torch.from_numpy(train_t).contiguous().float().flatten()

            gp_model = LearnedGPRegressionModel(train_x_tensor, train_t_tensor, self.likelihood,
                                              learned_kernel=self.nn_kernel_map, learned_mean=self.nn_mean_fn,
                                              covar_module=self.covar_module, mean_module=self.mean_module,
                                              feature_dim=feature_dim)
            self.task_dicts.append({
                'train_x': train_x_tensor,
                'train_t': train_t_tensor,
                'model': gp_model,
                'mll_fn': gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, gp_model)
            })

        if len(self.shared_parameters) > 0:
            self.optimizer = torch.optim.AdamW(self.shared_parameters)

        self.fitted = False


    def meta_fit(self, valid_tuples=None, verbose=True, log_period=500):
        """
        fits the VI and prior parameters of the  GPC model

        Args:
            valid_tuples: list of valid tuples, i.e. [(test_context_x_1, test_context_t_1, test_x_1, test_t_1), ...]
            verbose: (boolean) whether to print training progress
            log_period (int) number of steps after which to print stats
        """
        for task_dict in self.task_dicts: task_dict['model'].train()
        self.likelihood.train()

        assert (valid_tuples is None) or (all([len(valid_tuple) == 4 for valid_tuple in valid_tuples]))

        if len(self.shared_parameters) > 0:
            t = time.time()

            for itr in range(1, self.num_iter_fit + 1):

                self.optimizer.zero_grad()
                loss = 0.0

                for task_dict in self.rds_numpy.choice(self.task_dicts, size=self.task_batch_size):

                    output = task_dict['model'](task_dict['train_x'])
                    mll = task_dict['mll_fn'](output, task_dict['train_t'])
                    loss -= mll / task_dict['train_x'].shape[0]

                loss.backward()
                self.optimizer.step()

                # print training stats stats
                if verbose and (itr == 1 or itr % log_period == 0):
                    duration = time.time() - t
                    t = time.time()

                    message = 'Iter %d/%d - Loss: %.3f - Time %.3f sec' % (itr, self.num_iter_fit, loss.item(), duration)

                    # if validation data is provided  -> compute the valid log-likelihood
                    if valid_tuples is not None:
                        self.likelihood.eval()
                        valid_ll, _ = self.eval_datasets(valid_tuples)
                        self.likelihood.train()
                        message += ' - Valid-LL: %.3f' % np.mean(valid_ll)

                    self.logger.info(message)

        else:
            self.logger.info('Vanilla mode - nothing to fit')

        self.fitted = True

        for task_dict in self.task_dicts: task_dict['model'].eval()
        self.likelihood.eval()



    def predict(self, test_context_x, test_context_t, test_x, return_tensors=False):
        """
        computes the predictive distribution of the targets p(t|test_x, test_context_x, test_context_t)

        Args:
            test_context_x: (ndarray) context input data for which to compute the posterior
            test_context_x: (ndarray) context targets for which to compute the posterior
            test_x: (ndarray) query input data of shape (n_samples, ndim_x)
            return_tensors: (bool) whether to return result as torch tensors of ndarray

        Returns:
            (pred_mean, pred_std) predicted mean and standard deviation corresponding to p(t|test_x, test_context_x, test_context_t)
        """

        test_context_x, test_context_t = _handle_input_dimensionality(test_context_x, test_context_t)
        if test_x.ndim == 1:
            test_x = np.expand_dims(test_x, axis=-1)
        assert test_x.shape[1] == test_context_x.shape[1]

        with torch.no_grad():
            test_context_x_tensor = torch.from_numpy(test_context_x).contiguous().float()
            test_context_t_tensor = torch.from_numpy(test_context_t).contiguous().float().flatten()

            # compute posterior given the context data
            gp_model = LearnedGPRegressionModel(test_context_x_tensor, test_context_t_tensor, self.likelihood,
                                                learned_kernel=self.nn_kernel_map, learned_mean=self.nn_mean_fn,
                                                covar_module=self.covar_module, mean_module=self.mean_module,
                                                feature_dim=self.feature_dim)
            gp_model.eval()
            self.likelihood.eval()
            test_x_tensor = torch.from_numpy(test_x).contiguous().float()
            pred = self.likelihood(gp_model(test_x_tensor))
            pred_mean = pred.mean
            pred_std = pred.stddev

        if return_tensors:
            return pred_mean, pred_std
        else:
            return pred_mean.numpy(), pred_std.numpy()


    def eval(self, test_context_x, test_context_t, test_x, test_t):
        """
        Computes the average test log likelihood and the rmse on test data

        Args:
            test_x: (ndarray) test input data of shape (n_samples, ndim_x)
            test_t: (ndarray) test target data of shape (n_samples, 1)

        Returns: (avg_log_likelihood, rmse)

        """

        test_context_x, test_context_t = _handle_input_dimensionality(test_context_x, test_context_t)
        test_x, test_t = _handle_input_dimensionality(test_x, test_t)

        with torch.no_grad():
            pred_mean, pred_std = self.predict(test_context_x, test_context_t, test_x, return_tensors=True)

            test_t_tensor = torch.from_numpy(test_t).contiguous().float().flatten()

            pred_dist = torch.distributions.normal.Normal(loc=pred_mean, scale=pred_std)
            avg_log_likelihood = torch.mean(pred_dist.log_prob(test_t_tensor))
            rmse = torch.mean(torch.pow(pred_mean - test_t_tensor, 2)).sqrt()

            return avg_log_likelihood.item(), rmse.item()

    def eval_datasets(self, test_tuples):
        """
        Computes the average test log likelihood and the rmse over multiple test datasets

        Args:
            test_tuples: list of test set tuples, i.e. [(test_context_x_1, test_context_t_1, test_x_1, test_t_1), ...]

        Returns: (avg_log_likelihood, rmse)

        """

        assert (all([len(valid_tuple) == 4 for valid_tuple in test_tuples]))

        ll_list, rmse_list = list(zip(*[self.eval(*test_data_tuple) for test_data_tuple in test_tuples]))

        return np.mean(ll_list), np.mean(rmse_list)


    def _setup_gp_prior(self, mean_module, covar_module, learning_mode, feature_dim, mean_nn_layers, kernel_nn_layers):

        self.shared_parameters = []

        # a) determine kernel map & module
        if covar_module == 'NN':
            assert learning_mode in ['learn_kernel', 'both'], 'neural network parameters must be learned'
            self.nn_kernel_map = NeuralNetwork(input_dim=self.input_dim, output_dim=feature_dim,
                                          layer_sizes=kernel_nn_layers)
            self.shared_parameters.append(
                {'params': self.nn_kernel_map.parameters(), 'lr': self.lr_params, 'weight_decay': self.weight_decay})
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=feature_dim))
        else:
            self.nn_kernel_map = None

        if covar_module == 'SE':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=feature_dim))
        elif isinstance(covar_module, gpytorch.kernels.Kernel):
            self.covar_module = covar_module

        # b) determine mean map & module

        if mean_module == 'NN':
            assert learning_mode in ['learn_mean', 'both'], 'neural network parameters must be learned'
            self.nn_mean_fn = NeuralNetwork(input_dim=self.input_dim, output_dim=1, layer_sizes=mean_nn_layers)
            self.shared_parameters.append(
                {'params': self.nn_mean_fn.parameters(), 'lr': self.lr_params, 'weight_decay': self.weight_decay})
            self.mean_module = None
        else:
            self.nn_mean_fn = None

        if mean_module == 'constant':
            self.mean_module = gpytorch.means.ConstantMean()
        elif mean_module == 'zero':
            self.mean_module = gpytorch.means.ZeroMean()
        elif isinstance(mean_module, gpytorch.means.Mean):
            self.mean_module = mean_module

        # c) add parameters of covar and mean module if desired

        if learning_mode in ["learn_kernel", "both"]:
            self.shared_parameters.append({'params': self.covar_module.hyperparameters(), 'lr': self.lr_params})

        if learning_mode in ["learn_mean", "both"] and self.mean_module is not None:
            self.shared_parameters.append({'params': self.mean_module.hyperparameters(), 'lr': self.lr_params})


