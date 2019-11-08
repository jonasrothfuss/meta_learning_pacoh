import torch
import gpytorch
import time

import numpy as np

from src.models import LearnedGPRegressionModel, NeuralNetwork, AffineTransformedDistribution
from src.util import _handle_input_dimensionality, get_logger
from config import device

class GPRegressionMetaLearned:

    def __init__(self, meta_train_data, learning_mode='both', lr_params=1e-3, weight_decay=0.0, feature_dim=2,
                 num_iter_fit=1000, covar_module='NN', mean_module='NN', mean_nn_layers=(32, 32), kernel_nn_layers=(32, 32),
                 task_batch_size=5, normalize_data=True, optimizer='Adam', lr_scheduler=False, random_seed=None):
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
            optimizer: (str) type of optimizer to use - must be either 'Adam' or 'SGD'
            random_seed: (int) seed for pytorch
        """
        self.logger = get_logger()

        assert learning_mode in ['learn_mean', 'learn_kernel', 'both', 'vanilla']
        assert mean_module in ['NN', 'constant', 'zero'] or isinstance(mean_module, gpytorch.means.Mean)
        assert covar_module in ['NN', 'SE'] or isinstance(covar_module, gpytorch.kernels.Kernel)
        assert optimizer in ['Adam', 'SGD']

        self.lr_params, self.weight_decay, self.feature_dim = lr_params, weight_decay, feature_dim
        self.num_iter_fit, self.task_batch_size, self.normalize_data = num_iter_fit, task_batch_size, normalize_data
        self.lr_scheduler = lr_scheduler

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

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.likelihoods.noise_models.GreaterThan(1e-3)).to(device)
        self.shared_parameters.append({'params': self.likelihood.parameters(), 'lr': self.lr_params})

        self.task_dicts = []

        for train_x, train_t in meta_train_data: # TODO: consider parallelizing this loop

            task_dict = {}

            # add data stats to task dict
            task_dict = self._compute_normalization_stats(train_x, train_t, stats_dict=task_dict)
            train_x, train_t = self._normalize_data(train_x, train_t, stats_dict=task_dict)

            # Convert the data into arrays of torch tensors
            task_dict['train_x'] = torch.from_numpy(train_x).float().to(device)
            task_dict['train_t'] = torch.from_numpy(train_t).float().flatten().to(device)

            task_dict['model'] = LearnedGPRegressionModel(task_dict['train_x'], task_dict['train_t'], self.likelihood,
                                              learned_kernel=self.nn_kernel_map, learned_mean=self.nn_mean_fn,
                                              covar_module=self.covar_module, mean_module=self.mean_module)
            task_dict['mll_fn'] = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, task_dict['model']).to(device)

            self.task_dicts.append(task_dict)

        if optimizer == 'Adam':
            self.optimizer = torch.optim.AdamW(self.shared_parameters)
        elif optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.shared_parameters)
        else:
            raise NotImplementedError('Optimizer must be Adam or SGD')

        if self.lr_scheduler:
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.2)

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
            cum_loss = 0.0

            for itr in range(1, self.num_iter_fit + 1):

                loss = 0.0
                self.optimizer.zero_grad()

                for task_dict in self.rds_numpy.choice(self.task_dicts, size=self.task_batch_size):

                    output = task_dict['model'](task_dict['train_x'])
                    mll = task_dict['mll_fn'](output, task_dict['train_t'])
                    loss -= mll / task_dict['train_x'].shape[0]

                loss.backward()
                self.optimizer.step()

                cum_loss += loss

                # print training stats stats
                if itr == 1 or itr % log_period == 0:
                    duration = time.time() - t
                    avg_loss = cum_loss / (log_period if itr > 1 else 1.0)
                    cum_loss = 0.0
                    t = time.time()

                    message = 'Iter %d/%d - Loss: %.6f - Time %.2f sec' % (itr, self.num_iter_fit, avg_loss.item(), duration)

                    # if validation data is provided  -> compute the valid log-likelihood
                    if valid_tuples is not None:
                        self.likelihood.eval()
                        valid_ll, valid_rmse = self.eval_datasets(valid_tuples)
                        if self.lr_scheduler:
                            self.lr_scheduler.step(valid_ll)
                        self.likelihood.train()
                        message += ' - Valid-LL: %.3f - Valid-RMSE: %.3f' % (np.mean(valid_ll), np.mean(valid_rmse))

                    if verbose:
                        self.logger.info(message)

        else:
            self.logger.info('Vanilla mode - nothing to fit')

        self.fitted = True

        for task_dict in self.task_dicts: task_dict['model'].eval()
        self.likelihood.eval()


    def predict(self, test_context_x, test_context_t, test_x, return_density=False):
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
        if test_x.ndim == 1:
            test_x = np.expand_dims(test_x, axis=-1)
        assert test_x.shape[1] == test_context_x.shape[1]

        # normalize data and convert to tensor
        data_stats = self._compute_normalization_stats(test_context_x, test_context_t, stats_dict={})
        test_context_x, test_context_t = self._normalize_data(test_context_x, test_context_t, stats_dict=data_stats)
        test_context_x_tensor = torch.from_numpy(test_context_x).float().to(device)
        test_context_t_tensor = torch.from_numpy(test_context_t).float().flatten().to(device)

        test_x = self._normalize_data(X=test_x, Y=None, stats_dict=data_stats)
        test_x_tensor = torch.from_numpy(test_x).float().to(device)

        with torch.no_grad():
            # compute posterior given the context data
            gp_model = LearnedGPRegressionModel(test_context_x_tensor, test_context_t_tensor, self.likelihood,
                                                learned_kernel=self.nn_kernel_map, learned_mean=self.nn_mean_fn,
                                                covar_module=self.covar_module, mean_module=self.mean_module)
            gp_model.eval()
            self.likelihood.eval()
            pred_dist = self.likelihood(gp_model(test_x_tensor))
            pred_dist_transformed = AffineTransformedDistribution(pred_dist, normalization_mean=data_stats['y_mean'],
                                                                  normalization_std=data_stats['y_std'])

        if return_density:
            return pred_dist_transformed
        else:
            pred_mean = pred_dist_transformed.mean
            pred_std = pred_dist_transformed.stddev
            return pred_mean.cpu().numpy(), pred_std.cpu().numpy()

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
        test_t_tensor = torch.from_numpy(test_t).float().flatten().to(device)

        with torch.no_grad():
            pred_dist = self.predict(test_context_x, test_context_t, test_x, return_density=True)
            avg_log_likelihood = pred_dist.log_prob(test_t_tensor) / test_t_tensor.shape[0]
            rmse = torch.mean(torch.pow(pred_dist.mean - test_t_tensor, 2)).sqrt()

            return avg_log_likelihood.cpu().item(), rmse.cpu().item()

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


    def _setup_gp_prior(self, mean_module, covar_module, learning_mode, feature_dim, mean_nn_layers, kernel_nn_layers):

        self.shared_parameters = []

        # a) determine kernel map & module
        if covar_module == 'NN':
            assert learning_mode in ['learn_kernel', 'both'], 'neural network parameters must be learned'
            self.nn_kernel_map = NeuralNetwork(input_dim=self.input_dim, output_dim=feature_dim,
                                          layer_sizes=kernel_nn_layers).to(device)
            self.shared_parameters.append(
                {'params': self.nn_kernel_map.parameters(), 'lr': self.lr_params, 'weight_decay': self.weight_decay})
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=feature_dim)).to(device)
        else:
            self.nn_kernel_map = None

        if covar_module == 'SE':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=feature_dim)).to(device)
        elif isinstance(covar_module, gpytorch.kernels.Kernel):
            self.covar_module = covar_module.to(device)

        # b) determine mean map & module

        if mean_module == 'NN':
            assert learning_mode in ['learn_mean', 'both'], 'neural network parameters must be learned'
            self.nn_mean_fn = NeuralNetwork(input_dim=self.input_dim, output_dim=1, layer_sizes=mean_nn_layers).to(device)
            self.shared_parameters.append(
                {'params': self.nn_mean_fn.parameters(), 'lr': self.lr_params, 'weight_decay': self.weight_decay})
            self.mean_module = None
        else:
            self.nn_mean_fn = None

        if mean_module == 'constant':
            self.mean_module = gpytorch.means.ConstantMean().to(device)
        elif mean_module == 'zero':
            self.mean_module = gpytorch.means.ZeroMean().to(device)
        elif isinstance(mean_module, gpytorch.means.Mean):
            self.mean_module = mean_module.to(device)

        # c) add parameters of covar and mean module if desired

        if learning_mode in ["learn_kernel", "both"]:
            self.shared_parameters.append({'params': self.covar_module.hyperparameters(), 'lr': self.lr_params})

        if learning_mode in ["learn_mean", "both"] and self.mean_module is not None:
            self.shared_parameters.append({'params': self.mean_module.hyperparameters(), 'lr': self.lr_params})


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