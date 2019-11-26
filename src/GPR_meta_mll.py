import torch
import gpytorch
import time

import numpy as np

from src.models import LearnedGPRegressionModel, NeuralNetwork, AffineTransformedDistribution
from src.util import _handle_input_dimensionality, DummyLRScheduler
from src.abstract import RegressionModelMetaLearned
from config import device

class GPRegressionMetaLearned(RegressionModelMetaLearned):

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
            lr_scheduler: (str) whether to use a lr scheduler
            random_seed: (int) seed for pytorch
        """
        super().__init__(normalize_data, random_seed)

        assert learning_mode in ['learn_mean', 'learn_kernel', 'both', 'vanilla']
        assert mean_module in ['NN', 'constant', 'zero'] or isinstance(mean_module, gpytorch.means.Mean)
        assert covar_module in ['NN', 'SE'] or isinstance(covar_module, gpytorch.kernels.Kernel)
        assert optimizer in ['Adam', 'SGD']

        self.lr_params, self.weight_decay, self.feature_dim = lr_params, weight_decay, feature_dim
        self.num_iter_fit, self.task_batch_size, self.normalize_data = num_iter_fit, task_batch_size, normalize_data

        # Check that data all has the same size
        self._check_meta_data_shapes(meta_train_data)

        # Setup components that are shared across tasks
        self._setup_gp_prior(mean_module, covar_module, learning_mode, feature_dim, mean_nn_layers, kernel_nn_layers)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.likelihoods.noise_models.GreaterThan(1e-3)).to(device)
        self.shared_parameters.append({'params': self.likelihood.parameters(), 'lr': self.lr_params})

        # Setup components that are different across tasks
        self.task_dicts = []

        for train_x, train_y in meta_train_data: # TODO: consider parallelizing this loop
            task_dict = {}

            # a) prepare data
            x_tensor, y_tensor, task_dict = self._prepare_data_per_task(train_x, train_y, stats_dict=task_dict)
            task_dict['train_x'], task_dict['train_y'] = x_tensor, y_tensor

            # b) prepare model
            task_dict['model'] = LearnedGPRegressionModel(task_dict['train_x'], task_dict['train_y'], self.likelihood,
                                              learned_kernel=self.nn_kernel_map, learned_mean=self.nn_mean_fn,
                                              covar_module=self.covar_module, mean_module=self.mean_module)
            task_dict['mll_fn'] = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, task_dict['model']).to(device)

            self.task_dicts.append(task_dict)

        # c) prepare inference
        self._setup_optimizer(optimizer, lr_params, lr_scheduler)

        self.fitted = False


    def meta_fit(self, valid_tuples=None, verbose=True, log_period=500, n_iter=None):
        """
        fits the VI and prior parameters of the  GPC model

        Args:
            valid_tuples: list of valid tuples, i.e. [(test_context_x_1, test_context_t_1, test_x_1, test_t_1), ...]
            verbose: (boolean) whether to print training progress
            log_period: (int) number of steps after which to print stats
            n_iter: (int) number of gradient descent iterations
        """
        for task_dict in self.task_dicts: task_dict['model'].train()
        self.likelihood.train()

        assert (valid_tuples is None) or (all([len(valid_tuple) == 4 for valid_tuple in valid_tuples]))

        if len(self.shared_parameters) > 0:
            t = time.time()
            cum_loss = 0.0

            if n_iter is None:
                n_iter = self.num_iter_fit

            for itr in range(1, n_iter + 1):

                loss = 0.0
                self.optimizer.zero_grad()

                for task_dict in self.rds_numpy.choice(self.task_dicts, size=self.task_batch_size):

                    output = task_dict['model'](task_dict['train_x'])
                    mll = task_dict['mll_fn'](output, task_dict['train_y'])
                    loss -= mll / task_dict['train_x'].shape[0]

                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step(loss)

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
                        self.likelihood.train()
                        message += ' - Valid-LL: %.3f - Valid-RMSE: %.3f' % (np.mean(valid_ll), np.mean(valid_rmse))

                    if verbose:
                        self.logger.info(message)

        else:
            self.logger.info('Vanilla mode - nothing to fit')

        self.fitted = True

        for task_dict in self.task_dicts: task_dict['model'].eval()
        self.likelihood.eval()
        return loss.item()


    def predict(self, context_x, context_y, test_x, return_density=False):
        """
        computes the predictive distribution of the targets p(t|test_x, test_context_x, context_y)

        Args:
            context_x: (ndarray) context input data for which to compute the posterior
            context_y: (ndarray) context targets for which to compute the posterior
            test_x: (ndarray) query input data of shape (n_samples, ndim_x)
            return_density: (bool) whether to return result as mean and std ndarray or as MultivariateNormal pytorch object

        Returns:
            (pred_mean, pred_std) predicted mean and standard deviation corresponding to p(t|test_x, test_context_x, context_y)
        """

        context_x, context_y = _handle_input_dimensionality(context_x, context_y)
        test_x = _handle_input_dimensionality(test_x)
        assert test_x.shape[1] == context_x.shape[1]

        # normalize data and convert to tensor
        context_x, context_y, data_stats = self._prepare_data_per_task(context_x, context_y, stats_dict={})

        test_x = self._normalize_data(X=test_x, Y=None, stats_dict=data_stats)
        test_x = torch.from_numpy(test_x).float().to(device)

        with torch.no_grad():
            # compute posterior given the context data
            gp_model = LearnedGPRegressionModel(context_x, context_y, self.likelihood,
                                                learned_kernel=self.nn_kernel_map, learned_mean=self.nn_mean_fn,
                                                covar_module=self.covar_module, mean_module=self.mean_module)
            gp_model.eval()
            self.likelihood.eval()
            pred_dist = self.likelihood(gp_model(test_x))
            pred_dist_transformed = AffineTransformedDistribution(pred_dist, normalization_mean=data_stats['y_mean'],
                                                                  normalization_std=data_stats['y_std'])

        if return_density:
            return pred_dist_transformed
        else:
            pred_mean = pred_dist_transformed.mean
            pred_std = pred_dist_transformed.stddev
            return pred_mean.cpu().numpy(), pred_std.cpu().numpy()

    def eval(self, context_x, context_y, test_x, test_t):
        """
        Computes the average test log likelihood and the rmse on test data

        Args:
            context_x: (ndarray) context input data for which to compute the posterior
            context_y: (ndarray) context targets for which to compute the posterior
            test_x: (ndarray) test input data of shape (n_samples, ndim_x)
            test_t: (ndarray) test target data of shape (n_samples, 1)

        Returns: (avg_log_likelihood, rmse)

        """

        context_x, context_y = _handle_input_dimensionality(context_x, context_y)
        test_x, test_t = _handle_input_dimensionality(test_x, test_t)
        test_t_tensor = torch.from_numpy(test_t).float().flatten().to(device)

        with torch.no_grad():
            pred_dist = self.predict(context_x, context_y, test_x, return_density=True)
            avg_log_likelihood = pred_dist.log_prob(test_t_tensor) / test_t_tensor.shape[0]
            rmse = torch.mean(torch.pow(pred_dist.mean - test_t_tensor, 2)).sqrt()

            return avg_log_likelihood.cpu().item(), rmse.cpu().item()

    def eval_datasets(self, test_tuples):
        """
        Computes the average test log likelihood and the rmse over multiple test datasets

        Args:
            test_tuples: list of test set tuples, i.e. [(test_context_x_1, test_context_y_1, test_x_1, test_y_1), ...]

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

    def _setup_optimizer(self, optimizer, lr, lr_scheduler):
        if optimizer == 'Adam':
            self.optimizer = torch.optim.AdamW(self.shared_parameters, lr=lr)
        elif optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.shared_parameters, lr=lr)
        else:
            raise NotImplementedError('Optimizer must be Adam or SGD')

        if lr_scheduler:
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',
                                                                           factor=0.2, patience=100, threshold=1e-3)
        else:
            self.lr_scheduler = DummyLRScheduler()