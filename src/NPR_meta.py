import sys
import os
import torch
import gpytorch
import time
import math
import numpy as np
from torch.distributions.kl import kl_divergence
from torch.utils.data import DataLoader

sys.path.append("")

from third_party.neural_processes.utils import context_target_split
from third_party.neural_processes.training import NeuralProcessTrainer
from third_party.neural_processes.datasets import SineData
from third_party.neural_processes.neural_process import NeuralProcess

from src.models import AffineTransformedDistribution
from src.util import _handle_input_dimensionality, DummyLRScheduler
from src.abstract import RegressionModelMetaLearned
from config import device


class NPRegressionMetaLearned(RegressionModelMetaLearned):

    def __init__(self, meta_train_data, num_context, num_extra_target, lr_params=1e-3, r_dim=50, z_dim=50, h_dim=50, num_iter_fit=10000,
                 task_batch_size=5, normalize_data=True, optimizer='Adam', lr_decay=1.0, random_seed=None):
        """
        Neural Process regression model (https://arxiv.org/abs/1807.01622) that supports meta-learning.

        Args:
            meta_train_data: list of tuples of ndarrays[(train_x_1, train_t_1), ..., (train_x_n, train_t_n)]
            num_context: number of context points for all training task or list of numbers for each task
            num_extra_target: number of target points for all training task or list of numbers for each task
            lr_params: (float) learning rate for prior parameters
            r_dim: (float) dimensionality of the context representation
            z_dim: (float) dimensionality of the latent variable
            h_dim: (float) layer width of encoder and decoder
            num_iter_fit: (int) number of gradient steps for fitting the parameters
            task_batch_size: (int) batch size for meta training, i.e. number of tasks for computing grads
            normalize_data: (bool) decides whether the input data should be normalized
            optimizer: (str) type of optimizer to use - must be either 'Adam' or 'SGD'
            lr_decay: (float) decay rate for the learning rate
            random_seed: (int) seed for pytorch
        """
        super().__init__(normalize_data, random_seed)

        assert optimizer in ['Adam', 'SGD']
        assert type(num_context) in [int, list, np.ndarray]
        assert type(num_extra_target) in [int, list, np.ndarray]
        if type(num_context) != int:
            assert len(num_context) == len(meta_train_data)
        if type(num_extra_target) != int:
            assert len(num_extra_target) == len(meta_train_data)

        self.num_context, self.num_extra_target = num_context, num_extra_target
        self.lr_params, self.r_dim, self.z_dim, self.h_dim = lr_params, r_dim, z_dim, h_dim
        self.num_iter_fit, self.task_batch_size, self.normalize_data = num_iter_fit, task_batch_size, normalize_data

        # Check that data all has the same size
        self._check_meta_data_shapes(meta_train_data)
        self._compute_normalization_stats(meta_train_data)
        
        self.input_dim = meta_train_data[0][0].shape[-1]
        self.output_dim = meta_train_data[0][1].shape[-1]
        
        self.model = NeuralProcess(x_dim=self.input_dim,
                                  y_dim=self.output_dim,
                                  r_dim=self.r_dim,
                                  z_dim=self.z_dim,
                                  h_dim=self.h_dim)

        # Setup components that are shared across tasks
        self.shared_parameters = self.model.parameters()

        # Setup components that are different across tasks
        self.task_dicts = []

        for i, (train_x, train_y) in enumerate(meta_train_data): # TODO: consider parallelizing this loop
            task_dict = {}
            
            if len(train_x) < 10:
                print(len(train_x))

            # a) prepare data
            x_tensor, y_tensor = self._prepare_data_per_task(train_x, train_y, flatten_y=False)
            task_dict['train_x'], task_dict['train_y'] = x_tensor, y_tensor
            if type(self.num_context) != int:
                task_dict['num_context'] = self.num_context[i]
            if type(self.num_extra_target) != int:
                task_dict['num_extra_target'] = self.num_extra_target[i]

            self.task_dicts.append(task_dict)

        # c) prepare inference
        self._setup_optimizer(optimizer, lr_params, lr_decay)

        self.fitted = False


    def meta_fit(self, valid_tuples=None, verbose=True, log_period=500, n_iter=None):
        """
        fits the parameters of the Neural Process model

        Args:
            valid_tuples: list of valid tuples, i.e. [(test_context_x_1, test_context_t_1, test_x_1, test_t_1), ...]
            verbose: (boolean) whether to print training progress
            log_period: (int) number of steps after which to print stats
            n_iter: (int) number of gradient descent iterations
        """
        self.model.train()

        assert (valid_tuples is None) or (all([len(valid_tuple) == 4 for valid_tuple in valid_tuples]))

        t = time.time()
        cum_loss = 0.0

        if n_iter is None:
            n_iter = self.num_iter_fit

        for itr in range(1, n_iter + 1):

            loss = 0.0
            self.optimizer.zero_grad()
                
            batch = self.rds_numpy.choice(self.task_dicts, size=self.task_batch_size)
            for task in batch:
                batch_x = torch.unsqueeze(task["train_x"], dim=0)
                batch_y = torch.unsqueeze(task["train_y"], dim=0)

                if "num_context" in task:
                    num_context = task["num_context"]
                else:
                    num_context = self.num_context
                    
                if "num_extra_target" in task:
                    num_extra_target = task["num_extra_target"]
                else:
                    num_extra_target = self.num_extra_target
        
                x_context, y_context, x_target, y_target = \
                    context_target_split(batch_x, batch_y,
                                         num_context, num_extra_target)
                p_y_pred, q_target, q_context = \
                    self.model(x_context, y_context, x_target, y_target)
                loss += self._loss(p_y_pred, y_target, q_target, q_context)

            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

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
                    self.model.eval()
                    valid_ll, valid_rmse, calibr_err = self.eval_datasets(valid_tuples, flatten_y=False)
                    self.model.train()
                    message += ' - Valid-LL: %.3f - Valid-RMSE: %.3f - Calib-Err %.3f' % (valid_ll, valid_rmse, calibr_err)

                if verbose:
                    self.logger.info(message)


        self.fitted = True

        self.model.eval()
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

        train_old = self.model.training
        self.model.eval()
        
        context_x, context_y = _handle_input_dimensionality(context_x, context_y)
        test_x = _handle_input_dimensionality(test_x)
        assert test_x.shape[1] == context_x.shape[1]

        # normalize data and convert to tensor
        context_x, context_y = self._prepare_data_per_task(context_x, context_y, flatten_y=False)

        test_x = self._normalize_data(X=test_x, Y=None)
        test_x = torch.from_numpy(test_x).float().to(device)
        
        context_x = torch.unsqueeze(context_x, 0)
        context_y = torch.unsqueeze(context_y, 0)
        test_x = torch.unsqueeze(test_x, 0)

        with torch.no_grad():
            # compute posterior given the context data
            pred_dist = self.model(context_x, context_y, test_x)
            pred_dist_transformed = AffineTransformedDistribution(pred_dist, normalization_mean=self.y_mean,
                                                                  normalization_std=self.y_std)            

        if train_old:
            self.model.train()
            
        if return_density:
            return pred_dist_transformed
        else:
            pred_mean = pred_dist_transformed.mean
            pred_std = pred_dist_transformed.stddev
            return pred_mean.cpu().numpy(), pred_std.cpu().numpy()

        
    def state_dict(self):
        state_dict = {
            'optimizer': self.optimizer.state_dict(),
            'model': self.model.state_dict()
        }
        for key, tensor in self.model.state_dict().items():
            assert torch.all(state_dict['model'][key] == tensor).item()
        return state_dict

    
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        
        
    def _loss(self, p_y_pred, y_target, q_target, q_context):
        """
        Computes Neural Process loss.

        Parameters
        ----------
        p_y_pred : one of torch.distributions.Distribution
            Distribution over y output by Neural Process.

        y_target : torch.Tensor
            Shape (batch_size, num_target, y_dim)

        q_target : one of torch.distributions.Distribution
            Latent distribution for target points.

        q_context : one of torch.distributions.Distribution
            Latent distribution for context points.
        """
        # Log likelihood has shape (batch_size, num_target, y_dim). Take mean
        # over batch and sum over number of targets and dimensions of y
        log_likelihood = p_y_pred.log_prob(y_target).mean(dim=0).sum()
        # KL has shape (batch_size, r_dim). Take mean over batch and sum over
        # r_dim (since r_dim is dimension of normal distribution)
        kl = kl_divergence(q_target, q_context).mean(dim=0).sum()
        return -log_likelihood + kl
    

    def _setup_optimizer(self, optimizer, lr, lr_decay):
        if optimizer == 'Adam':
            self.optimizer = torch.optim.AdamW(self.shared_parameters, lr=lr)
        elif optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.shared_parameters, lr=lr)
        else:
            raise NotImplementedError('Optimizer must be Adam or SGD')

        if lr_decay < 1.0:
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1000, gamma=lr_decay)
        else:
            self.lr_scheduler = DummyLRScheduler()

            
    def _vectorize_pred_dist(self, pred_dist):
        return torch.distributions.Normal(pred_dist.mean, pred_dist.stddev)


if __name__ == "__main__":
    from experiments.data_sim import GPFunctionsDataset, SinusoidDataset, provide_data

    meta_train_data, _, meta_test_data = provide_data('sin_20')

    num_context, num_extra_target = 3, 2
    
    plot = True
    from matplotlib import pyplot as plt

    if plot:
        for x_train, y_train in meta_train_data:
            plt.scatter(x_train, y_train)
        plt.title('sample from the GP prior')
        plt.show()

    print('\n ---- NPR meta-learning ---- ')

    torch.set_num_threads(2)

    meta_np = NPRegressionMetaLearned(meta_train_data=meta_train_data)
    
    itrs = 0
    meta_np.meta_fit(valid_tuples=meta_test_data, log_period=1000, n_iter=10000)
    itrs += 20000
    
    test_context_x, test_context_y, test_target_x, test_target_y = meta_test_data[0]
    x_plot = np.linspace(-5, 5, num=150)
    x_plot = x_plot[:, np.newaxis]
    pred_mean, pred_std = meta_np.predict(context_x=test_context_x, context_y=test_context_y, test_x=x_plot)
    ucb, lcb = meta_np.confidence_intervals(test_context_x, test_context_y, x_plot, confidence=0.9)

    plt.scatter(test_context_x, test_context_y)
    plt.scatter(test_target_x, test_target_y)

    plt.plot(x_plot.flatten(), pred_mean.flatten())
    plt.fill_between(x_plot.flatten(), lcb.flatten(), ucb.flatten(), alpha=0.2)
    plt.title('NPR meta mll')
    plt.show()
    
    save = False
    if save:
        plt.savefig("npr_plot.pdf")