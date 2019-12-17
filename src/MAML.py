import time
import torch.nn as nn
import math
import torch
import numpy as np

from src.models import NeuralNetwork
from src.abstract import RegressionModelMetaLearned
from src.util import DummyLRScheduler, _handle_input_dimensionality

from config import device

class MAMLRegression(RegressionModelMetaLearned):

    def __init__(self, meta_train_data, layer_sizes=(32, 32, 32, 32), num_iter_fit=20000, lr_inner=0.05, num_inner_steps=1,
                 task_batch_size=5, lr_meta=1e-3, lr_decay=1.0, optimizer='Adam', normalize_data=True, random_seed=None):
        """
        Few Shot Regression with Model-Agnostic Meta-Learning (MAML)

        Args:
            meta_train_data: list of tuples of ndarrays[(train_x_1, train_t_1), ..., (train_x_n, train_t_n)]
            layer_sizes: (tuple) hidden layer sizes of  NN
            num_iter_fit: (int) number of gradient steps for fitting the parameters
            lr_inner: (float) learning rate for (inner) adaptation steps
            num_inner_steps: (int) number of inner gradient-steps
            task_batch_size (int): number of tasks for estimating the meta-gradient in each iteration
            lr_meta: (float) learning rate for optimizing the meta-objective
            lr_decay: (float) multiplicative lr decay factor applied after every 1000 gradient steps
            optimizer: (str) type of optimizer to use - must be either 'Adam' or 'SGD'
            normalize_data: (bool) whether to normalize the data
            random_seed: (int) seed for pytorch and numpy rng
        """
        super().__init__(normalize_data, random_seed)

        assert optimizer in ['Adam', 'SGD']

        # prepare data
        self._check_meta_data_shapes(meta_train_data)
        self._compute_normalization_stats(meta_train_data)

        self.meta_train_data = [self._prepare_data_per_task(train_x, train_y, flatten_y=False)
                                                                for train_x, train_y in meta_train_data]

        self.nn = NeuralNetwork(self.input_dim, self.output_dim, layer_sizes=layer_sizes)
        self.initial_params = list(self.nn.parameters())
        self.num_inner_steps = num_inner_steps
        self.lr_inner = lr_inner
        self.task_batch_size = task_batch_size
        self.num_iter_fit = num_iter_fit

        self.loss_fn = nn.MSELoss()

        # c) prepare inference
        self._setup_optimizer(optimizer, lr_meta, lr_decay)

        self.fitted = False

    def meta_fit(self, valid_tuples=None, verbose=True, log_period=500, n_iter=None):
        """
        fits the initial params via MAML

        Args:
            valid_tuples: list of valid tuples, i.e. [(test_context_x_1, test_context_t_1, test_x_1, test_t_1), ...]
            verbose: (boolean) whether to print training progress
            log_period: (int) number of steps after which to print stats
            n_iter: (int) number of gradient descent iterations
        """

        assert (valid_tuples is None) or (all([len(valid_tuple) == 4 for valid_tuple in valid_tuples]))

        t = time.time()

        if n_iter is None:
            n_iter = self.num_iter_fit

        cum_loss = 0.0

        for itr in range(1, n_iter + 1):

            task_batch_indices = self.rds_numpy.choice(len(self.meta_train_data), size=self.task_batch_size)
            meta_train_tuples = [self.meta_train_data[idx] for idx in task_batch_indices]
            loss = self._meta_step(meta_train_tuples)

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
                    valid_rmse = self.eval_datasets(valid_tuples)
                    message += ' Valid-RMSE: %.3f ' % valid_rmse

                if verbose:
                    self.logger.info(message)


        self.fitted = True

        return loss.item()

    def predict(self, context_x, context_y, test_x, return_tensor=False, num_steps_eval=None):
        """
        adapts the initial (MAML) parameters based on the context data and compute prediction for test_x

        Args:
            context_x: (ndarray) context input data for which to compute the posterior
            context_y: (ndarray) context targets for which to compute the posterior
            test_x: (ndarray) query input data of shape (n_samples, ndim_x)
            return_tensor: (bool) whether return a torch tensor or a numpy array
            num_steps_eval: (int) number of adaptation steps

        Returns:
            pred_mean: predicted mean corresponding to test_x
        """

        context_x, context_y = _handle_input_dimensionality(context_x, context_y)
        test_x = _handle_input_dimensionality(test_x)
        assert test_x.shape[1] == context_x.shape[1]

        # normalize data and convert to tensor
        context_x, context_y = self._prepare_data_per_task(context_x, context_y, flatten_y=False)

        test_x = self._normalize_data(X=test_x, Y=None)
        test_x = torch.from_numpy(test_x).float().to(device)

        # perform adaptation steps on context data
        adapted_params = self._eval_steps(context_x, context_y, num_steps_eval=num_steps_eval)
        with torch.no_grad():
            y_pred = self.nn.forward_parametrized(test_x, adapted_params)
            y_pred = y_pred * torch.Tensor(self.y_std).float()[None, :] + torch.Tensor(self.y_mean).float()[None, :]

            y_pred2 = self.nn.forward_parametrized(test_x, self.initial_params)
            y_pred2 = y_pred2 * torch.Tensor(self.y_std).float()[None, :] + torch.Tensor(self.y_mean).float()[None, :]

            if return_tensor:
                return y_pred
            else:
                return y_pred.cpu().numpy(), y_pred2.cpu().numpy()

    def eval(self, context_x, context_y, test_x, test_y, num_steps_eval=None):
        """
           Computes the rmse on the test data after adapting the parameters to the context data

           Args:
               context_x: (ndarray) context input data for which to compute the posterior
               context_y: (ndarray) context targets for which to compute the posterior
               test_x: (ndarray) test input data of shape (n_samples, ndim_x)
               test_y: (ndarray) test target data of shape (n_samples, ndim_y)

           Returns: rmse

        """
        test_x, test_y = _handle_input_dimensionality(test_x, test_y)
        test_y_tensor = torch.from_numpy(test_y).float().to(device)

        y_pred = self.predict(context_x, context_y, test_x, return_tensor=True, num_steps_eval=num_steps_eval)

        # print(self.loss_fn(y_pred, test_y_tensor).item())
        rmse = torch.mean(torch.sum(torch.pow(y_pred - test_y_tensor, 2), dim=-1)).sqrt()

        return rmse.cpu().item()

    def eval_datasets(self, test_tuples, **kwargs):
        """
           Computes the average test rmse over multiple test datasets

           Args:
               test_tuples: list of test set tuples, i.e. [(test_context_x_1, test_context_y_1, test_x_1, test_y_1), ...]

           Returns: rmse
        """
        assert (all([len(valid_tuple) == 4 for valid_tuple in test_tuples]))

        rmse_list = [self.eval(*test_data_tuple, **kwargs) for test_data_tuple in test_tuples]

        return np.mean(rmse_list)

    def _setup_optimizer(self, optimizer, lr, lr_decay):
        if optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.initial_params, lr=lr)
        elif optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.initial_params, lr=lr)
        else:
            raise NotImplementedError('Optimizer must be Adam or SGD')

        if lr_decay < 1.0:
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1000, gamma=lr_decay)
        else:
            self.lr_scheduler = DummyLRScheduler()

    def _inner_steps(self, x_data, y_data):
        # split data into two parts
        idx_split = math.ceil(x_data.shape[0] / 2.0)
        x_1, y_1 = x_data[:idx_split], y_data[:idx_split] # data for inner update
        x_2, y_2 = x_data[idx_split:], y_data[idx_split:] # data for computing meta loss

        # clone initial parameters
        temp_params = [param.clone() for param in self.initial_params]

        # inner steps
        for step in range(self.num_inner_steps):
            mse_loss = self.loss_fn(self.nn.forward_parametrized(x_1, temp_params), y_1)
            # compute grad and update inner params
            grads = torch.autograd.grad(mse_loss, temp_params, create_graph=True)
            temp_params = [p - self.lr_inner * g for p, g in zip(temp_params, grads)]

        # compute meta-objective
        task_loss = self.loss_fn(self.nn.forward_parametrized(x_2, temp_params), y_2)

        return task_loss

    def _meta_step(self, train_task_tuples):
        self.optimizer.zero_grad()

        meta_loss = 0.0
        for x_train, y_train in train_task_tuples:
            meta_loss += self._inner_steps(x_train, y_train)
        meta_loss /= len(train_task_tuples)

        # compute meta gradient of loss with respect to maml initial_params
        meta_grads = torch.autograd.grad(meta_loss, self.initial_params)

        # assign meta gradient to initial_params and take optimisation step
        for w, g in zip(self.initial_params, meta_grads):
            w.grad = g
        self.optimizer.step()
        return meta_loss

    def _eval_steps(self, x_data, y_data, num_steps_eval=None):

        if num_steps_eval is None:
            num_steps_eval = self.num_inner_steps

        temp_params = [param.clone() for param in self.initial_params]

        # inner steps
        for step in range(num_steps_eval):
            mse_loss = self.loss_fn(self.nn.forward_parametrized(x_data, temp_params), y_data)

            # compute grad and update inner params
            grads = torch.autograd.grad(mse_loss, temp_params, create_graph=False)
            temp_params = [p - self.lr_inner * g for p, g in zip(temp_params, grads)]
        return temp_params


if __name__ == "__main__":

    from experiments.data_sim import SinusoidDataset
    import torch
    import numpy as np

    import os
    print(os.getenv("PYCHARM_DISPLAY_PORT"))

    torch.set_num_threads(2)

    dataset = SinusoidDataset()
    meta_train_data = dataset.generate_meta_train_data(n_tasks=5000, n_samples=10)
    meta_test_data = dataset.generate_meta_test_data(n_tasks=1000, n_samples_context=5, n_samples_test=100)

    meta_learner = MAMLRegression(meta_train_data, task_batch_size=10, num_iter_fit=10000)
    meta_learner.meta_fit(meta_test_data[:200], log_period=1000)


    for i in range(4):
        x_context, y_context, x_test, y_test = meta_test_data[i]
        idx = np.argsort(x_test, axis=0).flatten()
        x_test,  y_test = x_test[idx], y_test[idx]

        rmse = meta_learner.eval(x_context, y_context, x_test, y_test)

        y_pred_post, y_pred_pre = meta_learner.predict(x_context, y_context, x_test)

        from matplotlib import pyplot as plt
        plt.scatter(x_context, y_context)
        plt.scatter(x_test, y_test, color='grey')
        plt.plot(x_test, y_pred_pre, color='red')
        plt.plot(x_test, y_pred_post, 'green')
        plt.title('num iter: %i'%10000)
        plt.show()



    # nn = NeuralNetwork(1, 1, layer_sizes=(32, 32))
    #
    # x_train = torch.Tensor(train_data[0][0]).float().to(device)
    # y_train = torch.Tensor(train_data[0][1]).float().to(device)
    #
    # params = list(nn.parameters())
    #
    # optim = torch.optim.Adam(params)
    #
    # for i in range(5000):
    #     optim.zero_grad()
    #     loss = 0.5 * torch.mean((nn.forward_parametrized(x_train, params) - y_train) ** 2)
    #     loss.backward()
    #     optim.step()
    #     if i % 500 == 0:
    #         print(i, loss.item())
    #
    #
    # x_plot = np.linspace(-4, 4, 200).reshape(-1, 1)
    #
    # with torch.no_grad():
    #     x = torch.Tensor(x_plot).float().to(device)
    #     y_pred = nn(x).numpy()
    #
    # from matplotlib import pyplot as plt
    #
    # plt.scatter(train_data[0][0], train_data[0][1])
    # plt.plot(x, y_pred)
    # plt.show()