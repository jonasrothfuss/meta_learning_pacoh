import numpy as np
import torch
from src.util import get_logger, _handle_input_dimensionality
from config import device


class RegressionModel:

    def __init__(self, normalize_data=True, random_seed=None):
        self.normalize_data = normalize_data
        self.logger = get_logger()
        self.input_dim = None
        self.output_dim = None
        self.n_train_samples = None
        self.train_x = None
        self.train_t = None

        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed+1)

    def predict(self, test_x, return_density=False, **kwargs):
        raise NotImplementedError

    def eval(self, test_x, test_t, **kwargs):
        """
        Computes the average test log likelihood and the rmse on test data

        Args:
            test_x: (ndarray) test input data of shape (n_samples, ndim_x)
            test_t: (ndarray) test target data of shape (n_samples, 1)

        Returns: (avg_log_likelihood, rmse)

        """
        # convert to tensors
        test_x, test_t = _handle_input_dimensionality(test_x, test_t)
        test_t_tensor = torch.from_numpy(test_t).contiguous().float().flatten().to(device)

        with torch.no_grad():
            pred_dist = self.predict(test_x, return_density=True, *kwargs)
            avg_log_likelihood = pred_dist.log_prob(test_t_tensor) / test_t_tensor.shape[0]
            rmse = torch.mean(torch.pow(pred_dist.mean - test_t_tensor, 2)).sqrt()

            pred_dist_vect = self._vectorize_pred_dist(pred_dist)
            calibr_error = self._calib_error(pred_dist_vect, test_t_tensor)

            return avg_log_likelihood.cpu().item(), rmse.cpu().item(), calibr_error.cpu().item()

    def confidence_intervals(self, test_x, confidence=0.9, **kwargs):
        pred_dist = self.predict(test_x, return_density=True, **kwargs)
        pred_dist = self._vectorize_pred_dist(pred_dist)

        alpha = (1 - confidence) / 2
        ucb = pred_dist.icdf(torch.ones(test_x.size) * (1 - alpha))
        lcb = pred_dist.icdf(torch.ones(test_x.size) * alpha)
        return ucb, lcb

    def _calib_error(self, pred_dist_vectorized, test_t_tensor):
        return _calib_error(pred_dist_vectorized, test_t_tensor)

    def _compute_normalization_stats(self, X, Y):
        # save mean and variance of data for normalization
        if self.normalize_data:
            self.x_mean, self.y_mean = np.mean(X, axis=0), np.mean(Y, axis=0)
            self.x_std, self.y_std = np.std(X, axis=0), np.std(Y, axis=0)
        else:
            self.x_mean, self.y_mean = np.zeros(X.shape[1]), np.zeros(Y.shape[1])
            self.x_std, self.y_std = np.ones(X.shape[1]), np.ones(Y.shape[1])

    def _normalize_data(self, X, Y=None):
        assert hasattr(self, "x_mean") and hasattr(self, "x_std"), "requires computing normalization stats beforehand"
        assert hasattr(self, "y_mean") and hasattr(self, "y_std"), "requires computing normalization stats beforehand"

        X_normalized = (X - self.x_mean[None, :]) / self.x_std[None, :]

        if Y is None:
            return X_normalized
        else:
            Y_normalized = (Y - self.y_mean[None, :]) / self.y_std[None, :]
            return X_normalized, Y_normalized

    def _unnormalize_pred(self, pred_mean, pred_std):
        assert hasattr(self, "x_mean") and hasattr(self, "x_std"), "requires computing normalization stats beforehand"
        assert hasattr(self, "y_mean") and hasattr(self, "y_std"), "requires computing normalization stats beforehand"

        if self.normalize_data:
            assert pred_mean.ndim == pred_std.ndim == 2 and pred_mean.shape[1] == pred_std.shape[1] == self.output_dim
            if isinstance(pred_mean, torch.Tensor) and isinstance(pred_std, torch.Tensor):
                y_mean_tensor, y_std_tensor = torch.tensor(self.y_mean).float(), torch.tensor(self.y_std).float()
                pred_mean = pred_mean.mul(y_std_tensor[None, :]) + y_mean_tensor[None, :]
                pred_std = pred_std.mul(y_std_tensor[None, :])
            else:
                pred_mean = pred_mean.multiply(self.y_std[None, :]) + self.y_mean[None, :]
                pred_std = pred_std.multiply(self.y_std[None, :])

        return pred_mean, pred_std

    def _initial_data_handling(self, train_x, train_t):
        train_x, train_t = _handle_input_dimensionality(train_x, train_t)
        self.input_dim, self.output_dim = train_x.shape[-1], train_t.shape[-1]
        self.n_train_samples = train_x.shape[0]

        # b) normalize data to exhibit zero mean and variance
        self._compute_normalization_stats(train_x, train_t)
        train_x_normalized, train_t_normalized = self._normalize_data(train_x, train_t)

        # c) Convert the data into pytorch tensors
        self.train_x = torch.from_numpy(train_x_normalized).contiguous().float().to(device)
        self.train_t = torch.from_numpy(train_t_normalized).contiguous().float().to(device)

        return self.train_x, self.train_t

    def _vectorize_pred_dist(self, pred_dist):
        raise NotImplementedError

class RegressionModelMetaLearned:

    def __init__(self, normalize_data=True, random_seed=None):
        self.normalize_data = normalize_data
        self.logger = get_logger()
        self.input_dim = None
        self.output_dim = None

        if random_seed is not None:
            torch.manual_seed(random_seed)
            self.rds_numpy = np.random.RandomState(random_seed + 1)
        else:
            self.rds_numpy = np.random

    def predict(self, context_x, context_y, test_x, **kwargs):
        raise NotImplementedError

    def eval(self, context_x, context_y, test_x, test_t, **kwargs):
        """
        Computes the average test log likelihood, rmse and calibration error n test data

        Args:
            context_x: (ndarray) context input data for which to compute the posterior
            context_y: (ndarray) context targets for which to compute the posterior
            test_x: (ndarray) test input data of shape (n_samples, ndim_x)
            test_t: (ndarray) test target data of shape (n_samples, 1)

        Returns: (avg_log_likelihood, rmse, calibr_error)

        """

        context_x, context_y = _handle_input_dimensionality(context_x, context_y)
        test_x, test_t = _handle_input_dimensionality(test_x, test_t)
        test_t_tensor = torch.from_numpy(test_t).float().flatten().to(device)

        with torch.no_grad():
            pred_dist = self.predict(context_x, context_y, test_x, return_density=True, **kwargs)
            avg_log_likelihood = pred_dist.log_prob(test_t_tensor) / test_t_tensor.shape[0]
            rmse = torch.mean(torch.pow(pred_dist.mean - test_t_tensor, 2)).sqrt()

            pred_dist_vect = self._vectorize_pred_dist(pred_dist)
            calibr_error = self._calib_error(pred_dist_vect, test_t_tensor)

            return avg_log_likelihood.cpu().item(), rmse.cpu().item(), calibr_error.cpu().item()

    def eval_datasets(self, test_tuples, **kwargs):
        """
        Computes the average test log likelihood, the rmse and the calibration error over multiple test datasets

        Args:
            test_tuples: list of test set tuples, i.e. [(test_context_x_1, test_context_y_1, test_x_1, test_y_1), ...]

        Returns: (avg_log_likelihood, rmse, calibr_error)

        """

        assert (all([len(valid_tuple) == 4 for valid_tuple in test_tuples]))

        ll_list, rmse_list, calibr_err_list = list(zip(*[self.eval(*test_data_tuple, **kwargs) for test_data_tuple in test_tuples]))

        return np.mean(ll_list), np.mean(rmse_list), np.mean(calibr_err_list)

    def confidence_intervals(self, context_x, context_y, test_x, confidence=0.9, **kwargs):
        pred_dist = self.predict(context_x, context_y, test_x, return_density=True, **kwargs)
        pred_dist = self._vectorize_pred_dist(pred_dist)

        alpha = (1-confidence) / 2
        ucb = pred_dist.icdf(torch.ones(test_x.size) * (1-alpha))
        lcb = pred_dist.icdf(torch.ones(test_x.size) * alpha)
        return ucb, lcb

    def _calib_error(self, pred_dist_vectorized, test_t_tensor):
       return _calib_error(pred_dist_vectorized, test_t_tensor)

    def _vectorize_pred_dist(self, pred_dist):
        raise NotImplementedError

    def _compute_normalization_stats(self, meta_train_tuples):
        X_stack, Y_stack = list(zip(*[_handle_input_dimensionality(x_train, y_train) for x_train, y_train in meta_train_tuples]))
        X, Y = np.concatenate(X_stack, axis=0), np.concatenate(Y_stack, axis=0)

        if self.normalize_data:
            self.x_mean, self.y_mean = np.mean(X, axis=0), np.mean(Y, axis=0)
            self.x_std, self.y_std = np.std(X, axis=0), np.std(Y, axis=0)
        else:
            self.x_mean, self.y_mean = np.zeros(X.shape[1]), np.zeros(Y.shape[1])
            self.x_std, self.y_std = np.ones(X.shape[1]), np.ones(Y.shape[1])

    def _normalize_data(self, X, Y=None):
        assert hasattr(self, "x_mean") and hasattr(self, "x_std"), "requires computing normalization stats beforehand"
        assert hasattr(self, "y_mean") and hasattr(self, "y_std"), "requires computing normalization stats beforehand"

        X_normalized = (X - self.x_mean[None, :]) / self.x_std[None, :]

        if Y is None:
            return X_normalized
        else:
            Y_normalized = (Y - self.y_mean[None, :]) / self.y_std[None, :]
            return X_normalized, Y_normalized

    def _check_meta_data_shapes(self, meta_train_data):
        for i in range(len(meta_train_data)):
            meta_train_data[i] = _handle_input_dimensionality(*meta_train_data[i])
        self.input_dim = meta_train_data[0][0].shape[-1]
        self.output_dim = meta_train_data[0][1].shape[-1]

        assert all([self.input_dim == train_x.shape[-1] and self.output_dim == train_t.shape[-1] for train_x, train_t in meta_train_data])

    def _prepare_data_per_task(self, x_data, y_data, flatten_y=True):
        # a) make arrays 2-dimensional
        x_data, y_data = _handle_input_dimensionality(x_data, y_data)

        # b) normalize data
        x_data, y_data = self._normalize_data(x_data, y_data)

        if flatten_y:
            assert y_data.shape[1] == 1
            y_data = y_data.flatten()

        # c) convert to tensors
        x_tensor = torch.from_numpy(x_data).float().to(device)
        y_tensor = torch.from_numpy(y_data).float().to(device)

        return x_tensor, y_tensor

def _calib_error(pred_dist_vectorized, test_t_tensor):
    cdf_vals = pred_dist_vectorized.cdf(test_t_tensor)

    num_points = test_t_tensor.shape[0]
    conf_levels = torch.linspace(0.05, 0.95, 20)
    emp_freq_per_conf_level = torch.sum(cdf_vals[:, None] <= conf_levels, dim=0).float() / num_points

    calib_rmse = torch.sqrt(torch.mean((emp_freq_per_conf_level - conf_levels)**2))
    return calib_rmse