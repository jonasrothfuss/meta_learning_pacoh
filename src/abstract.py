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
        self.train_x_tensor = None
        self.train_t_tensor = None

        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed+1)


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

    def initial_data_handling(self, train_x, train_t):
        train_x, train_t = _handle_input_dimensionality(train_x, train_t)
        self.input_dim, self.output_dim = train_x.shape[-1], train_t.shape[-1]
        self.n_train_samples = train_x.shape[0]

        # b) normalize data to exhibit zero mean and variance
        self._compute_normalization_stats(train_x, train_t)
        train_x_normalized, train_t_normalized = self._normalize_data(train_x, train_t)

        # c) Convert the data into pytorch tensors
        self.train_x_tensor = torch.from_numpy(train_x_normalized).contiguous().float().to(device)
        self.train_t_tensor = torch.from_numpy(train_t_normalized).contiguous().float().to(device)

        return self.train_x_tensor, self.train_t_tensor