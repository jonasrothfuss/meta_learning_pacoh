import numpy as np
import torch
from src.util import get_logger, _handle_input_dimensionality

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

        X_normalized = (X - self.x_mean) / self.x_std

        if Y is None:
            return X_normalized
        else:
            Y_normalized = (Y - self.y_mean) / self.y_std
            return X_normalized, Y_normalized

    def initial_data_handling(self, train_x, train_t):
        train_x, train_t = _handle_input_dimensionality(train_x, train_t)
        self.input_dim, self.output_dim = train_x.shape[-1], train_t.shape[-1]
        self.n_train_samples = train_x.shape[0]

        # b) normalize data to exhibit zero mean and variance
        self._compute_normalization_stats(train_x, train_t)
        train_x_normalized, train_t_normalized = self._normalize_data(train_x, train_t)

        # c) Convert the data into pytorch tensors
        self.train_x_tensor = torch.from_numpy(train_x_normalized).contiguous().float()
        self.train_t_tensor = torch.from_numpy(train_t_normalized).contiguous().float().flatten()

        return self.train_x_tensor, self.train_t_tensor