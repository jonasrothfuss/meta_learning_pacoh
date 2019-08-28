import numpy as np
import gpytorch
import torch
from numbers import Number
from matplotlib import pyplot as plt

X_LOW = -5
X_HIGH = 5

Y_HIGH = 2.5
Y_LOW = -2.5



class MetaDataset():

    def __init__(self, random_state=None):
        if random_state is None:
            self.random_state = np.random
        else:
            self.random_state = random_state

    def generate_meta_train_data(self, n_tasks, n_samples):
        pass

    def generate_meta_test_data(self, n_tasks, n_samples_context, n_samples_test):
        pass


class SinusoidMetaDataset(MetaDataset):

    def __init__(self, amp_low=0.8, amp_high=1.2,
                 period_low=1.0, period_high=1.0,
                 x_shift_mean=0.0, x_shift_std=0.0,
                 y_shift_mean=5.0, y_shift_std=0.05,
                 slope_mean=0.2, slope_std=0.05,
                 noise_std=0.01, x_low=-5, x_high=5, random_state=None):

        super().__init__(random_state)
        assert y_shift_std >= 0 and noise_std >= 0, "std must be non-negative"
        self.amp_low, self.amp_high= amp_low, amp_high
        self.period_low, self.period_high = period_low, period_high
        self.y_shift_mean, self.y_shift_std = y_shift_mean, y_shift_std
        self.x_shift_mean, self.x_shift_std = x_shift_mean, x_shift_std
        self.slope_mean, self.slope_std = slope_mean, slope_std
        self.noise_std = noise_std
        self.x_low, self.x_high = x_low, x_high

    def generate_meta_test_data(self, n_tasks, n_samples_context, n_samples_test):
        meta_test_tuples = []
        for i in range(n_tasks):
            f = self._sample_sinusoid()
            X = self.random_state.uniform(self.x_low, self.x_high, size=(n_samples_context + n_samples_test, 1))
            Y = f(X)
            meta_test_tuples.append((X[n_samples_context:], Y[n_samples_context:], X[:n_samples_context], Y[:n_samples_context]))

        return meta_test_tuples

    def generate_meta_train_data(self, n_tasks, n_samples):
        meta_train_tuples = []
        for i in range(n_tasks):
            f = self._sample_sinusoid()
            X = self.random_state.uniform(self.x_low, self.x_high, size=(n_samples, 1))
            Y = f(X)
            meta_train_tuples.append((X, Y))
        return meta_train_tuples

    def _sample_sinusoid(self):
        amplitude = self.random_state.uniform(self.amp_low, self.amp_high)
        x_shift = self.random_state.normal(loc=self.x_shift_mean, scale=self.x_shift_std)
        y_shift = self.random_state.normal(loc=self.y_shift_mean, scale=self.y_shift_std)
        slope = self.random_state.normal(loc=self.slope_mean, scale=self.slope_std)
        period = self.random_state.uniform(self.period_low, self.period_high)
        return lambda x: slope * x + amplitude * np.sin(period * (x - x_shift)) + y_shift


""" sinusoidal data """

# sinusoid function + gaussian noise
def _sinusoid(x, amplitude=1.0, period=1.0, x_shift=0.0, y_shift=0.0, slope=0.0, noise_std=0.0):
    f = slope*x + amplitude * np.sin(period * (x - x_shift)) + y_shift
    noise = np.random.normal(0, scale=noise_std, size=f.shape)
    return f + noise


def sample_sinusoid_classification_data(size=1, amp_low=0.5, amp_high=1.5, y_shift_mean=5.0, y_shift_std=0.3,
                                        slope_mean=0.2, slope_std=0.05, noise_std=0.1):
    """ samples classification data with a sinusoidal separation function
           Args:
                amp_low (float): min amplitude value
                amp_high (float): max amplitude value
                y_shift_mean (float): mean of Gaussian from which to sample the y_shift of the sinusoid
                y_shift_std (float): std of Gaussian from which to sample the y_shift of the sinusoid
                slope_mean (float: mean of Gaussian from which to sample the linear slope
                slope_std (float): std of Gaussian from which to sample the linear slope
                noise_std (float): std of the Gaussian observation noise

           Returns:
               (X, t): X is an ndarray of dimensionality (size, 2) and t an ndarray of dimensionality (size,)
                       containing {-1, 1} entries
       """

    if isinstance(size, Number):
        size = (int(size),)  # convert to tuple

    f = _sample_sinusoid(amp_low=amp_low, amp_high=amp_high, y_shift_mean=y_shift_mean, y_shift_std=y_shift_std,
                         slope_mean=slope_mean, slope_std=slope_std, noise_std=noise_std)
    X_1 = np.random.uniform(X_LOW, X_HIGH, size=size + (1,)) # first data dimension
    Y = np.random.uniform(y_shift_mean + Y_LOW, y_shift_mean + Y_HIGH, size=size + (1,)) # second data dimension
    target = np.sign(Y - f(X_1)).flatten()
    X = np.concatenate([X_1, Y], axis=-1)
    assert np.all(np.logical_or(target == 1.0, target -1.0))
    return X, target



if __name__ == "__main__":
    X, Y = sample_sinusoid_classification_data(100)
    print(X.shape)
    plt.scatter(X[:,0], X[:,1], c=Y)
    plt.show()