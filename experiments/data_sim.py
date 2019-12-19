import mnist
import numpy as np
import pandas as pd
from scipy.stats import truncnorm
import os


X_LOW = -5
X_HIGH = 5

Y_HIGH = 2.5
Y_LOW = -2.5

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
MNIST_DIR = os.path.join(DATA_DIR, 'mnist')
PHYSIONET_DIR = os.path.join(DATA_DIR, 'physionet2012')

class MetaDataset():

    def __init__(self, random_state=None):
        if random_state is None:
            self.random_state = np.random
        else:
            self.random_state = random_state


    def generate_meta_train_data(self, n_tasks: int, n_samples: int) -> list:
        raise NotImplementedError


    def generate_meta_test_data(self, n_tasks:int, n_samples_context: int, n_samples_test: int) -> list:
        raise NotImplementedError

        
class PhysionetDataset(MetaDataset):
    
    def __init__(self, random_state=None, variable_id=0, dtype=np.float32, physionet_dir=None):
        super().__init__(random_state)
        self.dtype = dtype
        if physionet_dir is not None:
            self.data_dir = physionet_dir
        elif PHYSIONET_DIR is not None:
            self.data_dir = PHYSIONET_DIR
        else:
            raise ValueError("No data directory provided.")
        self.variable_list = ['GCS', 'Urine', 'HCT', 'BUN', 'Creatinine', 'DiasABP']

        assert variable_id < len(self.variable_list), "Unknown variable ID"
        self.variable = self.variable_list[variable_id]

        self.data_path = os.path.join(self.data_dir, "set_a_merged.h5")

        with pd.HDFStore(self.data_path, mode='r') as hdf_file:
            self.keys = hdf_file.keys()
            
            
    def generate_meta_train_data(self, n_tasks, n_samples=47):
        """
        Samples n_tasks patients and returns measurements from the variable
        with the ID variable_id. n_samples defines in this case the cut-off
        of hours on the ICU, e.g., n_samples=24 returns all measurements that
        were taken in the first 24 hours. Generally, those will be less than
        24 measurements. If there are less than n_tasks patients that have
        any measurements of variable variable_id before hour n_samples, the
        returned list will contain less than n_tasks tuples.
        """

        assert n_tasks < 500, "We don't have that many tasks"
        assert n_samples < 48, "We don't have that many samples"

        meta_train_tuples = []
        
        for patient in self.keys:
            df = pd.read_hdf(self.data_path, patient, mode='r')[self.variable].dropna()
            times = df.index.values.astype(self.dtype)
            values = df.values.astype(self.dtype)
            times_context = [time for time in times if time <= n_samples]
            if len(times_context) > 0:
                times_context = np.array(times_context, dtype=self.dtype)
                values_context = values[:len(times_context)]
                if values_context.shape[0] >= 4:
                    meta_train_tuples.append((times_context, values_context))
                else:
                    continue
            if len(meta_train_tuples) >= n_tasks:
                break

        return meta_train_tuples
        
        
    def generate_meta_test_data(self, n_tasks, n_samples_context=24,
                                n_samples_test=-1, variable_id=0):
        """
        Samples n_tasks patients and returns measurements from the variable
        with the ID variable_id. n_samples defines in this case the cut-off
        of hours on the ICU, e.g., n_samples=24 returns all measurements that
        were taken in the first 24 hours. Generally, those will be less than
        24 measurements. The remaining measurements are returned as test points,
        i.e., n_samples_test is unused.
        If there are less than n_tasks patients that have any measurements
        of variable variable_id before hour n_samples, the
        returned list will contain less than n_tasks tuples.
        """

        assert n_tasks < 1000, "We don't have that many tasks"
        assert n_samples_context < 48, "We don't have that many samples"

        meta_test_tuples = []

        for patient in reversed(self.keys):
            df = pd.read_hdf(self.data_path, patient, mode='r')[self.variable].dropna()
            times = df.index.values.astype(self.dtype)
            values = df.values.astype(self.dtype)
            times_context = [time for time in times if time <= n_samples_context]
            times_test = [time for time in times if time > n_samples_context]
            if len(times_context) > 0 and len(times_test) > 0:
                times_context = np.array(times_context, dtype=self.dtype)
                times_test = np.array(times_test, dtype=self.dtype)
                values_context = values[:len(times_context)]
                values_test = values[len(times_context):]
                if values_context.shape[0] >= 4:
                    meta_test_tuples.append((times_context, values_context,
                                              times_test, values_test))
                else:
                    continue
            if len(meta_test_tuples) >= n_tasks:
                break

        return meta_test_tuples


class MNISTRegressionDataset(MetaDataset):

    def __init__(self, random_state=None, dtype=np.float32):
        super().__init__(random_state)
        self.dtype = dtype

        mnist_dir = MNIST_DIR if os.path.isdir(MNIST_DIR) else None

        self.train_images = mnist.download_and_parse_mnist_file('train-images-idx3-ubyte.gz', target_dir=mnist_dir)
        self.test_images = mnist.download_and_parse_mnist_file('t10k-images-idx3-ubyte.gz', target_dir=mnist_dir)

        self.train_images = self.train_images / 255.0
        self.test_images = self.train_images / 255.0

    def generate_meta_train_data(self, n_tasks, n_samples):

        meta_train_tuples = []

        train_indices = self.random_state.choice(self.train_images.shape[0], size=n_tasks,  replace=False)

        for idx in train_indices:
            x_context, t_context, _, _ = self._image_to_context_transform(self.train_images[idx], n_samples)
            meta_train_tuples.append((x_context, t_context))

        return meta_train_tuples

    def generate_meta_test_data(self, n_tasks, n_samples_context, n_samples_test=-1):

        meta_test_tuples = []

        test_indices = self.random_state.choice(self.test_images.shape[0], size=n_tasks, replace=False)

        for idx in test_indices:
            x_context, t_context, x_test, t_test = self._image_to_context_transform(self.train_images[idx],
                                                                                    n_samples_context)

            # chose only subsam
            if n_samples_test > 0 and n_samples_test < x_test.shape[0]:
                indices = self.random_state.choice(x_test.shape[0], size=n_samples_test, replace=False)
                x_test, t_test = x_test[indices], t_test[indices]

            meta_test_tuples.append((x_context, t_context, x_test, t_test))

        return meta_test_tuples

    def _image_to_context_transform(self, image, num_context_points):
        assert image.ndim == 2 and image.shape[0] == image.shape[1]
        image_size = image.shape[0]
        assert num_context_points <= image_size ** 2

        xx, yy = np.meshgrid(np.arange(image_size), np.arange(image_size))
        indices = np.array(list(zip(xx.flatten(), yy.flatten())))
        context_indices = indices[self.random_state.choice(image_size ** 2, size=num_context_points, replace=False)]
        context_values = image[tuple(zip(*context_indices))]

        dtype_indices = {'names': ['f{}'.format(i) for i in range(2)],
                         'formats': 2 * [indices.dtype]}

        # indices that have not been used as context
        test_indices_structured = np.setdiff1d(indices.view(dtype_indices), context_indices.view(dtype_indices))
        test_indices = test_indices_structured.view(indices.dtype).reshape(-1, 2)

        test_values = image[tuple(zip(*test_indices))]

        return (np.array(context_indices, dtype=self.dtype), np.array(context_values, dtype=self.dtype),
                np.array(test_indices, dtype=self.dtype), np.array(test_values, dtype=self.dtype))


class SinusoidDataset(MetaDataset):

    def __init__(self, amp_low=0.7, amp_high=1.3,
                 period_low=1.5, period_high=1.5,
                 x_shift_mean=0.0, x_shift_std=0.1,
                 y_shift_mean=5.0, y_shift_std=0.1,
                 slope_mean=0.5, slope_std=0.2,
                 noise_std=0.1, x_low=-5, x_high=5, random_state=None):

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
        assert n_samples_test > 0
        meta_test_tuples = []
        for i in range(n_tasks):
            f = self._sample_sinusoid()
            X = self.random_state.uniform(self.x_low, self.x_high, size=(n_samples_context + n_samples_test, 1))
            Y = f(X) + self.noise_std * self.random_state.normal(size=f(X).shape)
            meta_test_tuples.append((X[:n_samples_context], Y[:n_samples_context], X[n_samples_context:], Y[n_samples_context:]))

        return meta_test_tuples

    def generate_meta_train_data(self, n_tasks, n_samples):
        meta_train_tuples = []
        for i in range(n_tasks):
            f = self._sample_sinusoid()
            X = self.random_state.uniform(self.x_low, self.x_high, size=(n_samples, 1))
            Y = f(X) + self.noise_std * self.random_state.normal(size=f(X).shape)
            meta_train_tuples.append((X, Y))
        return meta_train_tuples

    def _sample_sinusoid(self):
        amplitude = self.random_state.uniform(self.amp_low, self.amp_high)
        x_shift = self.random_state.normal(loc=self.x_shift_mean, scale=self.x_shift_std)
        y_shift = self.random_state.normal(loc=self.y_shift_mean, scale=self.y_shift_std)
        slope = self.random_state.normal(loc=self.slope_mean, scale=self.slope_std)
        period = self.random_state.uniform(self.period_low, self.period_high)
        return lambda x: slope * x + amplitude * np.sin(period * (x - x_shift)) + y_shift


class SinusoidNonstationaryDataset(MetaDataset):

    def __init__(self, noise_std=0.0,  x_low=-5, x_high=5, random_state=None):

        super().__init__(random_state)
        self.noise_std = noise_std
        self.x_low, self.x_high = x_low, x_high

    def generate_meta_test_data(self, n_tasks, n_samples_context, n_samples_test):
        assert n_samples_test > 0
        meta_test_tuples = []
        for i in range(n_tasks):
            f = self._sample_fun()
            X = self.random_state.uniform(self.x_low, self.x_high, size=(n_samples_context + n_samples_test, 1))
            Y = f(X)
            meta_test_tuples.append((X[:n_samples_context], Y[:n_samples_context], X[n_samples_context:], Y[n_samples_context:]))

        return meta_test_tuples

    def generate_meta_train_data(self, n_tasks, n_samples):
        meta_train_tuples = []
        for i in range(n_tasks):
            f = self._sample_fun()
            X = self.random_state.uniform(self.x_low, self.x_high, size=(n_samples, 1))
            Y = f(X)
            meta_train_tuples.append((X, Y))
        return meta_train_tuples

    def _sample_fun(self):
        intersect = self.random_state.normal(loc=-2., scale=0.2)
        slope = self.random_state.normal(loc=1, scale=0.3)
        freq = lambda x: 1 + np.abs(x)
        mean = lambda x: intersect + slope * x
        return lambda x: mean(x) + np.sin(freq(x) * x) + self.random_state.normal(loc=0, scale=self.noise_std, size=x.shape)


class GPFunctionsDataset(MetaDataset):

    def __init__(self, noise_std=0.1, lengthscale=1.0, mean=0.0, x_low=-5, x_high=5, random_state=None):
        self.noise_std, self.lengthscale, self.mean = noise_std, lengthscale, mean
        self.x_low, self.x_high = x_low, x_high
        super().__init__(random_state)

    def generate_meta_test_data(self, n_tasks, n_samples_context, n_samples_test):
        assert n_samples_test > 0
        meta_test_tuples = []
        for i in range(n_tasks):
            X = self.random_state.uniform(self.x_low, self.x_high, size=(n_samples_context + n_samples_test, 1))
            Y = self._gp_fun_from_prior(X)
            meta_test_tuples.append(
                (X[:n_samples_context], Y[:n_samples_context], X[n_samples_context:], Y[n_samples_context:]))

        return meta_test_tuples

    def generate_meta_train_data(self, n_tasks, n_samples):
        meta_train_tuples = []
        for i in range(n_tasks):
            X = self.random_state.uniform(self.x_low, self.x_high, size=(n_samples, 1))
            Y = self._gp_fun_from_prior(X)
            meta_train_tuples.append((X, Y))
        return meta_train_tuples

    def _gp_fun_from_prior(self, X):
        assert X.ndim == 2

        n = X.shape[0]

        def kernel(a, b, lengthscale):
            sqdist = np.sum(a ** 2, 1).reshape(-1, 1) + np.sum(b ** 2, 1) - 2 * np.dot(a, b.T)
            return np.exp(-.5 * (1 / lengthscale) * sqdist)

        K_ss = kernel(X, X, self.lengthscale)
        L = np.linalg.cholesky(K_ss + 1e-8 * np.eye(n))
        f = self.mean + np.dot(L, self.random_state.normal(size=(n, 1)))
        y = f + self.random_state.normal(scale=self.noise_std, size=f.shape)
        return y


class CauchyDataset(MetaDataset):

    def __init__(self, noise_std=0.05, ndim_x=2, random_state=None):
        self.noise_std = noise_std
        self.ndim_x = ndim_x
        super().__init__(random_state)

    def generate_meta_train_data(self, n_tasks, n_samples):
        meta_train_tuples = []
        for i in range(n_tasks):
            X = truncnorm.rvs(-3, 2, loc=0, scale=2.5, size=(n_samples, self.ndim_x), random_state=self.random_state)
            Y = self._gp_fun_from_prior(X)
            meta_train_tuples.append((X, Y))
        return meta_train_tuples

    def generate_meta_test_data(self, n_tasks, n_samples_context, n_samples_test):
        assert n_samples_test > 0
        meta_test_tuples = []
        for i in range(n_tasks):
            X = truncnorm.rvs(-3, 2, loc=0, scale=2.5, size=(n_samples_context + n_samples_test, self.ndim_x), random_state=self.random_state)
            Y = self._gp_fun_from_prior(X)
            meta_test_tuples.append(
                (X[:n_samples_context], Y[:n_samples_context], X[n_samples_context:], Y[n_samples_context:]))

        return meta_test_tuples

    def _mean(self, x):
        loc1 = -1 * np.ones(x.shape[-1])
        loc2 = 2 * np.ones(x.shape[-1])
        cauchy1 = 1 / (np.pi * (1 + (np.linalg.norm(x - loc1, axis=-1))**2))
        cauchy2 = 1 / (np.pi * (1 + (np.linalg.norm(x - loc2, axis=-1))**2))
        return 6 * cauchy1 + 3 * cauchy2 + 1

    def _gp_fun_from_prior(self, X):
        assert X.ndim == 2

        n = X.shape[0]

        def kernel(a, b, lengthscale):
            sqdist = np.sum(a ** 2, 1).reshape(-1, 1) + np.sum(b ** 2, 1) - 2 * np.dot(a, b.T)
            return np.exp(-.5 * (1 / lengthscale) * sqdist)

        K_ss = kernel(X, X, 0.5)
        L = np.linalg.cholesky(K_ss + 1e-8 * np.eye(n))
        f = self._mean(X) + np.dot(L, self.random_state.normal(scale=0.2, size=(n, 1))).flatten()
        y = f + self.random_state.normal(scale=self.noise_std, size=f.shape)
        return y.reshape(-1, 1)


if __name__ == "__main__":
    x = np.linspace(-5, 5, num=200)

    #dataset = SinusoidDataset()


    #dataset = SinusoidNonstationaryDataset()
    #dataset = SinusoidDataset(noise_std=0.0, x_shift_std=0.3)


    from matplotlib import pyplot as plt


    for i in range(5):
        dataset = PhysionetDataset(variable_id=i)
        meta_data = dataset.generate_meta_train_data(n_tasks=400, n_samples=47)
        data_size = [x.shape[0] for x, y in meta_data]
        variable_name = dataset.variable_list[i]
        plt.hist(data_size)
        plt.title(variable_name)
        plt.show()
        meta_test_data = dataset.generate_meta_test_data(n_tasks=600, n_samples_context=10, n_samples_test=50)


    # for x, y in meta_data:
    #     # func = dataset._sample_fun()
    #     # y = func(x)
    #     idx = np.argsort(x, axis=0).flatten()
    #     plt.plot(x[idx], y[idx])
    # plt.show()
