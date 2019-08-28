import unittest
import numpy as np

from experiments.data_sim import SinusoidMetaDataset, MNISTRegressionDataset

class TestSinusoidMetaDataset(unittest.TestCase):

    def test_seed_reproducability(self):
        rds = np.random.RandomState(55)
        dataset = SinusoidMetaDataset(random_state=rds)
        data_test_1 = dataset.generate_meta_test_data(n_tasks=2, n_samples_context=5, n_samples_test=10)
        data_train_1 = dataset.generate_meta_train_data(n_tasks=5, n_samples=20)

        rds = np.random.RandomState(55)
        dataset = SinusoidMetaDataset(random_state=rds)
        data_test_2 = dataset.generate_meta_test_data(n_tasks=2, n_samples_context=5, n_samples_test=10)
        data_train_2 = dataset.generate_meta_train_data(n_tasks=5, n_samples=20)

        for test_tuple_1, test_tuple_2 in zip(data_test_1, data_test_2):
            for data_array_1, data_array_2 in zip(test_tuple_1, test_tuple_2):
                assert np.array_equal(data_array_1, data_array_2)

        for train_tuple_1, train_tuple_2 in zip(data_train_1, data_train_2):
            for data_array_1, data_array_2 in zip(train_tuple_1, train_tuple_2):
                assert np.array_equal(data_array_1, data_array_2)

    def test_no_noise(self):
        dataset = SinusoidMetaDataset(
                 amp_low=1.0, amp_high=1.0,
                 period_low=1.0, period_high=1.0,
                 x_shift_mean=0.0, x_shift_std=0.0,
                 y_shift_mean=0.0, y_shift_std=0.0,
                 slope_mean=1.0, slope_std=0.00,
                 noise_std=0.00, x_low=5, x_high=-5,)

        data_tuples = dataset.generate_meta_train_data(n_tasks=2, n_samples=500)

        true_fn = lambda x: x + np.sin(x)

        for data_tuple in data_tuples:
            x_train, y_train = data_tuple
            y_true = true_fn(x_train)
            abs_diff = np.mean(np.abs(y_true - y_train))
            self.assertAlmostEqual(abs_diff, 0.0)


    def test_context_test_consistency(self):
        dataset = SinusoidMetaDataset(noise_std=0.00, x_low=1, x_high=1)

        data_tuples = dataset.generate_meta_test_data(n_tasks=10, n_samples_context=1, n_samples_test=1)

        for data_tuple in data_tuples:
            x_context, y_context, x_test, y_test = data_tuple
            assert np.array_equal(y_context, y_test)
            print(y_context, y_test)


class TestMNISTRegressionDataset(unittest.TestCase):

    def test_seed_reproducability(self):
        rds = np.random.RandomState(55)
        dataset = MNISTRegressionDataset(random_state=rds)
        data_test_1 = dataset.generate_meta_test_data(n_tasks=2, n_samples_context=5, n_samples_test=10)
        data_train_1 = dataset.generate_meta_train_data(n_tasks=5, n_samples=20)

        rds = np.random.RandomState(55)
        dataset = MNISTRegressionDataset(random_state=rds)
        data_test_2 = dataset.generate_meta_test_data(n_tasks=2, n_samples_context=5, n_samples_test=10)
        data_train_2 = dataset.generate_meta_train_data(n_tasks=5, n_samples=20)

        for test_tuple_1, test_tuple_2 in zip(data_test_1, data_test_2):
            for data_array_1, data_array_2 in zip(test_tuple_1, test_tuple_2):
                assert np.array_equal(data_array_1, data_array_2)

        for train_tuple_1, train_tuple_2 in zip(data_train_1, data_train_2):
            for data_array_1, data_array_2 in zip(train_tuple_1, train_tuple_2):
                assert np.array_equal(data_array_1, data_array_2)

    def test_output_shapes_generate_test(self):
        rds = np.random.RandomState(123)
        dataset = MNISTRegressionDataset(random_state=rds)

        for n_tasks in [1, 5]:
            for n_samples_context in [1, 85]:
                for n_samples_test in [-1, 23]:

                    data_test = dataset.generate_meta_test_data(n_tasks=n_tasks, n_samples_context=n_samples_context,
                                                    n_samples_test=n_samples_test)

                    assert len(data_test) == n_tasks

                    for (x_context, t_context, x_test, t_test) in data_test:
                        assert x_context.shape[0] == t_context.shape[0]
                        assert x_context.shape[1] == x_test.shape[1] == 2

                    if n_samples_test == -1:
                        assert x_context.shape[0] + x_test.shape[0] == 28**2

    def test_output_shapes_generate_train(self):
        rds = np.random.RandomState(123)
        dataset = MNISTRegressionDataset(random_state=rds)

        for n_tasks in [24, 2]:
            for n_samples in [1, 85]:
                data_test = dataset.generate_meta_train_data(n_tasks=n_tasks, n_samples=n_samples)

                assert len(data_test) == n_tasks

                for (x_train, t_train) in data_test:
                    assert x_train.shape[0] == t_train.shape[0]
                    assert x_train.shape[1] == 2





        #data_train = dataset.generate_meta_train_data(n_tasks=5, n_samples=20)
