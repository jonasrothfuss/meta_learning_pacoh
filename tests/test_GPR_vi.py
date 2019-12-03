import unittest
import numpy as np

from src.GPR_mll_vi import GPRegressionLearnedVI
from src.GPR_meta_vi import GPRegressionMetaLearnedVI
import torch
import pyro


class TestGPR_mll_vi(unittest.TestCase):

    def setUp(self):
        ## --- generate toy data --- #

        torch.manual_seed(22)
        np.random.seed(25)

        # train
        n_train_points = 60
        self.x_train = np.linspace(-2, 2, num=n_train_points)

        self.y_train_zero = self.x_train * 0 + np.random.normal(scale=0.02, size=self.x_train.shape)
        self.y_train_two = self.x_train * 0 + 2 + np.random.normal(scale=0.02, size=self.x_train.shape)
        self.y_train_sin = np.sin(4* self.x_train)

        # test
        n_test_points = 80

        self.x_test = np.linspace(-2.1, 2.1, num=n_test_points)

        self.y_test_zero = self.x_test * 0 + np.random.normal(scale=0.02, size=self.x_test.shape)
        self.y_test_two = self.x_test * 0 + 2 + np.random.normal(scale=0.02, size=self.x_test.shape)
        self.y_test_sin = np.sin(4 * self.x_test)

    def test_random_seed_consistency(self):
        for covar_module in ['NN', 'SE']:
            for mean_module in ['NN', 'constant']:
                gpr_model_1 = GPRegressionLearnedVI(self.x_train, self.y_train_two, num_iter_fit=3,
                                                    mean_module=mean_module, covar_module=covar_module, random_seed=22)

                gpr_model_1.fit()
                t_predict_1 = gpr_model_1.predict(self.x_test, n_posterior_samples=10)

                gpr_model_2 = GPRegressionLearnedVI(self.x_train, self.y_train_two, num_iter_fit=3,
                                                    mean_module=mean_module, covar_module=covar_module, random_seed=22)

                gpr_model_2.fit()
                t_predict_2 = gpr_model_2.predict(self.x_test, n_posterior_samples=10)

                self.assertTrue(np.array_equal(t_predict_1, t_predict_2))

    def test_basic_learning(self):

        for mean_module, covar_module in [('constant', 'SE'), ('NN', 'NN')]:
            gpr_model = GPRegressionLearnedVI(self.x_train, self.y_train_sin, num_iter_fit=100, prior_factor=0.01,
                                              mean_module=mean_module, covar_module=covar_module, random_seed=25,
                                              mean_nn_layers=(8, 8), kernel_nn_layers=(8, 8))

            gpr_model.fit(valid_x=self.x_train, valid_t=self.y_train_sin)
            ll, rmse, _ = gpr_model.eval(self.x_train, self.y_train_sin)

            pyro.clear_param_store()

            gpr_model_long_train = GPRegressionLearnedVI(self.x_train, self.y_train_sin, prior_factor=0.01,
                                                         num_iter_fit=2000, mean_module=mean_module, covar_module=covar_module,
                                                         random_seed=25, mean_nn_layers=(8, 8), kernel_nn_layers=(8, 8))
            gpr_model_long_train.fit(valid_x=self.x_train, valid_t=self.y_train_sin)
            ll_long, rmse_long, _ = gpr_model_long_train.eval(self.x_train, self.y_train_sin)
            pyro.clear_param_store()

            print(ll, rmse)
            print(ll_long, rmse_long)
            self.assertGreater(ll_long, ll)
            self.assertLess(rmse_long, rmse)

class TestGPR_mll_meta_vi(unittest.TestCase):

    def setUp(self):
        ## --- generate toy data --- #

        torch.manual_seed(22)
        np.random.seed(23)

        #sample_data = lambda n_samples: sample_sinusoid_regression_data(n_samples_train, amp_low=0.9, amp_high=1.1, slope_std=0.01)
        # meta train
        n_train_datasets = 10
        n_samples_train = 5
        self.train_data_tuples = [sample_data_nonstationary(n_samples_train) for _ in range(n_train_datasets)]

        # test
        n_test_datasets = 10
        n_samples_test_context = 5
        n_samples_test = 50

        test_data = [sample_data_nonstationary(n_samples_test_context + n_samples_test) for _ in
                            range(n_test_datasets)]

        # split data into test_context and test_valid
        self.test_data_tuples = [(x[:n_samples_test_context], t[:n_samples_test_context],
                                  x[n_samples_test_context:], t[n_samples_test_context:]) for (x, t) in test_data]


    def test_random_seed_consistency(self):
        for mean_module, covar_module in [('constant', 'SE'), ('NN', 'NN')]:
            pyro.clear_param_store()
            gp_meta_1 = GPRegressionMetaLearnedVI(self.train_data_tuples[:2], num_iter_fit=5,
                                                covar_module=covar_module, mean_module=mean_module, random_seed=22)

            gp_meta_1.meta_fit(valid_tuples=self.test_data_tuples)

            x_context, t_context, x_test, _ = self.test_data_tuples[0]
            t_predict_1 = gp_meta_1.predict(x_context, t_context, x_test)


            gp_meta_2 = GPRegressionMetaLearnedVI(self.train_data_tuples[:2], num_iter_fit=5,
                                                covar_module=covar_module, mean_module=mean_module, random_seed=22)


            gp_meta_2.meta_fit(valid_tuples=self.test_data_tuples)
            t_predict_2 = gp_meta_2.predict(x_context, t_context, x_test)

            self.assertTrue(np.array_equal(t_predict_1, t_predict_2))

    def test_basic_learning(self):
        for mean_module, covar_module in [('constant', 'SE'), ('NN', 'NN')]:
            pyro.clear_param_store()
            gp_meta_1 = GPRegressionMetaLearnedVI(self.train_data_tuples, num_iter_fit=2, prior_factor=0.001,
                                                  covar_module=covar_module, mean_module=mean_module, cov_type='diag',
                                                  mean_nn_layers=(8, 8), kernel_nn_layers=(8, 8), random_seed=23)

            gp_meta_1.meta_fit(valid_tuples=self.test_data_tuples)
            ll1, rmse1, _ = gp_meta_1.eval_datasets(self.test_data_tuples)
            ll1_map, rmse1_map, _ = gp_meta_1.eval_datasets(self.test_data_tuples, mode='Bayes')

            gp_meta_2 = GPRegressionMetaLearnedVI(self.train_data_tuples, num_iter_fit=5000, prior_factor=0.001,
                                                  covar_module=covar_module, mean_module=mean_module, cov_type='diag',
                                                  mean_nn_layers=(8, 8), kernel_nn_layers=(8, 8), random_seed=23)

            gp_meta_2.meta_fit(valid_tuples=self.test_data_tuples, log_period=1000)
            ll2, rmse2, _ = gp_meta_2.eval_datasets(self.test_data_tuples)
            ll2_map, rmse2_map, _ = gp_meta_2.eval_datasets(self.test_data_tuples, mode='Bayes')


            self.assertGreater(ll2, ll1)
            self.assertGreater(ll2_map, ll1_map)
            self.assertLess(rmse2, rmse1)
            self.assertLess(rmse2_map, rmse1_map)



""" --- helper functions for data generation ---"""


def sample_data_nonstationary(size=1):
    def _sample_fun():
        slope = np.random.normal(loc=1, scale=0.2)
        freq = lambda x: 1 + np.abs(x)
        mean = lambda x: slope * x
        return lambda x: (mean(x) + np.sin(freq(x) * x)) / 5

    func = _sample_fun()
    X = np.random.uniform(-5, 5, size=(size, 1))
    Y = func(X)
    return X, Y

if __name__ == '__main__':
    unittest.main()