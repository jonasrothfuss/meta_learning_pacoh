import unittest
import numpy as np

from meta_learn.GPR_mll import GPRegressionLearned
from meta_learn.GPR_meta_mll import GPRegressionMetaLearned
from gpytorch.kernels import CosineKernel
import torch


class TestGPR_mll(unittest.TestCase):

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
        gpr_model_1 = GPRegressionLearned(self.x_train, self.y_train_two, learning_mode='both',
                                        num_iter_fit=5, mean_module='NN', covar_module='NN', random_seed=22)

        gpr_model_2 = GPRegressionLearned(self.x_train, self.y_train_two, learning_mode='both',
                                        num_iter_fit=5, mean_module='NN', covar_module='NN', random_seed=22)

        gpr_model_1.fit()
        t_predict_1 = gpr_model_1.predict(self.x_test)

        gpr_model_2.fit()
        t_predict_2 = gpr_model_2.predict(self.x_test)

        self.assertTrue(np.array_equal(t_predict_1, t_predict_2))

    def test_serializable(self):
        torch.manual_seed(40)
        np.random.seed(22)
        import itertools
        # check that more datasets improve performance
        for mean_module, covar_module in itertools.product(['constant', 'NN'], ['SE', 'NN']):

            gpr_model = GPRegressionLearned(self.x_train, self.y_train_two, learning_mode='both',
                                            num_iter_fit=10, mean_module=mean_module, covar_module='NN', random_seed=22)
            gpr_model.fit()
            pred_1 = gpr_model.predict(self.x_train)

            gpr_model2 = GPRegressionLearned(self.x_train, self.y_train_two, learning_mode='both',
                                            num_iter_fit=1, mean_module=mean_module, covar_module='NN', random_seed=345)
            gpr_model2.fit()
            pred_2 = gpr_model2.predict(self.x_train)


            file = ('/tmp/test_torch_serialization.pkl')
            torch.save(gpr_model.state_dict(), file)
            gpr_model2.load_state_dict(torch.load(file))
            pred_3 = gpr_model2.predict(self.x_train)
            assert not np.array_equal(pred_1, pred_2)
            assert np.array_equal(pred_1, pred_3)


    def test_mean_learning(self):
        for mean_module in ['NN']:

            gpr_model_vanilla = GPRegressionLearned(self.x_train, self.y_train_sin, learning_mode='vanilla', num_iter_fit=20,
                                                    mean_module='constant', covar_module='SE')
            gpr_model_vanilla.fit()

            gpr_model_learn_mean = GPRegressionLearned(self.x_train, self.y_train_sin, learning_mode='learn_mean', num_iter_fit=100,
                                                       mean_module=mean_module, covar_module='SE', mean_nn_layers=(16, 16))
            gpr_model_learn_mean.fit()

            ll_vanilla, rmse_vanilla, _ = gpr_model_vanilla.eval(self.x_train, self.y_train_two)
            ll_mean, rmse_mean, _ = gpr_model_learn_mean.eval(self.x_train, self.y_train_sin)

            print(ll_mean, ll_vanilla)
            print(rmse_mean, rmse_vanilla)
            self.assertGreater(ll_mean, ll_vanilla)
            self.assertLess(rmse_mean, rmse_vanilla)

    def test_kernel_learning_COS(self):

        for learning_mode in ['learn_kernel', 'both']:

            gpr_model_vanilla = GPRegressionLearned(self.x_train, self.y_train_sin, learning_mode='vanilla',
                                                    num_iter_fit=1,
                                                    mean_module='constant', covar_module=CosineKernel())
            gpr_model_vanilla.fit()


            gpr_model_learn_kernel = GPRegressionLearned(self.x_train, self.y_train_sin, learning_mode='learn_kernel',
                                                    num_iter_fit=500,
                                                    mean_module='constant', covar_module=CosineKernel())

            print(gpr_model_learn_kernel.model.covar_module.lengthscale)
            gpr_model_learn_kernel.fit(valid_x=self.x_train, valid_t=self.y_train_sin)
            print(gpr_model_learn_kernel.model.covar_module.lengthscale)

            ll_vanilla, rmse_vanilla, _ = gpr_model_vanilla.eval(self.x_train, self.y_train_sin)
            ll_kernel, rmse_kernel, _ = gpr_model_learn_kernel.eval(self.x_train, self.y_train_sin)

            print('learning_mode', learning_mode)
            print(ll_kernel, ll_vanilla)
            print(rmse_kernel, rmse_vanilla)
            self.assertGreater(ll_kernel, ll_vanilla)
            self.assertLess(rmse_kernel, rmse_vanilla)

    def test_kernel_learning_NN(self):

        for learning_mode in ['learn_kernel', 'both']:

            gpr_model_vanilla = GPRegressionLearned(self.x_train, self.y_train_sin, learning_mode='learn_kernel',
                                                    num_iter_fit=1,
                                                    mean_module='zero', covar_module='NN')
            gpr_model_vanilla.fit()


            gpr_model_learn_kernel = GPRegressionLearned(self.x_train, self.y_train_sin, learning_mode=learning_mode,
                                                    num_iter_fit=500, mean_module='constant', covar_module='NN',
                                                    kernel_nn_layers=(16, 16), mean_nn_layers=(16, 16))
            gpr_model_learn_kernel.fit(valid_x=self.x_train, valid_t=self.y_train_sin)

            ll_vanilla, rmse_vanilla, _ = gpr_model_vanilla.eval(self.x_train, self.y_train_sin)
            ll_kernel, rmse_kernel, _ = gpr_model_learn_kernel.eval(self.x_train, self.y_train_sin)

            print('learning_mode', learning_mode)
            print(ll_kernel, ll_vanilla)
            print(rmse_kernel, rmse_vanilla)
            self.assertGreater(ll_kernel, ll_vanilla)
            self.assertLess(rmse_kernel, rmse_vanilla)

class TestGPR_mll_meta(unittest.TestCase):

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
        gp_meta_1 = GPRegressionMetaLearned(self.train_data_tuples[:2], learning_mode='both', num_iter_fit=5,
                                            covar_module='NN', mean_module='NN', random_seed=22)

        gp_meta_2 = GPRegressionMetaLearned(self.train_data_tuples[:2], learning_mode='both', num_iter_fit=5,
                                            covar_module='NN', mean_module='NN', random_seed=22)

        gp_meta_1.meta_fit(valid_tuples=self.test_data_tuples)
        gp_meta_2.meta_fit(valid_tuples=self.test_data_tuples)

        for (x_context, t_context, x_test, _) in self.test_data_tuples[:3]:
            t_predict_1 = gp_meta_1.predict(x_context, t_context, x_test)
            t_predict_2 = gp_meta_2.predict(x_context, t_context, x_test)

            self.assertTrue(np.array_equal(t_predict_1, t_predict_2))

    def test_serializable(self):
        torch.manual_seed(40)
        np.random.seed(22)
        import itertools
        # check that more datasets improve performance
        for mean_module, covar_module in itertools.product(['constant', 'NN'], ['SE', 'NN']):

            gpr_model = GPRegressionMetaLearned(self.train_data_tuples[:3], learning_mode='both',
                                            num_iter_fit=5, mean_module=mean_module, covar_module='NN', random_seed=22)
            gpr_model.meta_fit()
            pred_1 = gpr_model.predict(*self.test_data_tuples[0][:3])

            gpr_model2 = GPRegressionMetaLearned(self.train_data_tuples[:3], learning_mode='both',
                                            num_iter_fit=5, mean_module=mean_module, covar_module='NN', random_seed=25)
            gpr_model2.meta_fit()
            pred_2 = gpr_model2.predict(*self.test_data_tuples[0][:3])


            file = ('/tmp/test_torch_serialization.pkl')
            torch.save(gpr_model.state_dict(), file)
            gpr_model2.load_state_dict(torch.load(file))
            pred_3 = gpr_model2.predict(*self.test_data_tuples[0][:3])
            assert not np.array_equal(pred_1, pred_2)
            assert np.array_equal(pred_1, pred_3)

            torch.manual_seed(25)
            gpr_model.rds_numpy = np.random.RandomState(55)
            gpr_model.meta_fit()
            torch.manual_seed(25)
            gpr_model2.rds_numpy = np.random.RandomState(55)
            gpr_model2.meta_fit()
            pred_1 = gpr_model.predict(*self.test_data_tuples[0][:3])
            pred_2 = gpr_model2.predict(*self.test_data_tuples[0][:3])
            assert np.array_equal(pred_1, pred_2)

    def test_mean_learning_more_datasets(self):
        torch.manual_seed(40)
        # check that more datasets improve performance

        # meta-learning with 2 datasets
        gp_meta = GPRegressionMetaLearned(self.train_data_tuples[:2], learning_mode='both', mean_nn_layers=(16, 16),
                                          kernel_nn_layers=(16, 16), num_iter_fit=3000, covar_module='SE',
                                          mean_module='NN', weight_decay=0.0)
        gp_meta.meta_fit(valid_tuples=self.test_data_tuples)

        test_ll_meta_2, test_rmse_meta_2, _ = gp_meta.eval_datasets(self.test_data_tuples)
        print('Test log-likelihood meta (2 datasets):', test_ll_meta_2)

        # meta-learning with 10 datasets
        gp_meta = GPRegressionMetaLearned(self.train_data_tuples, learning_mode='both', mean_nn_layers=(16, 16),
                                          kernel_nn_layers=(16, 16), num_iter_fit=3000, covar_module='SE',
                                          mean_module='NN', weight_decay=0.0)
        gp_meta.meta_fit(valid_tuples=self.test_data_tuples)

        test_ll_meta_10, test_rmse_meta_10, _ = gp_meta.eval_datasets(self.test_data_tuples)
        print('Test log-likelihood meta (10 datasets):', test_ll_meta_10)


        self.assertGreater(test_ll_meta_10, test_ll_meta_2)
        self.assertLess(test_rmse_meta_10, test_rmse_meta_2)


    def test_normal_vs_meta(self):

        # check that meta-learning improves upon normal learned GP
        torch.manual_seed(60)
        num_iter_fit = 1000

        # meta-learning
        gp_meta = GPRegressionMetaLearned(self.train_data_tuples, learning_mode='both', mean_nn_layers=(64, 64),
                                          covar_module='SE', mean_module='NN', weight_decay=0.0, num_iter_fit=num_iter_fit)

        gp_meta.meta_fit(valid_tuples=self.test_data_tuples)

        test_ll_meta, test_rmse_meta, _ = gp_meta.eval_datasets(self.test_data_tuples)
        print('Test log-likelihood meta:', test_ll_meta)

        def fit_eval_gpr(x_context, t_context, x_test, t_test):
            gpr = GPRegressionLearned(x_context, t_context, learning_mode='both', mean_nn_layers=(64, 64),
                                      covar_module='SE', mean_module='NN', weight_decay=0.0, num_iter_fit=num_iter_fit)
            gpr.fit(valid_x=x_test, valid_t=t_test)

            return gpr.eval(x_test, t_test)[0]

        ll_list = [fit_eval_gpr(*data_tuple) for data_tuple in self.test_data_tuples]
        test_ll_normal = np.mean(ll_list)

        print('Test log-likelihood normal:', test_ll_normal)

        self.assertGreater(test_ll_meta, test_ll_normal)


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