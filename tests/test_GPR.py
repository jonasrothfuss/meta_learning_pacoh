import unittest
import numpy as np

from src.GPR_mll import GPRegressionLearned
from gpytorch.kernels import MaternKernel, CosineKernel
import torch

class TestGPR_mll(unittest.TestCase):

    def setUp(self):
        ## --- generate toy data --- #

        torch.manual_seed(22)

        # train
        n_train_points = 60
        self.x_train = np.linspace(-2, 2, num=n_train_points)

        self.y_train_zero = self.x_train * 0
        self.y_train_two = self.x_train * 0 + 2
        self.y_train_sin = np.sin(4* self.x_train)

        # test
        n_test_points = 80

        self.x_test = np.linspace(-2.1, 2.1, num=n_test_points)

        self.y_test_zero = self.x_test * 0
        self.y_test_two = self.x_test * 0 + 2
        self.y_test_sin = np.sin(4 * self.x_test)

    def test_mean_learning(self):
        for mean_module in ['NN', 'constant']:

            gpr_model_vanilla = GPRegressionLearned(self.x_train, self.y_train_two, learning_mode='vanilla', num_iter_fit=20,
                                                    mean_module='constant', covar_module='SE')
            gpr_model_vanilla.fit()

            gpr_model_learn_mean = GPRegressionLearned(self.x_train, self.y_train_two, learning_mode='learn_mean', num_iter_fit=100,
                                                       mean_module=mean_module, covar_module='SE', mean_nn_layers=(16, 16))
            gpr_model_learn_mean.fit()

            ll_vanilla, rsme_vanilla = gpr_model_vanilla.eval(self.x_train, self.y_train_two)
            ll_mean, rsme_mean = gpr_model_learn_mean.eval(self.x_train, self.y_train_two)

            print(ll_mean, ll_vanilla)
            print(rsme_mean, rsme_vanilla)
            self.assertGreater(ll_mean, ll_vanilla)
            self.assertLess(rsme_mean, rsme_vanilla)

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

            ll_vanilla, rsme_vanilla = gpr_model_vanilla.eval(self.x_train, self.y_train_sin)
            ll_kernel, rsme_kernel = gpr_model_learn_kernel.eval(self.x_train, self.y_train_sin)

            print('learning_mode', learning_mode)
            print(ll_kernel, ll_vanilla)
            print(rsme_kernel, rsme_vanilla)
            self.assertGreater(ll_kernel, ll_vanilla)
            self.assertLess(rsme_kernel, rsme_vanilla)

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

            ll_vanilla, rsme_vanilla = gpr_model_vanilla.eval(self.x_train, self.y_train_sin)
            ll_kernel, rsme_kernel = gpr_model_learn_kernel.eval(self.x_train, self.y_train_sin)

            print('learning_mode', learning_mode)
            print(ll_kernel, ll_vanilla)
            print(rsme_kernel, rsme_vanilla)
            self.assertGreater(ll_kernel, ll_vanilla)
            self.assertLess(rsme_kernel, rsme_vanilla)


if __name__ == '__main__':
    unittest.main()