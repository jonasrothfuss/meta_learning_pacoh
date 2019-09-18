import unittest
import numpy as np

import torch

from src.BNN import BayesianNeuralNetwork

class TestBNN(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(22)
        np.random.seed(25)
        # train
        n_train_points = 200
        self.x_train = np.linspace(-2.5, 2.5, num=n_train_points)

        self.y_train_sin = np.sin(self.x_train) + 0.1 * np.abs(self.x_train) * np.random.normal(0, 1, size=self.x_train.shape[0])

        # test
        n_test_points = 80

        self.x_test = np.linspace(-2.5, 2.5, num=n_test_points)
        self.y_test_sin = np.sin(self.x_test)


    def test_random_seed_consistency(self):
        gpr_model_1 = BayesianNeuralNetwork(self.x_train, self.y_train_sin, epochs=5, layer_sizes=(8, 8), random_seed=22)

        gpr_model_1.fit()
        t_predict_1 = gpr_model_1.predict(self.x_test)

        gpr_model_2 = BayesianNeuralNetwork(self.x_train, self.y_train_sin, epochs=5, layer_sizes=(8, 8), random_seed=22)
        gpr_model_2.fit()
        t_predict_2 = gpr_model_2.predict(self.x_test)

        self.assertTrue(np.array_equal(t_predict_1, t_predict_2))

    def test_basic_fit_eval(self):
        bnn = BayesianNeuralNetwork(self.x_train, self.y_train_sin, epochs=2000, layer_sizes=(16, 16), random_seed=25)
        bnn.fit()

        mean, std = bnn.predict(self.x_test, n_posterior_samples=200)
        assert mean.shape[0] == self.x_test.shape[0]
        assert std.shape[0] == self.x_test.shape[0]


        x_tail = np.concatenate([self.x_test[:10], self.x_test[-10:]])
        y_tail = np.concatenate([self.y_test_sin[:10], self.y_test_sin[-10:]])

        x_center = self.x_test[10:-10]
        y_center = self.y_test_sin[10:-10]

        ll_tails, rmse_tails = bnn.eval(x_tail, y_tail)
        ll_center, rmse_center = bnn.eval(x_center, y_center)

        self.assertGreater(rmse_tails, rmse_center)
        self.assertLess(ll_tails, ll_center)
