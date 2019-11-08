import unittest
import numpy as np

import torch

from src.BNN_VI import BayesianNeuralNetworkVI
from src.BNN_SVGD import BayesianNeuralNetworkSVGD

class TestBNNVI(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(22)
        np.random.seed(25)
        # train
        n_train_points = 50
        self.x_train = np.sort(np.random.normal(-1, 1, size=n_train_points))
        self.y_train = -3 + 2 * self.x_train + 0.2 * np.random.normal(0, 1, size=self.x_train.shape[0])

        n_test_points = 100
        self.x_test = np.linspace(-5, 3, num=n_test_points)
        self.y_test = -3 + 2 * self.x_test

    def test_random_seed_consistency(self):
        gpr_model_1 = BayesianNeuralNetworkVI(self.x_train, self.y_train, epochs=5, layer_sizes=(8, 8), random_seed=22)

        gpr_model_1.fit()
        t_predict_1 = gpr_model_1.predict(self.x_test)

        gpr_model_2 = BayesianNeuralNetworkVI(self.x_train, self.y_train, epochs=5, layer_sizes=(8, 8), random_seed=22)
        gpr_model_2.fit()
        t_predict_2 = gpr_model_2.predict(self.x_test)

        self.assertTrue(np.array_equal(t_predict_1, t_predict_2))

    def test_basic_fit_eval(self):
        bnn = BayesianNeuralNetworkVI(self.x_train, self.y_train, epochs=5000, layer_sizes=(16, 16), num_svi_samples=1,
                                      weight_prior_std=2.0, lr=1e-2, random_seed=25, likelihood_std=0.2)

        bnn.fit(valid_x=self.x_train, valid_t=self.y_train, log_period=2000)
        mean, std = bnn.predict(self.x_train, n_posterior_samples=200)

        # check shape
        assert mean.shape[0] == self.x_train.shape[0]
        assert std.shape[0] == self.x_train.shape[0]

        # check fit
        pred_residual = np.mean(np.abs(mean.squeeze() - self.y_train))
        print(pred_residual)
        assert pred_residual < 0.5

        # check uncertainties
        x_tail = np.concatenate([self.x_test[:10], self.x_test[-10:]])
        y_tail = np.concatenate([self.y_test[:10], self.y_test[-10:]])

        x_center = self.x_test[10:-10]
        y_center = self.y_test[10:-10]

        ll_tails, rmse_tails = bnn.eval(x_tail, y_tail)
        ll_center, rmse_center = bnn.eval(x_center, y_center)

        self.assertGreater(rmse_tails, rmse_center)
        self.assertLess(ll_tails, ll_center)


class TestBNNSVGD(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(22)
        np.random.seed(25)
        # train
        n_train_points = 50
        self.x_train = np.sort(np.random.normal(-1, 1, size=n_train_points))
        self.y_train = -3 + 2 * self.x_train + 0.2 * np.random.normal(0, 1, size=self.x_train.shape[0])

        n_test_points = 100
        self.x_test = np.linspace(-5, 3, num=n_test_points)
        self.y_test = -3 + 2 * self.x_test

    def test_random_seed_consistency(self):
        model_1 = BayesianNeuralNetworkSVGD(self.x_train, self.y_train, epochs=5, layer_sizes=(8, 8), random_seed=22)

        model_1.fit()
        t_predict_1 = model_1.predict(self.x_train)

        gpr_model_2 = BayesianNeuralNetworkSVGD(self.x_train, self.y_train, epochs=5, layer_sizes=(8, 8), random_seed=22)
        gpr_model_2.fit()
        t_predict_2 = gpr_model_2.predict(self.x_train)

        self.assertTrue(np.array_equal(t_predict_1, t_predict_2))

    def test_basic_fit_eval(self):
        bnn = BayesianNeuralNetworkSVGD(self.x_train, self.y_train, epochs=2000, layer_sizes=(16, 16),
                                        weight_prior_std=2.0, lr=1e-2, num_particles=20, random_seed=25, likelihood_std=0.2)

        bnn.fit(valid_x=self.x_train, valid_t=self.y_train)
        mean, std = bnn.predict(self.x_train,)

        # check shapes
        assert mean.shape[0] == self.x_train.shape[0]
        assert std.shape[0] == self.x_train.shape[0]

        # check rmse
        _, rmse = bnn.eval(self.x_train, self.y_train)
        assert rmse < 0.4

        x_tail = np.concatenate([self.x_test[:10], self.x_test[-10:]])
        y_tail = np.concatenate([self.y_test[:10], self.y_test[-10:]])

        x_center = self.x_test[10:-10]
        y_center = self.y_test[10:-10]

        ll_tails, rmse_tails = bnn.eval(x_tail, y_tail)
        ll_center, rmse_center = bnn.eval(x_center, y_center)

        self.assertGreater(rmse_tails, rmse_center)
        self.assertLess(ll_tails, ll_center)