import torch
import time
import copy
from collections import OrderedDict
from pyro.distributions import Normal
import numpy as np

from src.svgd import SVGD, RBF_Kernel, IMQSteinKernel
from src.util import _handle_input_dimensionality
from src.abstract import RegressionModel
from src.models import NeuralNetworkVectorized, EqualWeightedMixtureDist, AffineTransformedDistribution

class ProbabilisticNN:

    def __init__(self, size_in, size_out, n_train_samples, layer_sizes=(32, 32), likelihood_std=0.1,
                 nonlinearlity=torch.tanh, weight_prior_std=1.0, bias_prior_std=3.0):

        self.size_in = size_in
        self.size_out = size_out
        self.n_train_samples = n_train_samples

        # Neural Network
        self.net = NeuralNetworkVectorized(size_in, size_out, layer_sizes=layer_sizes, nonlinearlity=nonlinearlity)

        # Prior
        self.priors = OrderedDict()
        prior_stds = []
        for name, shape in self.net.parameter_shapes().items():
            if "weight" in name:
                prior_stds.append(weight_prior_std * torch.ones(shape))
            elif "bias" in name:
                prior_stds.append(bias_prior_std * torch.ones(shape))
            else:
                raise NotImplementedError
        prior_stds = torch.cat(prior_stds, dim=-1)
        assert prior_stds.ndim == 1
        self.prior = Normal(torch.zeros(prior_stds.shape), prior_stds).to_event(1)

        # Likelihood Distribution
        self.likelihood_dist = Normal(torch.zeros(size_out), likelihood_std * torch.ones(size_out)).to_event(1)

    @property
    def event_shape(self):
        return self.prior.event_shape

    def sample_params_from_prior(self):
       return self.prior.sample()

    def get_forward_fn(self, params):
        reg_model = copy.deepcopy(self.net)
        reg_model.set_parameters_as_vector(params)
        return reg_model

    def sample_fn_from_prior(self):
        sampled_params = self.sample_params_from_prior()
        reg_model = self.get_forward_fn(sampled_params)
        return reg_model

    def _log_prob_prior(self, params):
        return self.prior.log_prob(params)

    def _log_prob_likelihood(self, params, x_data, y_data):
        fn = self.get_forward_fn(params)
        y_residuals = fn(x_data) - y_data
        log_probs = self.likelihood_dist.log_prob(y_residuals)
        return torch.tensor(self.n_train_samples) * torch.mean(log_probs, dim=1) # sum over batch size

    def log_prob(self, params, x_data, y_data):
        return self._log_prob_prior(params) + self._log_prob_likelihood(params, x_data, y_data)


class BayesianNeuralNetworkSVGD(RegressionModel):

    def __init__(self, train_x, train_t, lr=1e-2, layer_sizes=(32, 32), epochs=10000, batch_size=64,
                 kernel='RBF', bandwidth=None, num_particles=10, optimizer='Adam', normalize_data=True,
                 weight_prior_std=1.0, likelihood_std=0.1, random_seed=None):
        """
        Bayesian Neural Network with SVGD posterior inference

        Args:
            train_x: (ndarray) train inputs - shape: (n_sampls, ndim_x)
            train_t: (ndarray) train targets - shape: (n_sampls, ndim_y)
            lr: (float) learning rate for prior parameters
            layer_sizes: (tuple) hidden layer sizes of NN
            epochs: (int) number of training epochs
            batch_size: (int) number of training points per mini-batch
            kernel (std): SVGD kernel, either 'RBF' or 'IMQ'
            bandwidth (float): bandwidth of kernel, if None the bandwidth is chosen via heuristic
            num_particles (int): number of posterior particles
                        optimizer: (str) type of optimizer to use - must be either 'Adam' or 'SGD'
            optimizer: (str) type of optimizer to use - must be either 'Adam' or 'SGD'
            normalize_data (bool): whether to normalize the data
            weight_prior_std (float): std of Gaussian prior on weights
            likelihood_std (float): std of Gaussian likelihood
            random_seed: (int) seed for pytorch
        """
        super().__init__(normalize_data=normalize_data, random_seed=random_seed)

        assert optimizer in ['Adam', 'SGD']
        assert kernel in ['RBF', 'IMQ']

        self.lr, self.epochs, self.batch_size = lr, epochs, batch_size

        """ ------Data handling ------ """
        self.train_x_tensor, self.train_t_tensor = self.initial_data_handling(train_x, train_t)

        # setup batch sampler
        dataset = torch.utils.data.TensorDataset(self.train_x_tensor, self.train_t_tensor)
        self.batch_sampler = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # setup probabilistic NN
        self.likelihood_std = likelihood_std
        self.posterior_dist = ProbabilisticNN(self.input_dim, self.output_dim, self.n_train_samples,
                                              layer_sizes=layer_sizes, weight_prior_std=weight_prior_std,
                                              likelihood_std=likelihood_std)

        # setup inference procedure
        if optimizer == 'Adam':
            self.optimizer = torch.optim.Adam
        elif optimizer == 'SGD':
            self.optimizer = torch.optim.SGD
        else:
            raise NotImplementedError('Optimizer must be Adam or SGD')

        self.particles = torch.stack([self.posterior_dist.sample_params_from_prior() for _ in range(num_particles)], dim=0)


        if kernel == 'RBF':
            kernel = RBF_Kernel(bandwidth=bandwidth)
        elif kernel == 'IMQ':
            kernel = IMQSteinKernel(bandwidth=bandwidth)
        else:
            raise NotImplemented

        self.svgd = SVGD(self.posterior_dist, kernel, self.optimizer([self.particles], lr=self.lr))

        self.fitted = False

    def fit(self, valid_x=None, valid_t=None, verbose=True, log_period=500):

        assert (valid_x is None and valid_t is None) or (
                    isinstance(valid_x, np.ndarray) and isinstance(valid_x, np.ndarray))

        t = time.time()
        for epoch in range(self.epochs):
            for j, (x_batch, y_batch) in enumerate(self.batch_sampler):
                self.svgd.step(self.particles, x_batch, y_batch)
                if epoch % log_period == 0 and j == 0 and verbose:
                    message = 'Epoch %d/%d - Time %.2f sec' % (epoch+1, self.epochs, time.time()-t)
                    t = time.time()

                    # if validation data is provided  -> compute the valid log-likelihood
                    if valid_x is not None:
                        valid_ll, valid_rmse = self.eval(valid_x, valid_t)
                        message += ' - Valid-LL: %.3f - Valid-RMSE: %.3f' % (valid_ll, valid_rmse)

                    self.logger.info(message)


        self.fitted = True

    def predict(self, test_x, return_density=False, mode='prob'):
        """
        computes the predictive distribution, that is, samples NNs from the posterior and computes the mean and std of
        NN predictions for the provided x

        Args:
            test_x: (ndarray) query input data of shape (n_samples, ndim_x)
            return_density: (bool) whether to return a predictive density object or predictive mean & std as numpy array
            mode: (std) either 'prob' or 'reg'.
                    'prob' mode: the likelihood distribution is taken into account
                    'reg' mode: only the mean & std of the prediction mean are taken into account

        Returns:
            (pred_mean, pred_std) predicted mean and standard deviation or a torch density object
        """
        assert mode in ["prob", 'reg']
        assert not return_density or mode == 'prob', 'can only return density in probabilistic mode'

        if test_x.ndim == 1:
            test_x = np.expand_dims(test_x, axis=-1)

        with torch.no_grad():
            test_x_normalized = self._normalize_data(test_x)
            test_x_tensor = torch.from_numpy(test_x_normalized).contiguous().float()

            pred_fn = self.posterior_dist.get_forward_fn(self.particles)
            preds = pred_fn(test_x_tensor)

            if mode == 'prob':
                pred_dists = [Normal(preds[i], self.likelihood_std).to_event(1) for i in range(preds.shape[0])]
                pred_dist = EqualWeightedMixtureDist(pred_dists)
                pred_dist = AffineTransformedDistribution(pred_dist, normalization_mean=self.y_mean,
                                                                normalization_std=self.y_std)
                if return_density:
                    return pred_dist
                else:
                    pred_mean = pred_dist.mean.numpy()
                    pred_std = pred_dist.stddev.numpy()
                    return pred_mean, pred_std

            elif mode == 'pred':
                pred_mean, pred_std = torch.mean(preds, dim=0), torch.std(preds, dim=0)
                pred_mean, pred_std = self._unnormalize_pred(pred_mean, pred_mean)
                return pred_mean.numpy(), pred_mean.numpy()
            else:
                raise NotImplementedError

    def eval(self, test_x, test_t):
        """
        Computes the average test log likelihood and the rmse on test data

        Args:
            test_x: (ndarray) test input data of shape (n_samples, ndim_x)
            test_t: (ndarray) test target data of shape (n_samples, ndim_y)

        Returns: (avg_log_likelihood, rmse)

        """

        # convert to tensors
        test_x, test_t = _handle_input_dimensionality(test_x, test_t)
        test_t_tensor = torch.from_numpy(test_t).contiguous().float()

        with torch.no_grad():
            pred_dist = self.predict(test_x, return_density=True)
            avg_log_likelihood = torch.sum(pred_dist.log_prob(test_t_tensor)) / test_t_tensor.shape[0]
            rmse = torch.mean(torch.pow(pred_dist.mean - test_t_tensor, 2)).sqrt()

            return avg_log_likelihood.item(), rmse.item()

    def state_dict(self): #TODO
        state_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        return state_dict

    def load_state_dict(self, state_dict): #TODO
        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    """ ----------------------- """
    """    1) Generate data     """
    """ ----------------------- """

    for norm_data in [True, False]:
        with torch.no_grad():
            n_train_samples = 10
            n_test_samples = 200

            torch.manual_seed(23)
            x_data = torch.normal(mean=-1, std=2.0, size=(n_train_samples + n_test_samples, 1))
            W = torch.tensor([[0.6]])
            b = torch.tensor([-1])
            y_data = x_data.matmul(W.T) + b + torch.normal(mean=0.0, std=0.1, size=(n_train_samples + n_test_samples, 1))

            x_data_train, x_data_test = x_data[:n_train_samples].numpy(), x_data[n_train_samples:].numpy()
            y_data_train, y_data_test = y_data[:n_train_samples].numpy(), y_data[n_train_samples:].numpy()

        """ Train BNN """
        bnn = BayesianNeuralNetworkSVGD(x_data_train, y_data_train, bandwidth=1.0, num_particles=50, layer_sizes=(32, 64, 32),
                                        weight_prior_std=1.0, lr=0.01, batch_size=10, epochs=5000, normalize_data=norm_data,
                                        random_seed=22, likelihood_std=0.1)
        bnn.fit(x_data_train, y_data_train)
        print("normalization=%s"%str(bnn.normalize_data))
        print(bnn.x_mean, bnn.x_std, bnn.y_mean, bnn.y_std)

        x_plot = np.expand_dims(np.linspace(-10, 5, num=100), axis=-1)
        y_pred_mean, y_pred_std = bnn.predict(x_plot)
        y_pred_mean, y_pred_std = y_pred_mean.squeeze(), y_pred_std.squeeze()
        plt.scatter(x_data_train[:, 0], y_data_train)
        plt.plot(x_plot, y_pred_mean)
        plt.fill_between(x_plot.squeeze(), y_pred_mean - y_pred_std, y_pred_mean + y_pred_std, alpha=0.5)
        plt.title("SVGD ESTIMATOR (normalization=%s"%str(bnn.normalize_data))
        plt.show()
