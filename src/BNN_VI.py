import torch
import time
import numpy as np
import pyro

from src.models import NeuralNetwork, AffineTransformedDistribution, EqualWeightedMixtureDist
from src.util import _handle_input_dimensionality, get_logger
from src.abstract import RegressionModel
from pyro.distributions import Normal, Uniform
from pyro.infer import SVI, Trace_ELBO
import torch.nn.functional as F
from config import device



class BayesianNeuralNetworkVI(RegressionModel):

    def __init__(self, train_x, train_t, lr=1e-3, layer_sizes=(32, 32), epochs=20000, batch_size=64,
                 num_svi_samples=1, optimizer='Adam', normalize_data=True,
                 likelihood_std=0.1, weight_prior_std=1.0, random_seed=None):
        """
        Bayesian Neural Network with fully-factorized Gaussian Variational Distribution

        Args:
            train_x: (ndarray) train inputs - shape: (n_sampls, ndim_x)
            train_t: (ndarray) train targets - shape: (n_sampls, ndim_y)
            lr: (float) learning rate for prior parameters
            layer_sizes: (tuple) hidden layer sizes of NN
            epochs: (int) number of training epochs
            batch_size: (int) number of training points per mini-batch
            num_svi_samples: (int) number of samples from VI posterior to estimate gradients
            optimizer: (str) type of optimizer to use - must be either 'Adam' or 'SGD'
            ormalize_data (bool): whether to normalize the data
            likelihood_std (float): std of Gaussian likelihood
            weight_prior_std (float): std of Gaussian prior on weights
            random_seed: (int) seed for pytorch
        """
        super().__init__(normalize_data=normalize_data, random_seed=random_seed)

        assert optimizer in ['Adam', 'SGD']

        self.lr, self.epochs, self.batch_size = lr, epochs, batch_size
        self.likelihood_std = likelihood_std

        """ ------Data handling ------ """
        self.train_x_tensor, self.train_t_tensor = self.initial_data_handling(train_x, train_t)

        # setup batch sampler
        dataset = torch.utils.data.TensorDataset(self.train_x_tensor, self.train_t_tensor)
        self.batch_sampler = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # setup neural network
        nn_prefix = str(self.__hash__())[-6:] + '_'
        self.net = NeuralNetwork(input_dim=self.input_dim, output_dim=self.output_dim, layer_sizes=layer_sizes, prefix=nn_prefix).to(device)

        # setup probabilistic model -> prior * likelihood
        def model(x_data, y_data):
            priors = {}
            for name, param in self.net.named_parameters():
                if 'bias' in name:
                    priors[name] = Normal(loc=torch.zeros_like(param).to(device),
                                          scale=1000 * torch.ones_like(param).to(device)).to_event(1)
                if 'weight' in name:
                    priors[name] = Normal(loc=torch.zeros_like(param).to(device),
                                          scale=weight_prior_std * torch.ones_like(param).to(device)).to_event(2)

            # lift module parameters to random variables sampled from the priors
            lifted_module = pyro.random_module("module", self.net, priors)
            lifted_reg_model = lifted_module()

            #TODO: learnable scale ??

            with pyro.plate("map", len(x_data)):
                # run the nn forward on data
                prediction_mean = lifted_reg_model(x_data)
                # condition on the observed data
                pyro.sample("obs", Normal(prediction_mean, torch.tensor(self.likelihood_std).to(device)), obs=y_data)

            return prediction_mean

        # setup variational posterior (fully factorized gaussian)
        def guide(x_data, y_data):
            vi_posteriors = {}
            for name, param in self.net.named_parameters():
                assert 'bias' in name or 'weight' in name
                vi_mu_param = pyro.param("%s_mu" % name, torch.randn_like(param))
                vi_sigma_param = F.softplus(pyro.param("%s_sigma_raw" % name, torch.randn_like(param)))
                vi_posteriors[name] = Normal(loc=vi_mu_param, scale=vi_sigma_param)

            lifted_module = pyro.random_module(name="module", nn_module=self.net, prior=vi_posteriors)
            return lifted_module()

        self.model = model
        self.guide = guide

        # setup inference procedure
        if optimizer == 'Adam':
            self.optimizer = pyro.optim.Adam({'lr': self.lr})
        elif optimizer == 'SGD':
            self.optimizer = pyro.optim.SGD({'lr': self.lr})
        else:
            raise NotImplementedError('Optimizer must be Adam or SGD')

        self.svi = SVI(model, guide, self.optimizer, loss=Trace_ELBO(num_particles=num_svi_samples))

        self.fitted = False

    def fit(self, valid_x=None, valid_t=None, verbose=True, log_period=500):

        assert (valid_x is None and valid_t is None) or (
                    isinstance(valid_x, np.ndarray) and isinstance(valid_x, np.ndarray))

        t = time.time()
        epoch_losses = []
        for epoch in range(self.epochs):
            for j, (x_batch, y_batch) in enumerate(self.batch_sampler):
                loss = self.svi.step(x_batch, y_batch)
                epoch_losses.append(loss)
                if epoch % log_period == 0 and j == 0 and verbose:
                    avg_loss = np.mean(epoch_losses) / self.batch_size
                    message = 'Epoch %d/%d - Loss: %.6f - Time %.2f sec' % (epoch+1, self.epochs, avg_loss, time.time()-t)
                    epoch_losses = []
                    t = time.time()

                    # if validation data is provided  -> compute the valid log-likelihood
                    if valid_x is not None:
                        valid_ll, valid_rmse = self.eval(valid_x, valid_t, n_posterior_samples=100)
                        message += ' - Valid-LL: %.3f - Valid-RMSE: %.3f' % (valid_ll, valid_rmse)

                    self.logger.info(message)


        self.fitted = True

    def predict(self, test_x, n_posterior_samples=100, return_density=False, mode='prob'):
        """
        computes the predictive distribution, that is, samples NNs from the posterior and computes the mean and std of
        NN predictions for the provided x

        Args:
            test_x: (ndarray) query input data of shape (n_samples, ndim_x)
            n_posterior_samples: (int) number of samples from posterior to average over
            return_density: (bool) whether to return a predictive density object or predictive mean & std as numpy array
            mode: (std) either 'prob' or 'reg'.
                    'prob' mode: the likelihood distribution is taken into account
                    'reg' mode: only the mean & std of the prediction mean are taken into account

        Returns:
            (pred_mean, pred_std) predicted mean and standard deviation across predictions a torch density object
        """
        if test_x.ndim == 1:
            test_x = np.expand_dims(test_x, axis=-1)

        with torch.no_grad():
            test_x_normalized = self._normalize_data(test_x)
            test_x_tensor = torch.from_numpy(test_x_normalized).contiguous().float().to(device)

            sampled_models = [self.guide(None, None) for _ in range(n_posterior_samples)]

            if mode == 'prob':
                pred_dists = [Normal(model(test_x_tensor), self.likelihood_std).to_event(1)
                              for model in sampled_models]
                pred_dist = EqualWeightedMixtureDist(pred_dists)
                pred_dist = AffineTransformedDistribution(pred_dist, normalization_mean=self.y_mean,
                                                          normalization_std=self.y_std)
                if return_density:
                    return pred_dist
                else:
                    pred_mean = pred_dist.mean.cpu().numpy()
                    pred_std = pred_dist.stddev.cpu().numpy()
                    return pred_mean, pred_std

            elif mode == 'pred':
                preds = torch.stack([model(test_x_tensor) for model in sampled_models], dim=0)
                mean, std = torch.mean(preds, dim=0), torch.std(preds, dim=0)
                mean, std = self._unnormalize_pred(mean, std)

                return mean.cpu().numpy(), std.cpu().numpy()

            else:
                raise NotImplementedError

    def eval(self, test_x, test_t, n_posterior_samples=100):
        """
        Computes the average test log likelihood and the rmse on test data

        Args:
            test_x: (ndarray) test input data of shape (n_samples, ndim_x)
            test_t: (ndarray) test target data of shape (n_samples, ndim_y)
            n_posterior_samples: (int) number of samples from posterior to average over

        Returns: (avg_log_likelihood, rmse)

        """

        # convert to tensors
        test_x, test_t = _handle_input_dimensionality(test_x, test_t)
        test_t_tensor = torch.from_numpy(test_t).contiguous().float().to(device)

        with torch.no_grad():
            pred_dist = self.predict(test_x, n_posterior_samples=n_posterior_samples, return_density=True)
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
    import torch
    import numpy as np
    from matplotlib import pyplot as plt


    for norm_data in [True, False]:
        with torch.no_grad():
            n_train_samples = 60
            n_test_samples = 200

            torch.manual_seed(23)
            x_data = torch.normal(mean=-1, std=2.0, size=(n_train_samples + n_test_samples, 1))
            W = torch.tensor([[0.6]])
            b = torch.tensor([-1])
            y_data = x_data.matmul(W.T) + b + torch.normal(mean=0.0, std=0.1, size=(n_train_samples + n_test_samples, 1))

            x_data_train, x_data_test = x_data[:n_train_samples].numpy(), x_data[n_train_samples:].numpy()
            y_data_train, y_data_test = y_data[:n_train_samples].numpy(), y_data[n_train_samples:].numpy()

        """ Train BNN """
        bnn = BayesianNeuralNetworkVI(x_data_train, y_data_train, layer_sizes=(16, 16),
                                    weight_prior_std=1.0,  lr=0.01, batch_size=20, epochs=8000, normalize_data=norm_data,
                                    random_seed=22, num_svi_samples=5)

        bnn.fit(x_data_train, y_data_train, log_period=200)

        x_plot = np.expand_dims(np.linspace(-10, 5, num=100), axis=-1)
        y_pred_mean, y_pred_std = bnn.predict(x_plot)
        y_pred_mean, y_pred_std = y_pred_mean.squeeze(), y_pred_std.squeeze()
        plt.scatter(x_data_train[:, 0], y_data_train)
        plt.plot(x_plot, y_pred_mean)
        plt.fill_between(x_plot.squeeze(), y_pred_mean - y_pred_std, y_pred_mean + y_pred_std, alpha=0.5)
        plt.title("SVI ESTIMATOR (normalization=%s"%str(bnn.normalize_data))
        plt.show()