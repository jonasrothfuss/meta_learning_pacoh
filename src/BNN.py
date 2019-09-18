import torch
import time
import numpy as np
import pyro

from src.models import NeuralNetwork
from src.util import _handle_input_dimensionality, get_logger
from pyro.distributions import Normal, Uniform
from pyro.infer import SVI, Trace_ELBO
import torch.nn.functional as F



class BayesianNeuralNetwork:

    def __init__(self, train_x, train_t, lr_params=1e-3, layer_sizes=(32, 32), epochs=2000, batch_size=64,
                 weight_prior_scale=0.1, optimizer='Adam', random_seed=None):
        """
        Bayesian Neural Network

        Args:
            train_x: (ndarray) train inputs - shape: (n_sampls, ndim_x)
            train_t: (ndarray) train targets - shape: (n_sampls, ndim_y)
            lr_params: (float) learning rate for prior parameters
            layer_sizes: (tuple) hidden layer sizes of NN
            epochs: (int) number of training epochs
            batch_size: (int) number of training points per mini-batch
            optimizer: (str) type of optimizer to use - must be either 'Adam' or 'SGD'
            random_seed: (int) seed for pytorch
        """
        pyro.set_rng_seed(random_seed)

        self.logger = get_logger()

        assert optimizer in ['Adam', 'SGD']

        self.lr_params, self.epochs, self.batch_size = lr_params, epochs, batch_size
        self.likelihood_scale = 1.0

        # Convert the data into pytorch tensors
        train_x, train_t = _handle_input_dimensionality(train_x, train_t)
        input_dim, output_dim = train_x.shape[-1], train_t.shape[-1]
        self.train_x_tensor = torch.from_numpy(train_x).contiguous().float()
        self.train_t_tensor = torch.from_numpy(train_t).contiguous().float()

        dataset = torch.utils.data.TensorDataset(self.train_x_tensor, self.train_t_tensor)
        self.batch_sampler = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # setup neural network
        nn_prefix = str(self.__hash__())[-6:] + '_'
        self.net = NeuralNetwork(input_dim=input_dim, output_dim=output_dim, layer_sizes=layer_sizes, prefix=nn_prefix)

        # setup probabilistic model -> prior * likelihood
        def model(x_data, y_data):
            priors = {}
            for name, param in self.net.named_parameters():
                if 'bias' in name:
                    priors[name] = Normal(loc=torch.zeros_like(param), scale=1000 * torch.ones_like(param)).to_event(1)
                if 'weight' in name:
                    priors[name] = Normal(loc=torch.zeros_like(param), scale=weight_prior_scale * torch.ones_like(param)).to_event(2)

            # lift module parameters to random variables sampled from the priors
            lifted_module = pyro.random_module("module", self.net, priors)
            lifted_reg_model = lifted_module()

            #TODO: learnable scale ??

            with pyro.plate("map", len(x_data)):
                # run the nn forward on data
                prediction_mean = lifted_reg_model(x_data)
                # condition on the observed data
                pyro.sample("obs", Normal(prediction_mean, self.likelihood_scale), obs=y_data)

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
            self.optimizer = pyro.optim.Adam({'lr':lr_params})
        elif optimizer == 'SGD':
            self.optimizer = pyro.optim.SGD({'lr':lr_params})
        else:
            raise NotImplementedError('Optimizer must be Adam or SGD')

        self.svi = SVI(model, guide, self.optimizer, loss=Trace_ELBO())

        self.fitted = False

    def fit(self, verbose=True, valid_x=None, valid_t=None):

        assert (valid_x is None and valid_t is None) or (
                    isinstance(valid_x, np.ndarray) and isinstance(valid_x, np.ndarray))

        t = time.time()
        epoch_losses = []
        for epoch in range(self.epochs):
            for j, (x_batch, y_batch) in enumerate(self.batch_sampler):
                loss = self.svi.step(x_batch, y_batch)
                epoch_losses.append(loss)
                if epoch % 100 == 0 and j == 0 and verbose:
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

    def predict(self, test_x, n_posterior_samples=1000, return_tensors=False):
        """
        computes the predictive distribution, that is, samples NNs from the posterior and computes the mean and std of
        NN predictions for the provided x

        Args:
            test_x: (ndarray) query input data of shape (n_samples, ndim_x)

        Returns:
            (pred_mean, pred_std) predicted mean and standard deviation
        """
        if test_x.ndim == 1:
            test_x = np.expand_dims(test_x, axis=-1)

        with torch.no_grad():
            test_x_tensor = torch.from_numpy(test_x).contiguous().float()

            sampled_models = [self.guide(None, None) for _ in range(n_posterior_samples)]
            preds = torch.stack([model(test_x_tensor) for model in sampled_models], dim=0)

        mean, std = torch.mean(preds, dim=0), torch.std(preds, dim=0)
        if return_tensors:
            return mean, std
        else:
            return mean.numpy(), std.numpy()


    def eval(self, test_x, test_t, n_posterior_samples=100):
        """
        Computes the average test log likelihood and the rmse on test data

        Args:
            test_x: (ndarray) test input data of shape (n_samples, ndim_x)
            test_t: (ndarray) test target data of shape (n_samples, ndim_y)

        Returns: (avg_log_likelihood, rmse)

        """
        test_x, test_t = _handle_input_dimensionality(test_x, test_t)

        test_t_tensor = torch.from_numpy(test_x).contiguous().float()

        with torch.no_grad():
            mean, _ = self.predict(test_x, n_posterior_samples=n_posterior_samples, return_tensors=True)

            pred_dist = torch.distributions.normal.Normal(loc=mean, scale=self.likelihood_scale)
            avg_log_likelihood = torch.mean(pred_dist.log_prob(test_t_tensor))
            rmse = torch.mean(torch.sum(torch.pow(mean - test_t_tensor, 2), dim=-1)).sqrt()

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


