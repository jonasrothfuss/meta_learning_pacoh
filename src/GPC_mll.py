import torch
import gpytorch
import time

from src.models import LearnedGPClassificationModel, NeuralNetwork


class GPClassificationLearned:

    def __init__(self, train_x, train_t, mode='both', lr_params=1e-3, lr_vi=1e-2, weight_decay=1e-3, feature_dim=2,
                 num_iter_fit=1000, covar_module=None, mean_module=None, mean_n_layers=(64, 64), kernel_nn_layers=(64, 64)):

        assert mode in ['learned_mean', 'learned_kernel', 'both', 'vanilla']
        assert lr_params <= lr_vi, "parameter learning rate should be smaller than VI learning rate"

        self.lr_params, self.lr_vi, self.weight_decay, self.num_iter_fit = lr_params, lr_vi, weight_decay, num_iter_fit

        # Convert the data into pytorch tensors
        input_dim = train_x.shape[-1]
        self.train_x_tensor = torch.from_numpy(train_x).contiguous().float()
        self.train_t_tensor = torch.from_numpy(train_t).contiguous().float()

        # Setup model
        self.parameters = []

        if mode in ["learned kern", "both"]:
            kernel_function = NeuralNetwork(input_dim=input_dim, output_dim=feature_dim, layer_sizes=mean_n_layers)
            self.parameters.append({'params': kernel_function.parameters(),'lr': self.lr_params, 'weight_decay': self.weight_decay})
        else:
            kernel_function = None
            feature_dim = input_dim

        if mode in ["learned mean", "both"]:
            mean_function = NeuralNetwork(input_dim=input_dim, output_dim=1, layer_sizes=kernel_nn_layers)
            self.parameters.append({'params': mean_function.parameters(), 'lr': self.lr_params, 'weight_decay': self.weight_decay})
        else:
            mean_function = None

        self.likelihood = gpytorch.likelihoods.BernoulliLikelihood()

        self.model = LearnedGPClassificationModel(self.train_x_tensor, learned_kernel=kernel_function,
                                                  learned_mean=mean_function,
                                                  covar_module=covar_module, mean_module=mean_module,
                                                  feature_dim=feature_dim)

        self.parameters.append({'params': self.model.covar_module.hyperparameters(), 'lr': self.lr_params})
        self.parameters.append({'params': self.model.mean_module.hyperparameters(), 'lr': self.lr_params})
        self.parameters.append({'params': self.model.variational_parameters(), 'lr': self.lr_vi})

        self.mll = gpytorch.mlls.variational_elbo.VariationalELBO(self.likelihood, self.model, self.train_t_tensor.numel())
        self.optimizer = torch.optim.Adam(self.parameters)

        self.fitted = False

    def fit(self, verbose=True):
        self.model.train()
        self.likelihood.train()

        t = time.time()

        for itr in range(1, self.num_iter_fit + 1):

            self.optimizer.zero_grad()
            output = self.model(self.train_x_tensor)
            loss = - self.mll(output, self.train_t_tensor)
            loss.backward()
            self.optimizer.step()

            if verbose and (itr == 1 or itr % 100 == 0):
                duration = time.time() - t
                t = time.time()
                print('Iter %d/%d - Loss: %.3f - Time %.3f sec' % (itr, self.num_iter_fit, loss.item(), duration))

        self.fitted = True

        self.model.eval()
        self.likelihood.eval()

    def predict(self, test_x):
        test_x_tensor = torch.from_numpy(test_x).contiguous().float()

        pred_prob = self.likelihood(self.model(test_x_tensor)).mean
        pred_label = torch.sign(pred_prob - 0.5)

        return pred_prob.numpy(), pred_label.numpy()


    def eval(self, test_x, test_t):
        test_x_tensor = torch.from_numpy(test_x).contiguous().float()
        test_t_tensor = torch.from_numpy(test_t).contiguous().float()

        output = self.model(test_x_tensor)
        mll = self.mll(output, test_t_tensor)

        pred_prob = self.likelihood(output).mean
        pred_label = torch.sign(pred_prob - 0.5)
        accuracy = torch.mean((pred_label == test_t_tensor).float())

        return mll.item(), accuracy.item()