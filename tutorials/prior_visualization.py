import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import numpy as np

torch.set_default_tensor_type(torch.DoubleTensor)
torch.manual_seed(7)
np.random.seed(22)

""" The data """

# Training data is 11 points in [0,1] inclusive regularly spaced
f = lambda x: 5 + 7*x + torch.sin(x * (2 * math.pi))

train_x = torch.linspace(0, 1, 100)
train_y = f(train_x) + torch.randn(train_x.size()) * 0.1

test_x = torch.linspace(0, 1, 100)
test_y = f(test_x) + torch.randn(test_x.size()) * 0.1


# plt.plot(train_x.numpy(), train_y.numpy())
# plt.show()

""" The GP model """

class ExactGpModel(gpytorch.models.ExactGP):

  def __init__(self, train_x, train_y, likelihood):
    super(ExactGpModel, self).__init__(train_x, train_y, likelihood)
    self.mean_module = gpytorch.means.ConstantMean()
    self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())


  def forward(self, x):
    mean_x = self.mean_module(x)
    covar_x = self.covar_module(x)
    return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGpModel(train_x, train_y, likelihood)



#training
model.train()
likelihood.train()

optimizer = torch.optim.Adam([
  {'params': model.parameters()}
], lr=0.1)

# "Loss" for GPs - the mrginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

train_iter = 200
for i in range(train_iter):
  optimizer.zero_grad()
  output = model(train_x)

  loss = -mll(output, train_y)
  loss.backward()

  print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        i + 1, train_iter, loss.item(),
        model.covar_module.base_kernel.lengthscale.item(),
        model.likelihood.noise.item()
    ))

  optimizer.step()

# prior

with gpytorch.settings.lazily_evaluate_kernels(False), torch.no_grad():
    K = model.covar_module(test_x).numpy()
    n = K.shape[0]
    L = np.linalg.cholesky(K + 1e-12*np.eye(n))
    f_prior = np.dot(L, np.random.normal(size=(n,5)))

fig, axes = plt.subplots(1,2)
axes[0].plot(test_x.numpy(), f_prior)
axes[0].set_title('prior')



# posterior
model.eval()
likelihood.eval()

with torch.no_grad():
    posterior_dist = likelihood(model(test_x))
    K = posterior_dist.covariance_matrix.numpy()
    n = K.shape[0]
    L = np.linalg.cholesky(K + 1e-12*np.eye(n))
    mean = posterior_dist.mean.numpy()
    f_post = np.expand_dims(mean,-1) + np.dot(L, np.random.normal(size=(n, 5)))


axes[1].plot(test_x.numpy(), mean)
axes[1].plot(test_x.numpy(), f(test_x).numpy(), linestyle='dashed')
axes[1].set_title('posterior')
plt.show()

