import torch
import gpytorch
import numpy as np

from src.data_sim import sample_sinusoid_classification_data
from src.models import LearnedGPClassificationModel, LearnedMean, LearnedKernel

# set seed
np.random.seed(22)

""" Simulate meta-train and meta-test data """
N_TASKS_TRAIN = 1 #TODO set to 50
N_SAMPLES_TRAIN = 500 # TOOD set to 100

N_TASKS_TEST = 100
N_SAMPLES_TEST_CONTEXT = 100
N_SAMPLES_TEST = 100

train_data_tuples = [sample_sinusoid_classification_data(N_SAMPLES_TRAIN) for _ in range(N_TASKS_TRAIN)]
test_data_tuples = [sample_sinusoid_classification_data(N_SAMPLES_TEST_CONTEXT + N_SAMPLES_TEST) for _ in range(N_TASKS_TEST)]


""" Training """

LR = 1e-2
N_STEPS = 5000
MODEL_CLASS = "vanilla"

print("Training ...")
print(MODEL_CLASS)

if MODEL_CLASS in ["learned mean", "both"]:
    mean_function = LearnedMean(input_dim=2)
else:
    mean_function = None

if MODEL_CLASS in ["learned kern", "both"]:
    kernel_function = LearnedKernel(input_dim=2)
    input_dim = 2
else:
    kernel_function = None
    input_dim = 2


train_x, train_t = train_data_tuples[0]
train_x = torch.from_numpy(train_x).contiguous().float()
train_t = torch.from_numpy(train_t).contiguous().squeeze().float()

model = LearnedGPClassificationModel(train_x, learned_kernel=kernel_function, learned_mean=mean_function, input_dim=input_dim)
likelihood = gpytorch.likelihoods.BernoulliLikelihood()

model.train()
likelihood.train()

# define the parameters to train
params = [
    {'params': model.variational_parameters()},
    {'params': model.hyperparameters(), 'lr': LR * 0.1},
    {'params': likelihood.parameters()}

]
if MODEL_CLASS in ["learned mean", "both"]:
    params.append({'params': model.learned_mean.parameters(), 'weight_decay': 1e-3, 'lr': LR * 0.1})
if MODEL_CLASS in ["learned kern", "both"]:
    params.append({'params': model.learned_kernel.parameters(), 'weight_decay': 1e-3, 'lr': LR * 0.1})

optimizer = torch.optim.Adam(params, lr=LR)
mll = gpytorch.mlls.variational_elbo.VariationalELBO(likelihood, model, train_t.numel())

for i in range(N_STEPS):
    # Zero backpropped gradients from previous iteration
    optimizer.zero_grad()
    # Get predictive output
    output = model(train_x)
    # Calc loss and backprop gradients
    loss = -mll(output, train_t)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, N_STEPS, loss.item()))
    optimizer.step()