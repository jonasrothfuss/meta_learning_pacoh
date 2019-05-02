import torch
import gpytorch
import numpy as np
import time

from src.data_sim import sample_sinusoid_classification_data
from src.models import LearnedGPClassificationModel, LearnedMean, LearnedKernel

# set seed
np.random.seed(22)

""" Simulate meta-train and meta-test data """
N_TASKS_TEST = 10
N_SAMPLES_TEST_CONTEXT = 20
N_SAMPLES_TEST = 200

test_data_tuples = [sample_sinusoid_classification_data(N_SAMPLES_TEST_CONTEXT + N_SAMPLES_TEST) for _ in range(N_TASKS_TEST)]


""" Prepare Training """

# A) Set training parameters

LR_VI = 1e-2
LR_params = 1e-3
N_STEPS_TEST_TRAIN = 500 #0
WEIGHT_DECAY = 1e-3
MODEL_CLASS = "both"

print(MODEL_CLASS)

# B) Setup models

shared_params = []

if MODEL_CLASS in ["learned mean", "both"]:
    mean_function = LearnedMean(input_dim=2)
    shared_params.append({'params': mean_function.parameters(), 'lr': LR_params, 'weight_decay': WEIGHT_DECAY})
else:
    mean_function = None

if MODEL_CLASS in ["learned kern", "both"]:
    kernel_function = LearnedKernel(input_dim=2)
    shared_params.append({'params': kernel_function.parameters(), 'lr': LR_params, 'weight_decay': WEIGHT_DECAY})
    input_dim = 2
else:
    kernel_function = None
    input_dim = 2



likelihood = gpytorch.likelihoods.BernoulliLikelihood()

""" ----------------- Evaluate ------------------------ """

test_accuracies = []

for task_id, (test_x, test_t) in enumerate(test_data_tuples):

    # samples for est inference
    context_x_tensor = torch.from_numpy(test_x[:N_SAMPLES_TEST_CONTEXT]).contiguous().float()
    context_t_tensor = torch.from_numpy(test_t[:N_SAMPLES_TEST_CONTEXT]).contiguous().float()

    # corresponding samples for evaluating the test score
    test_x_tensor = torch.from_numpy(test_x[N_SAMPLES_TEST_CONTEXT:]).contiguous().float()
    test_t_tensor = torch.from_numpy(test_t[N_SAMPLES_TEST_CONTEXT:]).contiguous().float()

    # initialize the approximate GPC model
    test_model = LearnedGPClassificationModel(context_x_tensor, learned_kernel=kernel_function, learned_mean=mean_function,
                                 covar_module=None, mean_module=gpytorch.means.ConstantMean(), input_dim=input_dim)

    mll = gpytorch.mlls.variational_elbo.VariationalELBO(likelihood, test_model, context_t_tensor.numel())

    test_model.train()
    likelihood.train()

    # define the task parameters to train
    params = [
        {'params': test_model.variational_parameters()},
        {'params': test_model.hyperparameters(), 'lr': LR_params}
    ]
    # now only optimize over task params
    test_optimizer = torch.optim.Adam(params, lr=LR_VI)

    print('\n', 'Training test-model %i'%task_id)
    # train VI parameters of the test model using the context data
    for itr in range(1, N_STEPS_TEST_TRAIN + 1):

        # Zero backpropped gradients from previous iteration
        test_optimizer.zero_grad()

        output = test_model(context_x_tensor)
        loss = - mll(output, context_t_tensor)
        loss.backward()
        test_optimizer.step()

        if itr == 1 or itr % 100 == 0:
            print('Iter %d/%d - Loss: %.3f' % (itr, N_STEPS_TEST_TRAIN, loss.item()))

    # evaluate test model using the the left out test data

    test_model.eval()
    likelihood.eval()

    with torch.no_grad():
        # Get classification predictions
        observed_pred = likelihood(test_model(test_x_tensor))
        pred_labels = torch.sign(observed_pred.mean - 0.5)
        accuracy = torch.mean((pred_labels == test_t_tensor).float()).item()
        test_accuracies.append(accuracy)
        print("Test-Accuracy:", accuracy)

        # observed_pred = likelihood(model(train_x))
        # pred_labels = torch.sign(observed_pred.mean - 0.5)
        # accuracy = torch.mean((pred_labels == train_t).float()).item()
        # print("Train-Accuracy:", accuracy)

print("Overall test accuracy - mean = %.4f, std = %.4f"%(float(np.mean(test_accuracies)), float(np.std(test_accuracies))))