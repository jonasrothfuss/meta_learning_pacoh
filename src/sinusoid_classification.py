import torch
import gpytorch
import numpy as np
import time

from src.data_sim import sample_sinusoid_classification_data
from src.models import LearnedGPClassificationModel, LearnedMean, LearnedKernel

# set seed
np.random.seed(22)

""" Simulate meta-train and meta-test data """
N_TASKS_TRAIN = 10 #TODO set to 50
N_SAMPLES_TRAIN = 50 # TOOD set to 100

N_TASKS_TEST = 10
N_SAMPLES_TEST_CONTEXT = 50
N_SAMPLES_TEST = 50

train_data_tuples = [sample_sinusoid_classification_data(N_SAMPLES_TRAIN) for _ in range(N_TASKS_TRAIN)]
test_data_tuples = [sample_sinusoid_classification_data(N_SAMPLES_TEST_CONTEXT + N_SAMPLES_TEST) for _ in range(N_TASKS_TEST)]


""" Prepare Training """

# A) Set training parameters

LR_VI = 1e-2
LR_params = 1e-3
N_STEPS_META_TRAIN = 2000 #0
N_STEPS_TEST_TRAIN = 200 #0
WEIGHT_DECAY = 1e-3
MODEL_CLASS = "vanilla"
SHARE_SHALLOW_PRIOR_PARAMS = True # whether to share parameters of the SE kernel

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

if SHARE_SHALLOW_PRIOR_PARAMS:
    covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=input_dim))
    shared_params.append({'params': covar_module.parameters(), 'lr': LR_params})
else:
    covar_module = None

task_dicts = []

for task_id, (train_x, train_t) in enumerate(train_data_tuples):

    # Convert the data into pytorch tensors
    train_x_tensor = torch.from_numpy(train_x).contiguous().float()
    train_t_tensor = torch.from_numpy(train_t).contiguous().float()

    # initialize the approximate GPC model
    gp_model = LearnedGPClassificationModel(train_x_tensor, learned_kernel=kernel_function, learned_mean=mean_function,
                                 covar_module=covar_module, input_dim=input_dim)

    # define the task parameters to train
    task_params = [{'params': list(gp_model.variational_parameters())}] # VI parameters of q(u), i.e. the mean and cov matrix

    if not SHARE_SHALLOW_PRIOR_PARAMS:
        task_params.append({'params': gp_model.covar_module.parameters(), 'lr': LR_params})

    task_dict = {
        'train_x': train_x_tensor,
        'train_t': train_t_tensor,
        'model': gp_model,
        'params': task_params,
        'mll_fn': gpytorch.mlls.variational_elbo.VariationalELBO(likelihood, gp_model, train_t_tensor.numel())
    }
    task_dicts.append(task_dict)

# merge all params into one array
task_params_all = [param for task_dict in task_dicts for param in task_dict['params']]

params_all = task_params_all + shared_params

optimizer = torch.optim.Adam(params_all, lr=LR_VI)

print("Training ...")

t = time.time()
for itr in range(1, N_STEPS_META_TRAIN+1):

    # Zero backpropped gradients from previous iteration
    optimizer.zero_grad()

    losses = torch.zeros(len(task_dicts))
    likelihood.train()
    for task_idx, task_dict in enumerate(task_dicts):
        task_dict['model'].train()
        # Get predictive output
        output = task_dict['model'](task_dict['train_x'])
        losses[task_idx] = - task_dict['mll_fn'](output, task_dict['train_t'])

    # Calc loss and backprop gradients
    loss = torch.mean(losses)
    loss.backward()
    optimizer.step()


    if itr == 1 or itr % 100 == 0:
        duration = time.time() - t
        t = time.time()
        print('Iter %d/%d - Loss: %.3f, Time: %.3f sec' % (itr, N_STEPS_META_TRAIN, loss.item(), duration))


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
                                 covar_module=covar_module, input_dim=input_dim)

    mll = gpytorch.mlls.variational_elbo.VariationalELBO(likelihood, test_model, context_t_tensor.numel())

    test_model.train()
    likelihood.train()

    # define the task parameters to train
    task_params = [{'params': list(test_model.variational_parameters())}] # VI parameters of q(u), i.e. the mean and cov matrix

    if not SHARE_SHALLOW_PRIOR_PARAMS:
        task_params.append({'params': gp_model.covar_module.parameters(), 'lr': LR_params})

    # now only optimize over task params
    test_optimizer = torch.optim.Adam(task_params, lr=LR_VI)

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