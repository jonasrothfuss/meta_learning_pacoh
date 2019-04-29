import torch
import gpytorch
import numpy as np

from src.data_sim import sample_sinusoid_data
from src.models import LearnedGPRegressionModel, LearnedMean, LearnedKernel

# set seed
np.random.seed(22)

""" Simulate meta-train and meta-test data """
N_TASKS_TRAIN = 50
N_SAMPLES_TRAIN = 100

N_TASKS_TEST = 100
N_SAMPLES_TEST_CONTEXT = 100
N_SAMPLES_TEST = 100

train_data_tuples = [sample_sinusoid_data(N_SAMPLES_TRAIN) for _ in range(N_TASKS_TRAIN)]
test_data_tuples = [sample_sinusoid_data(N_SAMPLES_TEST_CONTEXT + N_SAMPLES_TEST) for _ in range(N_TASKS_TEST)]


""" Training """

N_STEPS = 50000 #TODO
MODEL_CLASS = "both"

print("Training ...")
print(MODEL_CLASS)

likelihood = gpytorch.likelihoods.GaussianLikelihood()
loss_array = []

if MODEL_CLASS in ["learned mean", "both"]:
    mean_function = LearnedMean(input_dim=1)
else:
    mean_function = None

if MODEL_CLASS in ["learned kern", "both"]:
    kernel_function = LearnedKernel(input_dim=1)
    input_dim = 2
else:
    kernel_function = None
    input_dim = 1


def compute_test_scores():
    test_mses = []
    test_likelihoods = []

    for j in range(len(test_data_tuples)):
        x_test, y_test = test_data_tuples[j]

        # samples for test inference
        x_context_tensor = torch.from_numpy(x_test[:N_SAMPLES_TEST_CONTEXT]).contiguous().float()
        y_context_tensor = torch.from_numpy(y_test[:N_SAMPLES_TEST_CONTEXT]).contiguous().squeeze().float()

        # corresponding samples for evaluating the test score
        x_test_tensor = torch.from_numpy(x_test[N_SAMPLES_TEST_CONTEXT:]).contiguous().float()
        y_test_tensor = torch.from_numpy(y_test[N_SAMPLES_TEST_CONTEXT:]).contiguous().squeeze().float()

        model_test = LearnedGPRegressionModel(x_context_tensor, y_context_tensor, likelihood,
                                              learned_mean=mean_function, learned_kernel=kernel_function,
                                              input_dim=input_dim)

        mll_test = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model_test)
        mse_test = torch.nn.MSELoss(reduction='mean')

        model_test.eval()
        likelihood.eval()

        with torch.no_grad(), gpytorch.settings.use_toeplitz(False):
            preds = model_test(x_test_tensor)
            means_pred = preds.mean
            #test_likelihoods.append(mll_test(preds, y_test_tensor).item())
            test_mses.append(mse_test(means_pred, y_test_tensor).item())

    test_mse, test_mse_std = float(np.mean(test_mses)), float(np.std(test_mses))
    #test_likelihood, test_likelihood_std = float(np.mean(test_likelihoods)), float(np.std(test_likelihoods))

    test_likelihood, test_likelihood_std = 0.0, 0.0
    return test_mse, test_likelihood, test_mse_std, test_likelihood_std


for i in range(N_STEPS):

    # select random task
    # TODO: support mini-batches on the task level
    rand_idx = np.random.choice(len(train_data_tuples))
    x_train, y_train = train_data_tuples[rand_idx]

    x_train_tensor = torch.from_numpy(x_train).contiguous().float()
    y_train_tensor = torch.from_numpy(y_train).contiguous().squeeze().float()

    model = LearnedGPRegressionModel(x_train_tensor, y_train_tensor, likelihood,
                                 learned_mean=mean_function, learned_kernel=kernel_function,
                                 input_dim=input_dim)
    model.train()
    likelihood.train()

    # define parameters to train
    params = [
        # {'params': model.covar_module.parameters()},
        {'params': model.likelihood.parameters()}]

    if MODEL_CLASS in ["learned mean", "both"]:
        params.append({'params': model.learned_mean.parameters(), 'weight_decay': 1e-3})
    if MODEL_CLASS in ["learned kern", "both"]:
        params.append({'params': model.learned_kernel.parameters(), 'weight_decay': 1e-3})

    optimizer = torch.optim.Adam(params)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    optimizer.zero_grad()
    # Get output from model
    output = model(x_train_tensor)

    loss = -mll(output, y_train_tensor)
    loss_array.append(loss.item())

    loss.backward()
    optimizer.step()

    if i % 1000 == 0:
        test_mse, test_likelihood, _, _ = compute_test_scores()
        print("Step {}|  train-loss = {:.4f}, test-mse = {:.4f}".format(i, np.mean(loss_array), test_mse))
        loss_array = []


test_mse, test_likelihood, test_mse_std, test_likelihood_std = compute_test_scores()

print("test MSE: %.4f  test MSE std: %.4f"%(test_mse, test_mse_std))
print("test likelihood: %.4f  test likelihood std: %.4f"%(test_likelihood, test_likelihood_std))

