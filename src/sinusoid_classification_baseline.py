import numpy as np
from src.GPC_mll import GPClassificationLearned

from src.data_sim import sample_sinusoid_classification_data

# set seed
np.random.seed(22)

""" Simulate meta-train and meta-test data """
N_TASKS_TEST = 10
N_SAMPLES_TRAIN = 20
N_SAMPLES_TEST = 200

data_tuples = [sample_sinusoid_classification_data(N_SAMPLES_TRAIN + N_SAMPLES_TEST) for _ in range(N_TASKS_TEST)]


""" Prepare Training """

# B) Setup models

""" ----------------- Evaluate ------------------------ """

test_accuracies = []

for task_id, (data_x, data_t) in enumerate(data_tuples):

    train_x, train_t = data_x[:N_SAMPLES_TRAIN], data_t[:N_SAMPLES_TRAIN]
    test_x, test_t = data_x[N_SAMPLES_TRAIN:], data_t[N_SAMPLES_TRAIN:]

    gpc_model = GPClassificationLearned(train_x, train_t, mode='both', lr_vi=5e-2)
    gpc_model.fit()

    test_mll, test_accuracy = gpc_model.eval(test_x, test_t)

    # evaluate test model using the the left out test data
    print("Test-Accuracy:", test_accuracy)
    test_accuracies.append(test_accuracy)



print("Overall test accuracy - mean = %.4f, std = %.4f"%(float(np.mean(test_accuracies)), float(np.std(test_accuracies))))