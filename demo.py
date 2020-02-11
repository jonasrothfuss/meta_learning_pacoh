import os
import sys
import numpy as np

# add BADE_DIR to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


""" Generate meta-training and meta-testing data """

from experiments.data_sim import SinusoidDataset

random_state = np.random.RandomState(26)
task_environment = SinusoidDataset(random_state=random_state)

meta_train_data = task_environment.generate_meta_train_data(n_tasks=20, n_samples=5)
meta_test_data = task_environment.generate_meta_test_data(n_tasks=20, n_samples_context=5, n_samples_test=50)


""" Meta-Training w/ PACOH-MAP """

from meta_learn import GPRegressionMetaLearned

random_gp = GPRegressionMetaLearned(meta_train_data, weight_decay=0.2, num_iter_fit=12000, random_seed=30)
random_gp.meta_fit(meta_test_data, log_period=1000)


""" Meta-Testing w/ PACOH-MAP"""

print('\n')
ll, rmse, calib_err = random_gp.eval_datasets(meta_test_data)
print('Test log-likelihood:', ll)
print('Test RMSE:', rmse)
print('Test calibration error:', calib_err)


try:
    from matplotlib import pyplot as plt
    x_plot = np.linspace(-5, 5, num=150)
    x_context, y_context, x_test, y_test = meta_test_data[0]
    pred_mean, pred_std = random_gp.predict(x_context, y_context, x_plot)
    ucb, lcb = random_gp.confidence_intervals(x_context, y_context, x_plot, confidence=0.9)

    plt.scatter(x_test, y_test, label='target_testing points' )
    plt.scatter(x_context, y_context, label='target training points')

    plt.plot(x_plot, pred_mean)
    plt.fill_between(x_plot, lcb, ucb, alpha=0.2, label='90 % confidence interval')
    plt.legend()
    plt.title("meta-testing prediction on new target task")
    plt.show()
except:
    print('\n Could not plot results since matplotlib package is not installed. ')

