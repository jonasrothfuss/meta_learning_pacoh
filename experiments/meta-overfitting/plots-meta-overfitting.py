import numpy as np
import os

from matplotlib import pyplot as plt
from experiments.util import collect_exp_results

DIR = os.path.dirname(os.path.abspath(__file__))

""" ------- nonstationary sinusoid ------- """

for num_layer in [1, 3]:
    results_df = collect_exp_results('meta-overfitting-sin')
    results_df = results_df[results_df['num_layers'] == num_layer]

    fig, axes = plt.subplots(3, 2, figsize=(10, 15))

    for col, n_train_samples in enumerate([5, 10, 20]):

        df_aggregated = results_df[results_df['n_train_samples'] == n_train_samples]\
            .groupby(['weight_decay', 'n_train_tasks',]).aggregate(
            {'test_ll': [np.mean, np.std],
             'test_rmse': [np.mean, np.std]})


        weight_decay_values = sorted(set(df_aggregated.index.get_level_values('weight_decay')))


        for i, metric in enumerate(['test_ll', 'test_rmse']):

            for weight_decay in weight_decay_values:
                sub_df = df_aggregated.loc[weight_decay]
                x = sub_df.index
                y_mean = sub_df[(metric, 'mean')]
                y_std = sub_df[(metric, 'std')]

                axes[col][i].plot(y_mean, label=weight_decay)
                axes[col][i].fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.1)
                axes[col][i].set_title('n_train_samples: %i'%n_train_samples)
                axes[col][i].set_ylabel(metric)
                axes[col][i].set_xlabel('n_train_tasks')

    plt.legend(weight_decay_values, title='weight decay', loc='upper right')
    plt.suptitle('Non-stationary Sinusoids')
    plt.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(os.path.join(DIR, 'meta-overfit-sin-n_tasks-%i-layers.png'%num_layer))
    fig.savefig(os.path.join(DIR, 'meta-overfit-sin-n_tasks-%i-layers.pdf'%num_layer))

    fig, axes = plt.subplots(3, 2, figsize=(10, 15))

    for col, n_train_samples in enumerate([5, 10, 20]):

        df_aggregated = results_df[results_df['n_train_samples'] == n_train_samples]\
            .groupby(['n_train_tasks', 'weight_decay']).aggregate(
            {'test_ll': [np.mean, np.std],
             'test_rmse': [np.mean, np.std]})


        n_train_tasks_list = sorted(set(df_aggregated.index.get_level_values('n_train_tasks')))


        for i, metric in enumerate(['test_ll', 'test_rmse']):

            for n_train_tasks in n_train_tasks_list:
                sub_df = df_aggregated.loc[n_train_tasks]
                x = sub_df.index
                y_mean = sub_df[(metric, 'mean')]
                y_std = sub_df[(metric, 'std')]

                axes[col][i].plot(y_mean, label=str(n_train_tasks))
                axes[col][i].fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.1)
                axes[col][i].set_title('n_train_samples: %i'%n_train_samples)
                axes[col][i].set_ylabel(metric)
                axes[col][i].set_xscale('log')
                axes[col][i].set_xlabel('weight_decay')

    plt.legend(n_train_tasks_list, title='n_train_tasks', loc='upper right')
    plt.suptitle('Non-stationary Sinusoids')
    plt.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(os.path.join(DIR, 'meta-overfit-sin-weight_dec-%i-layers.png'%num_layer))
    fig.savefig(os.path.join(DIR, 'meta-overfit-sin-weight_dec-%i-layers.pdf'%num_layer))


#
# """ ------- MNIST ------- """
#
# results_df = collect_exp_results('meta-overfitting-mnist')
#
# fig, axes = plt.subplots(1, 2, figsize=(10, 5))
#
#
# for col, n_train_samples in enumerate([28*28]):
#
#     df_aggregated = results_df[results_df['n_train_samples'] == n_train_samples]\
#         .groupby(['weight_decay', 'n_train_tasks',]).aggregate(
#         {'test_ll': [np.mean, np.std],
#          'test_rmse': [np.mean, np.std]})
#
#
#     weight_decay_values = sorted(set(df_aggregated.index.get_level_values('weight_decay')))
#
#
#     for i, metric in enumerate(['test_ll', 'test_rmse']):
#
#         for weight_decay in weight_decay_values:
#             sub_df = df_aggregated.loc[weight_decay]
#             x = sub_df.index
#             y_mean = sub_df[(metric, 'mean')]
#             y_std = sub_df[(metric, 'std')]
#
#             axes[i].plot(y_mean, label=weight_decay)
#             axes[i].fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.1)
#             axes[i].set_title('n_train_samples: %i'%n_train_samples)
#             axes[i].set_ylabel(metric)
#             axes[i].set_xlabel('n_train_tasks')
#
# plt.legend(weight_decay_values, title='weight decay', loc='upper right')
# plt.suptitle('MNIST Regression')
# plt.show()
#
#
# fig, axes = plt.subplots(1, 2, figsize=(10, 5))
#
# for col, n_train_samples in enumerate([400]):
#
#     df_aggregated = results_df[results_df['n_train_samples'] == n_train_samples]\
#         .groupby(['n_train_tasks', 'weight_decay']).aggregate(
#         {'test_ll': [np.mean, np.std],
#          'test_rmse': [np.mean, np.std]})
#
#
#     n_train_tasks_list = sorted(set(df_aggregated.index.get_level_values('n_train_tasks')))
#
#
#     for i, metric in enumerate(['test_ll', 'test_rmse']):
#
#         for n_train_tasks in n_train_tasks_list:
#             sub_df = df_aggregated.loc[n_train_tasks]
#             x = sub_df.index
#             y_mean = sub_df[(metric, 'mean')]
#             y_std = sub_df[(metric, 'std')]
#
#             axes[i].plot(y_mean, label=str(n_train_tasks))
#             axes[i].fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.1)
#             axes[i].set_title('n_train_samples: %i'%n_train_samples)
#             axes[i].set_ylabel(metric)
#             axes[i].set_xscale('log')
#             axes[i].set_xlabel('weight_decay')
#
# plt.legend(n_train_tasks_list, title='n_train_tasks', loc='upper right')
# plt.suptitle('MNIST Regression')
# plt.show()