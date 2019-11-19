from experiments.util import collect_exp_results
import numpy as np
from matplotlib import pyplot as plt

import os

DIR = os.path.dirname(os.path.abspath(__file__))

""" ------- GP-funcs ------- """

results_df = collect_exp_results('meta-vi-gp-funcs')

results_df = results_df[results_df['num_layers'] == 5]
results_df = results_df[results_df['weight_prior_scale'] == 1.0]


fig, axes = plt.subplots(3, 2, figsize=(8, 10), constrained_layout=False)

for col, n_train_samples in enumerate([5, 10, 20]):

    df_aggregated = results_df[results_df['n_train_samples'] == n_train_samples]\
        .groupby(['prior_factor', 'n_train_tasks']).aggregate(
        {'test_ll_bayes': [np.mean, np.std],
         'test_rmse_bayes': [np.mean, np.std]})


    prior_factor_values = sorted(set(df_aggregated.index.get_level_values('prior_factor')))


    for i, metric in enumerate(['test_ll_bayes', 'test_rmse_bayes']):

        for prior_factor in prior_factor_values:
            sub_df = df_aggregated.loc[prior_factor]
            x = sub_df.index
            y_mean = sub_df[(metric, 'mean')]
            y_std = sub_df[(metric, 'std')]

            axes[col][i].plot(y_mean, label=prior_factor)
            axes[col][i].fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.1)
            axes[col][i].set_title('n_train_samples: %i'%n_train_samples)
            axes[col][i].set_ylabel(metric)
            axes[col][i].set_xlabel('n_train_tasks')

plt.legend(prior_factor_values, title='prior_factor', loc='upper right')
fig.suptitle('GP functions', fontsize='x-large')

plt.tight_layout(rect=(0, 0, 1., 0.97))
fig.savefig(os.path.join(DIR, 'meta-vi-gp-funcs.png'))
fig.savefig(os.path.join(DIR, 'meta-vi-gp-funcs.pdf'))
fig.show()



""" ------- nonstationary sinusoid ------- """


results_df = collect_exp_results('meta-vi-sin')

results_df = results_df[results_df['num_layers'] == 2]
results_df = results_df[results_df['weight_prior_scale'] == 0.1]


fig, axes = plt.subplots(3, 2, figsize=(8, 10), constrained_layout=False)

for col, n_train_samples in enumerate([5, 10, 20]):

    df_aggregated = results_df[results_df['n_train_samples'] == n_train_samples]\
        .groupby(['prior_factor', 'n_train_tasks']).aggregate(
        {'test_ll_bayes': [np.mean, np.std],
         'test_rmse_bayes': [np.mean, np.std]})


    prior_factor_values = sorted(set(df_aggregated.index.get_level_values('prior_factor')))


    for i, metric in enumerate(['test_ll_bayes', 'test_rmse_bayes']):

        for prior_factor in prior_factor_values:
            sub_df = df_aggregated.loc[prior_factor]
            x = sub_df.index
            y_mean = sub_df[(metric, 'mean')]
            y_std = sub_df[(metric, 'std')]

            axes[col][i].plot(y_mean, label=prior_factor)
            axes[col][i].fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.1)
            axes[col][i].set_title('n_train_samples: %i'%n_train_samples)
            axes[col][i].set_ylabel(metric)
            axes[col][i].set_xlabel('n_train_tasks')

plt.legend(prior_factor_values, title='prior_factor', loc='upper right')
fig.suptitle('Nonstationary Sinusoids', fontsize='x-large')

plt.tight_layout(rect=(0, 0, 1., 0.97))
fig.savefig(os.path.join(DIR, 'meta-vi-sin.png'))
fig.savefig(os.path.join(DIR, 'meta-vi-sin.pdf'))
fig.show()

