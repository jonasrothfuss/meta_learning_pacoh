import numpy as np
import os

from matplotlib import pyplot as plt
from experiments.util import collect_exp_results



DIR = os.path.dirname(os.path.abspath(__file__))

lines = []
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

""" ------- sinusoid ------- """
results_df = collect_exp_results('meta-overfitting-sin')
n_train_samples = 5
results_df = results_df[results_df['n_train_samples'] == n_train_samples]

df_aggregated = results_df.groupby(['n_train_tasks', 'weight_decay']).aggregate(
    {'test_ll': [np.mean, np.std],
     'test_rmse': [np.mean, np.std],
     'calib_err': [np.mean, np.std]})


n_train_tasks_list = sorted(set(df_aggregated.index.get_level_values('n_train_tasks')))

metric ='test_rmse'


for n_train_tasks in n_train_tasks_list:
    sub_df = df_aggregated.loc[n_train_tasks]
    x = sub_df.index
    y_mean = sub_df[(metric, 'mean')]
    y_std = sub_df[(metric, 'std')]

    lines.append(axes[0].plot(y_mean, label=str(n_train_tasks))[0])
    axes[0].fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.2)
    axes[0].set_title('Sinusoids')
    axes[0].set_ylabel('test RMSE')
    axes[0].set_xscale('log')
    axes[0].set_xlabel('weight decay')

axes[0].set_ylim((0.28, 0.78))

""" ----- Cauchy Dataset ------- """


results_df = collect_exp_results('meta-overfitting-cauchy')
n_train_samples = 20
results_df = results_df[results_df['n_train_samples'] == n_train_samples]

df_aggregated = results_df.groupby(['n_train_tasks', 'weight_decay']).aggregate(
    {'test_ll': [np.mean, np.std],
     'test_rmse': [np.mean, np.std],
     'calib_err': [np.mean, np.std]})


n_train_tasks_list = sorted(set(df_aggregated.index.get_level_values('n_train_tasks')))

metric ='test_rmse'

for n_train_tasks in n_train_tasks_list:
    sub_df = df_aggregated.loc[n_train_tasks]
    x = sub_df.index
    y_mean = sub_df[(metric, 'mean')]
    y_std = sub_df[(metric, 'std')]

    axes[1].plot(y_mean, label=str(n_train_tasks))
    axes[1].fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.2)
    axes[1].set_title('Mixtures of Cauchy')
    axes[1].set_ylabel('test RMSE')
    axes[1].set_xscale('log')
    axes[1].set_xlabel('weight decay')




""" ----- Swissfel Dataset ------- """


results_df = collect_exp_results('meta-overfitting-swissfel')

df_aggregated = results_df.groupby(['weight_decay']).aggregate(
    {'test_ll': [np.mean, np.std],
     'test_rmse': [np.mean, np.std],
     'calib_err': [np.mean, np.std]})


metric ='test_rmse'

sub_df = df_aggregated
x = sub_df.index
y_mean = sub_df[(metric, 'mean')]
y_std = sub_df[(metric, 'std')]

lines.append(axes[2].plot(y_mean, c='green')[0])
axes[2].fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.2, color='green')
axes[2].set_title('SwissFEL')
axes[2].set_ylabel('test RMSE')
axes[2].set_xscale('log')
axes[2].set_xlabel('weight decay')

axes[2].set_ylim((0.39, 0.98))

lines = [lines[2], lines[0], lines[1]]

#plt.suptitle('Cauchy Meta-Dataset')
plt.legend(handles=lines, labels=('5', '10', '20'), title='number train tasks', loc='upper right')
plt.tight_layout(rect=(0, 0, 1, 0.98))
plt.show()
fig.savefig(os.path.join(DIR, 'meta-overfitting.png'))
fig.savefig(os.path.join(DIR, 'meta-overfitting.pdf'))
