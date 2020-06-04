import numpy as np
import pandas as pd
import os

from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from experiments.util import collect_exp_results

params = {'legend.fontsize': 9.5,}
plt.rcParams.update(params)

DIR = os.path.dirname(os.path.abspath(__file__))

lines = []
fig, axes = plt.subplots(1, 2, figsize=(9, 3))

ALGO = 'pacoh-map'

for ALGO in ['pacoh-map', 'mll']:

    """ ------- sinusoid ------- """
    results_df = collect_exp_results('meta-overfitting-v2-%s-sin'%ALGO)
    n_context_samples = 5
    results_df = results_df[results_df['n_context_samples'] == n_context_samples]
    results_df = results_df[results_df['n_train_tasks'] >= 4]

    n_train_tasks_list = sorted(list(set(results_df['n_train_tasks'])))
    if 'map' in ALGO:
        best_row_per_n_tasks = []
        n_tasks_list = sorted(list(set(results_df['n_train_tasks'])))
        for n_tasks in n_tasks_list:
            df_aggregated = results_df.groupby(['n_train_tasks', 'weight_decay']).aggregate(
                {'test_rmse_meta_train': [np.mean, np.std],
                 'test_rmse_meta_test': [np.mean, np.std],
                 }
            )
            df_aggregated_sub = df_aggregated.loc[n_tasks]
            best_result_row = df_aggregated_sub.loc[df_aggregated_sub['test_rmse_meta_test']['mean'].idxmin(axis=1)]
            best_row_per_n_tasks.append(best_result_row)
        df_aggregated =pd.concat(best_row_per_n_tasks, axis=1, keys=n_tasks_list).T
    else:
        df_aggregated = results_df.groupby(['n_train_tasks']).aggregate(
            {'test_rmse_meta_train': [np.mean, np.std],
             'test_rmse_meta_test': [np.mean, np.std],
             }
        )

    print(""" ----- Sinusoid %s (n_context_samples=%i) ------"""%(ALGO, n_context_samples))
    print(df_aggregated.to_string(), '\n')


    metrics = ['test_rmse_meta_train', 'test_rmse_meta_test']


    for metric in metrics:
        x = df_aggregated.index
        y_mean = df_aggregated[(metric, 'mean')]
        y_std = df_aggregated[(metric, 'std')]

        linestyle = '--' if ALGO=='mll' else '-'
        lines.append(axes[0].plot(y_mean, label=str(metric), linestyle=linestyle)[0])
        axes[0].fill_between(x, y_mean - y_std * (1.96/np.sqrt(25)), y_mean + y_std*(1.96/np.sqrt(25)), alpha=0.2)
        axes[0].set_title('Sinusoid')
        axes[0].set_ylabel('test RMSE')
        axes[0].set_xscale('log')
        #axes[0].set_yscale('log')
        axes[0].set_xlabel('number of tasks')

    """ ------- cauchy  ------- """
    results_df = collect_exp_results('meta-overfitting-v2-%s-cauchy'%ALGO)
    n_context_samples = 20
    results_df = results_df[results_df['n_context_samples'] == n_context_samples]
    results_df = results_df[results_df['n_train_tasks'] >= 4]


    if 'map' in ALGO:
        best_row_per_n_tasks = []
        n_tasks_list = sorted(list(set(results_df['n_train_tasks'])))
        for n_tasks in n_tasks_list:
            df_aggregated = results_df.groupby(['n_train_tasks', 'weight_decay']).aggregate(
                {'test_rmse_meta_train': [np.mean, np.std],
                 'test_rmse_meta_test': [np.mean, np.std],
                 }
            )
            df_aggregated_sub = df_aggregated.loc[n_tasks]
            best_result_row = df_aggregated_sub.loc[df_aggregated_sub['test_rmse_meta_test']['mean'].idxmin(axis=1)]
            best_row_per_n_tasks.append(best_result_row)
        df_aggregated =pd.concat(best_row_per_n_tasks, axis=1, keys=n_tasks_list).T
    else:
        df_aggregated = results_df.groupby(['n_train_tasks']).aggregate(
            {'test_rmse_meta_train': [np.mean, np.std],
             'test_rmse_meta_test': [np.mean, np.std],
             }
        )

    print(""" ----- Cauchy %s (n_context_samples=%i) ------"""%(ALGO, n_context_samples))
    print(df_aggregated.to_string(), '\n')

    metrics = ['test_rmse_meta_train', 'test_rmse_meta_test']


    for metric in metrics:
        x = df_aggregated.index
        y_mean = df_aggregated[(metric, 'mean')]
        y_std = df_aggregated[(metric, 'std')]

        linestyle = '--' if ALGO=='mll' else '-'
        lines.append(axes[1].plot(y_mean, label=str(metric), linestyle=linestyle)[0])
        axes[1].fill_between(x, y_mean - y_std * (1.96/np.sqrt(25)), y_mean + y_std * (1.96/np.sqrt(25)), alpha=0.2)
        axes[1].set_title('Cauchy')
        axes[1].set_ylabel('test RMSE')
        axes[1].set_xscale('log')
        #axes[1].set_yscale('log')
        axes[1].set_xlabel('number of tasks')


# axes[0].set_ylim((0.28, 0.9))
# axes[1].set_ylim((0.0, 0.4))
for i in [0, 1]:
    axes[i].set_xticks(ticks=[5, 10, 20, 50, 100, 200, 500])
    for axis in [axes[i].xaxis, axes[i].yaxis]:
        axis.set_major_formatter(ScalarFormatter())

#axes[0].set_yticks(ticks=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

# axes[1].legend(axes[0].lines, ('pacoh-map (meta-train tasks)','pacoh-map (meta-test tasks)',
#                                'mll (meta-train tasks)','mll (meta-test tasks)') )
fig.suptitle('')
lgd = axes[0].legend(axes[0].lines, ('PACOH-MAP (meta-train tasks)','PACOH-MAP (meta-test tasks)',
                               'MLL (meta-train tasks)','MLL (meta-test tasks)'))
#fig.tight_layout(rect=[0.5, 0.2, 0.5, 1])


fig.show()
fig.savefig('meta_overfitting_v2_map_vs_mll.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')


