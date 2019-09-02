from experiments.util import collect_exp_results
import numpy as np
from matplotlib import pyplot as plt

results_df = collect_exp_results('meta-overfitting')

df_aggregated = results_df.groupby(['weight_decay', 'n_train_tasks',]).aggregate(
    {'test_ll': [np.mean, np.std],
     'test_rmse': [np.mean, np.std]})

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

weight_decay_values = sorted(set(df_aggregated.index.get_level_values('weight_decay')))

for i, metric in enumerate(['test_ll', 'test_rsme']):

    for weight_decay in weight_decay_values:
        sub_df = df_aggregated.loc[weight_decay]
        x = sub_df.index
        y_mean = sub_df[('test_ll', 'mean')]
        y_std = sub_df[('test_ll', 'std')]

        axes[i].plot(y_mean, label=weight_decay)
        axes[i].fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.1)
        axes[i].set_title(metric)
        axes[i].set_ylabel(metric)
        axes[i].set_xlabel('n_train_tasks')

plt.legend(weight_decay_values, title='weight decay', loc='lower right')
plt.show()