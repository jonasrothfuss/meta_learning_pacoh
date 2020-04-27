import pandas as pd
import os
import numpy as np
import glob

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')


""" --- Baselines --- """

CSV_PATH_BASELINES = os.path.join(DATA_DIR, 'baseline_comparison_n_tasks/baseline_comp_Apr_22_2020_18:17:31.csv') #'baseline_comparison_n_tasks/baseline_comp_Apr_20_2020_22:58:16.csv')

results_df_baselines = pd.read_csv(CSV_PATH_BASELINES)

METRIC = 'rmse'


datasets = list(set(results_df_baselines['dataset']))

if METRIC == 'rmse':
    learners = list(set(results_df_baselines['learner']))
elif METRIC == 'll':
    learners = list(set(results_df_baselines['learner']) - {'maml'})
else:
    raise NotImplementedError


result_dict = {'sin': dict([(learner, []) for learner in learners]),
                'cauchy': dict([(learner, []) for learner in learners])}


for dataset in sorted(datasets):
    for learner in sorted(learners):
        df = results_df_baselines[results_df_baselines['dataset'] == dataset]
        df = df[df['learner'] == learner]
        df = df.dropna(axis=1, how='all')

        hyperparams = list(set(df.columns) - {'Unnamed: 0', 'seed', 'll', 'rmse', 'calib_err'})

        if learner == 'maml':
            df_aggregated = df.groupby(hyperparams).aggregate(
                {'rmse': [np.mean, np.std]})
        else:
            df_aggregated = df.groupby(hyperparams).aggregate(
                {'ll': [np.mean, np.std],
                 'rmse': [np.mean, np.std],
                 'calib_err': [np.mean, np.std]})

        if METRIC == 'rmse':
            best_result_row = df_aggregated.loc[df_aggregated['rmse']['mean'].idxmin(axis=1)]['rmse']
        elif METRIC == 'll':
            best_result_row = df_aggregated.loc[df_aggregated['ll']['mean'].idxmax(axis=1)]['ll']
        else:
            raise NotImplementedError

        print(" --- dataset: %s    learner: %s    "%(dataset, learner))
        print(str(best_result_row), '\n')
        dataset_key = dataset.split("_")[0]
        n_tasks = dataset.split("_")[1]
        result_dict[dataset_key][learner].append((int(n_tasks), best_result_row['mean'], best_result_row['std']))

        #print('---- DATASET: %s , LEARNER: %s -----' % (dataset, learner))
        #print(df_aggregated.to_string(), '\n\n')


""" --- PACOH ----- """
import re

PACOH_LEARNERS = ['pacoh_mll', 'pacoh_svgd', 'pacoh_vi']
DATASETS = ['sin', 'cauchy']

for dataset in DATASETS:
    for method in PACOH_LEARNERS:
        result_dict[dataset][method] = []

PACOH_RESULT_DIR = os.path.join(DATA_DIR, 'tune-hparam-ntasks')
csv_files = glob.glob(os.path.join(PACOH_RESULT_DIR, "*.csv"))

for csv_file in csv_files:
    result_df_pacoh = pd.read_csv(csv_file)

    # determine learner from csv file name
    print(csv_file)
    learner = 'pacoh_%s'%re.compile('_vi_|_svgd_|_mll_').findall(os.path.basename(csv_file))[0][1:-1]
    assert learner in PACOH_LEARNERS

    # determine dataset & n_tasks
    dataset_str = re.compile('cauchy_[0-9]+|sin_[0-9]+').findall(os.path.basename(csv_file))[0]
    dataset = dataset_str.split('_')[0]
    assert dataset in DATASETS
    n_tasks = int(dataset_str.split('_')[1])

    hyperparams = list(set(result_df_pacoh.columns) - {'Unnamed: 0', 'random_seed', 'll', 'rmse', 'calib_err'})

    df_aggregated = result_df_pacoh.groupby(hyperparams).aggregate(
        {'ll': [np.mean, np.std],
         'rmse': [np.mean, np.std],
         'calib_err': [np.mean, np.std]})

    if METRIC == 'rmse':
        best_result_row = df_aggregated.loc[df_aggregated['rmse']['mean'].idxmin(axis=1)]['rmse']
    elif METRIC == 'll':
        best_result_row = df_aggregated.loc[df_aggregated['ll']['mean'].idxmax(axis=1)]['ll']
    else:
        raise NotImplementedError

    result_dict[dataset][learner].append((n_tasks, best_result_row['mean'], best_result_row['std']))


from matplotlib import pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

lines = []

for i, dataset in enumerate(DATASETS):
    for method, resuts_array in result_dict[dataset].items():

        print(method)
        x, y_mean, y_std = map(lambda x:np.array(x), zip(*sorted(resuts_array, key =lambda x: x[0])))

        lines.append(axes[i].plot(x, y_mean, label=method)[0])
        axes[i].fill_between(x, y_mean - y_std * (1.96 / np.sqrt(25)), y_mean + y_std * (1.96 / np.sqrt(25)), alpha=0.2)
        axes[i].set_title(dataset)
        axes[i].set_ylabel('test RMSE')
        axes[i].set_xscale('log')
        axes[i].set_yscale('log')
        axes[i].set_xlabel('number of tasks')

fig.legend()
fig.show()
fig.savefig('comparison_n_tasks.pdf')

# print table
for i, dataset in enumerate(DATASETS):
    result_df_dict = {}
    for method, resuts_array in result_dict[dataset].items():
        index = list(zip(*sorted(resuts_array, key =lambda x: x[0])))[0]
        result_df_dict[method] = list(zip(*sorted(resuts_array, key =lambda x: x[0])))[1]

    df = pd.DataFrame(result_df_dict, index=index)
    print(' --- %s --- '%dataset)
    print(df.to_string())
