import numpy as np
import pandas as pd
import os

N_TRAIN_TASKS = 20

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
EXP_DIR = os.path.join(DATA_DIR, 'baseline_comparison')

CSV = os.path.join(EXP_DIR, 'baseline_comparison_all_Jan22_2020.csv')

df_orig = pd.read_csv(CSV)


datasets = list(set(df_orig['dataset']))
learners = list(set(df_orig['learner']))

for dataset in datasets:

    if dataset == 'physionet_2':

        for learner in learners:
            df = df_orig[df_orig['dataset'] == dataset]
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

            print('---- DATASET: %s , LEARNER: %s -----' % (dataset, learner))
            print(df_aggregated.to_string(), '\n\n')
