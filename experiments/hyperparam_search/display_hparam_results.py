import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
HPARAM_EXP_DIR = os.path.join(DATA_DIR, 'tune-hparam')


DATASET = '' # put dataset name here to filer results

csv_files = [path for path in os.listdir(HPARAM_EXP_DIR) if '.csv' in path and 'NN' in path]

for csv_name in csv_files:
    if DATASET in csv_name:

        CSV_PATH = os.path.join(HPARAM_EXP_DIR, csv_name)

        # 1) load df
        df = pd.read_csv(CSV_PATH)

        # 2) aggregate over seeds

        hyperparams = list(set(df.columns) - {'Unnamed: 0', 'random_seed', 'll', 'rmse', 'calib_err'})
        df_aggregated = df.groupby(hyperparams).aggregate(
            {'ll': [np.mean, np.std],
             'rmse': [np.mean, np.std],
             'calib_err': [np.mean, np.std]})

        print(csv_name)
        print(df_aggregated.to_string(), '\n')