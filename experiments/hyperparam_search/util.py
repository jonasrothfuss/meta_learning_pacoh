import pandas as pd
import numpy as np
import os

def select_best_configs(analysis, metric, mode='max', N=5):
    try:
        rows = analysis._retrieve_rows(metric=metric, mode=mode)
    except:
        # Get header
        missing_header = []
        for path, df in analysis.trial_dataframes.items():
            try:
                df[metric]
                header = df.columns
            except:
                missing_header.append(path)

        # Read problematic results again
        for path in missing_header:
            df = pd.read_csv(os.path.join(path, 'progress.csv'), header=None)
            df.columns = header
            analysis.trial_dataframes[path] = df

        rows = {}
        for path, df in analysis.trial_dataframes.items():
            if mode == "max":
                idx = df[metric].idxmax()
            elif mode == "min":
                idx = df[metric].idxmin()
            else:
                idx = -1
            if np.isnan(idx):
                print('NaN value in experiment: ', path)
                continue
            rows[path] = df.iloc[idx].to_dict()

    all_configs = analysis.get_all_configs()
    reverse = mode == 'max'
    best_paths = sorted(rows, key=lambda k: rows[k][metric], reverse=reverse)[:N]
    best_configs = [all_configs[path] for path in best_paths]
    return best_configs
