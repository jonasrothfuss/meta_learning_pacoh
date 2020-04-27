import numpy as np
import argparse
import os
import ray
import torch
import itertools
import pandas
import sys
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
EXP_DIR = os.path.join(DATA_DIR, 'baseline_comparison_n_tasks')
if not os.path.isdir(EXP_DIR): os.makedirs(EXP_DIR)

# Configuration w.r.t. data


DATASETS = ['%s_%i'%(dataset, n_tasks) for n_tasks in [5, 10, 20, 40, 80, 160, 320] for dataset in ['cauchy', 'sin']]


DATA_SEED = 28
MODEL_SEEDS = [22, 23, 24, 25, 26]

LAYER_SIZES = [32, 32, 32, 32]


@ray.remote
def fit_eval_meta_algo(param_dict):
    sys.path.append(BASE_DIR)

    from meta_learn.GPR_meta_svgd import GPRegressionMetaLearnedSVGD
    from meta_learn.GPR_meta_vi import GPRegressionMetaLearnedVI
    from meta_learn.GPR_meta_mll import GPRegressionMetaLearned
    from meta_learn.NPR_meta import NPRegressionMetaLearned
    from meta_learn.MAML import MAMLRegression

    torch.set_num_threads(1)

    meta_learner = param_dict.pop("meta_learner")
    dataset = param_dict.pop("dataset")
    seed = param_dict.pop("seed")

    ALGO_MAP = {
        'gpr_meta_mll': GPRegressionMetaLearned,
        'gpr_meta_vi': GPRegressionMetaLearnedVI,
        'gpr_meta_svgd': GPRegressionMetaLearnedSVGD,
        'maml': MAMLRegression,
        'neural_process': NPRegressionMetaLearned,
    }
    meta_learner_cls = ALGO_MAP[meta_learner]

    # 1) prepare reults dict
    results_dict = {
        'learner': meta_learner,
        'dataset': dataset,
        'seed': seed,
    }
    results_dict.update(**param_dict)

    try:
        # 1) Generate Data
        from experiments.data_sim import provide_data
        data_train, _, data_test = provide_data(dataset, DATA_SEED)

        # 2) Fit model
        model = meta_learner_cls(data_train, **param_dict, random_seed=seed)
        model.meta_fit(data_test, log_period=5000)

        # 3) evaluate model
        if meta_learner == 'neural_process':
            eval_result = model.eval_datasets(data_test, flatten_y=False)
        else:
            eval_result = model.eval_datasets(data_test)

        if meta_learner == 'maml':
            rmse = eval_result
            results_dict.update(rmse=rmse)
        else:
            ll, rmse, calib_err = eval_result
            results_dict.update(ll=ll, rmse=rmse, calib_err=calib_err)

    except Exception as e:
        print(e)
        results_dict.update(ll=np.nan, rmse=np.nan, calib_err=np.nan)
    return results_dict

def _create_configurations(param_configs):
  confs = []
  for conf_dict in param_configs:
    conf_dict = dict([(key, val if type(val) == list or type(val) == tuple else [val, ]) for key, val in conf_dict.items()])
    conf_product = list(itertools.product(*list(conf_dict.values())))
    conf_product_dicts = [(dict(zip(conf_dict.keys(), conf))) for conf in conf_product]
    confs.extend(conf_product_dicts)
  return confs

def main(args):
    param_configs = [
    {
        'meta_learner': 'gpr_meta_mll',
        'dataset': DATASETS,
        'seed': MODEL_SEEDS,
        'covar_module': ['NN'],
        'mean_module': 'NN',
        'num_iter_fit': 40000,
        'weight_decay': 0.0,
        'task_batch_size': [4],
        'lr_decay': [0.97],
        'lr_params': [5e-3, 1e-3, 5e-4],
        'mean_nn_layers': [LAYER_SIZES],
        'kernel_nn_layers': [LAYER_SIZES],
    },
    {
        'meta_learner': 'maml',
        'dataset': DATASETS,
        'seed': MODEL_SEEDS,
        'num_iter_fit': 40000,
        'task_batch_size': 4,
        'lr_inner': [0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2],
        'layer_sizes': [LAYER_SIZES],
    },
    {
        'meta_learner': 'neural_process',
        'dataset': DATASETS,
        'seed': MODEL_SEEDS,
        'num_iter_fit': 40000,
        'task_batch_size': 4,
        'lr_decay': 0.97,
        'lr_params': 1e-3,
        'r_dim': [32, 64, 124],
        'weight_decay': [1e-4, 1e-3, 1e-2, 1e-1, 2e-1, 4e-1, 8e-1]
    },
    ]

    param_configs = _create_configurations(param_configs)


    result_dicts = []

    answer = input("About to run %i jobs with ray. Proceed? [yes/no]\n" % len(param_configs))
    if answer == 'yes':
        result_dicts += ray.get([fit_eval_meta_algo.remote(param_dict) for param_dict in param_configs])

    result_df = pandas.DataFrame(result_dicts)
    csv_file_name = os.path.join(EXP_DIR, 'baseline_comp_%s.csv' %(datetime.now().strftime("%b_%d_%Y_%H:%M:%S")))
    result_df.to_csv(csv_file_name)
    print(result_df.to_string())
    print("\nDumped the csv file to %s"%csv_file_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run meta mll hyper-parameter search.')
    parser.add_argument('--num_cpus', type=int, default=72, help='dataset')

    args = parser.parse_args()

    ray.init(memory=52428800000, num_cpus=args.num_cpus)
    print('Running', os.path.abspath(__file__), '\n')
    print('\n')

    main(args)