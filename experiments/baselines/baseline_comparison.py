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
EXP_DIR = os.path.join(DATA_DIR, 'baseline_comparison')
if not os.path.isdir(EXP_DIR): os.makedirs(EXP_DIR)

# Configuration w.r.t. data
DATASETS = ['cauchy_20', 'sin_20', 'physionet_0', 'physionet_2', 'swissfel']


DATA_SEED = 28
MODEL_SEEDS = [22, 23, 24, 25, 26]

LAYER_SIZES = [32, 32, 32, 32]



def fit_eval_GPR_mll(param_dict):
    sys.path.append(BASE_DIR)

    dataset = param_dict.pop("dataset")
    seed = param_dict.pop("seed")

    # 1) prepare reults dict
    results_dict = {
        'learner': 'gpr_mll',
        'dataset': dataset,
        'seed': seed,
    }
    results_dict.update(**param_dict)

    # 2) generate data
    from experiments.data_sim import provide_data
    data_train, _, data_test = provide_data(dataset, DATA_SEED)

    # 3) fit and evaluate

    @ray.remote
    def fit_eval(x_context, y_context, x_test, y_test, params):
        from meta_learn.GPR_mll import GPRegressionLearned
        torch.set_num_threads(1)
        model = GPRegressionLearned(x_context, y_context, **params, random_seed=seed)
        model.fit(verbose=False)
        return model.eval(x_test, y_test)

    results = ray.get([fit_eval.remote(*data, param_dict) for data in data_test])
    results = list(zip(*results))
    assert len(results) == 3

    results_dict['ll'] = np.mean(results[0])
    results_dict['rmse'] = np.mean(results[1])
    results_dict['calib_err'] = np.mean(results[2])

    return results_dict

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
    # {
    #     'meta_learner': 'gpr_meta_mll',
    #     'dataset': [DATASET],
    #     'seed': MODEL_SEEDS,
    #     'covar_module': ['SE', 'NN'],
    #     'mean_module': 'NN',
    #     'num_iter_fit': 40000,
    #     'weight_decay': [0.01, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
    #     'task_batch_size': 2,
    #     'lr_decay': [0.97],
    #     'mean_nn_layers': [LAYER_SIZES],
    #     'kernel_nn_layers': [LAYER_SIZES],
    # },
    # {
    #     'meta_learner': 'gpr_meta_vi',
    #     'dataset': [DATASET],
    #     'seed': MODEL_SEEDS,
    #     'covar_module': ['SE', 'NN'],
    #     'mean_module': 'NN',
    #     'num_iter_fit': 30000,
    #     'svi_batch_size': 10,
    #     'task_batch_size': 4,
    #     'cov_type': 'diag',
    #     'lr': 3e-3,
    #     'lr_decay': [0.90],
    #     'prior_factor': [5e-5, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-3, 5e-2, 1e-1, 5e-1],
    #     'mean_nn_layers': [LAYER_SIZES],
    #     'kernel_nn_layers': [LAYER_SIZES],
    # },
    # {
    #     'meta_learner': 'gpr_meta_svgd',
    #     'dataset': [DATASET],
    #     'seed': MODEL_SEEDS,
    #     'covar_module': ['SE', 'NN'],
    #     'mean_module': 'NN',
    #     'num_iter_fit': 40000,
    #     'bandwidth': [None],
    #     'task_batch_size': 4,
    #     'num_particles': 10,
    #     'prior_factor': [5e-5, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3],
    #     'mean_nn_layers': [LAYER_SIZES],
    #     'kernel_nn_layers': [LAYER_SIZES],
    # },
    {
        'meta_learner': 'gpr_meta_mll',
        'dataset': DATASETS,
        'seed': MODEL_SEEDS,
        'covar_module': ['SE', 'NN'],
        'mean_module': 'NN',
        'num_iter_fit': 40000,
        'weight_decay': 0.0,
        'task_batch_size': 4,
        'lr_decay': [0.97],
        'mean_nn_layers': [LAYER_SIZES],
        'kernel_nn_layers': [LAYER_SIZES],
    },
    {
        'meta_learner': 'maml',
        'dataset': DATASETS,
        'seed': MODEL_SEEDS,
        'num_iter_fit': 30000,
        'task_batch_size': 4,
        'lr_inner': [0.02, 0.05, 0.1],
        'layer_sizes': [LAYER_SIZES],
    },
    {
        'meta_learner': 'neural_process',
        'dataset': DATASETS,
        'seed': MODEL_SEEDS,
        'num_iter_fit': 30000,
        'task_batch_size': 4,
        'lr_decay': 0.97,
        'lr_params': 1e-3,
        'r_dim': [32, 64],
        'weight_decay': [1e-2, 1e-1, 2e-1, 4e-1, 8e-1]
    },
    ]

    param_configs = _create_configurations(param_configs)


    result_dicts = []

    answer = input("About to run %i jobs with ray. Proceed? [yes/no]\n" % len(param_configs))
    if answer == 'yes':
        result_dicts += ray.get([fit_eval_meta_algo.remote(param_dict) for param_dict in param_configs])


    param_configs_gpr_mll = [
        {
        'dataset': DATASETS,
        'seed': MODEL_SEEDS,
        'covar_module': ['SE'],
        'mean_module': ['constant'],
        'learning_mode': ['both', 'vanilla'],
        'num_iter_fit': 20000,
        'mean_nn_layers': [LAYER_SIZES],
    }
    ]
    param_configs_gpr_mll = _create_configurations(param_configs_gpr_mll)
    result_dicts += [fit_eval_GPR_mll(param_dict) for param_dict in param_configs_gpr_mll]

    result_df = pandas.DataFrame(result_dicts)
    csv_file_name = os.path.join(EXP_DIR, 'baseline_comp_%s.csv' %(datetime.now().strftime("%b_%d_%Y_%H:%M:%S")))
    result_df.to_csv(csv_file_name)
    print(result_df.to_string())
    print("\nDumped the csv file to %s"%csv_file_name)

if __name__ == '__main__':
    ray.init(memory=52428800000)

    parser = argparse.ArgumentParser(description='Run meta mll hyper-parameter search.')
    parser.add_argument('--num_cpus', type=int, default=64, help='dataset')

    args = parser.parse_args()

    print('Running', os.path.abspath(__file__), '\n')
    print('\n')

    main(args)