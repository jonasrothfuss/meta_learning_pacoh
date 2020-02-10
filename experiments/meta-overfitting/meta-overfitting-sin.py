import os
import sys
import hashlib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from experiments.util import AsyncExecutor, generate_launch_commands
import experiments.meta_GPR_mll_base_exp
import numpy as np

cluster = False

N_THREADS = 2

exp_config = {
    'exp_name': ['meta-overfitting-sin'],
    'dataset': ['sin'],
    'n_threads': [N_THREADS],
    'seed': [31, 32, 33, 34, 35],
    'weight_decay': list(np.logspace(-3, -0.25, num=10)),
    'covar_module': ['NN'],
    'mean_module': ['NN'],
    'num_layers': [4],
    'layer_size': [32],
    'n_iter_fit': [30000],
    'n_train_tasks': [10, 20],
    'n_train_samples': [5, 10, 20],
    'n_test_tasks': [1000],
    'n_context_samples': [5],
    'n_test_samples': [100],
}

command_list = generate_launch_commands(experiments.meta_GPR_mll_base_exp, exp_config)

if cluster :
    cluster_cmds = []
    for python_cmd in command_list:
        cmd_hash = hashlib.md5(str.encode(python_cmd)).hexdigest()

        bsub_cmd = 'bsub -oo /cluster/project/infk/krause/rojonas/stdout/gp-priors/meta-overfitting/%s.out' \
                   ' -W 03:59'\
                   ' -R "rusage[mem=1048]"' \
                   ' -n %i '% (cmd_hash, N_THREADS)
        cluster_cmds.append(bsub_cmd + ' ' + python_cmd)
    answer = input("About to submit %i compute jobs to the cluster. Proceed? [yes/no]\n"%len(cluster_cmds))
    if answer == 'yes':
        for cmd in cluster_cmds:
            os.system(cmd)
else:
    exec_fn = lambda cmd: os.system(cmd)
    executor = AsyncExecutor(n_jobs=-1)
    executor.run(exec_fn, command_list)