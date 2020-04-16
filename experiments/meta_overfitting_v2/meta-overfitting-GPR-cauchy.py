import os
import sys
import hashlib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from experiments.util import AsyncExecutor, generate_launch_commands
import experiments.meta_overfitting_v2.meta_GPR_overfitting_base
import numpy as np

cluster = False

N_THREADS = 1

exp_config = {
    'exp_name': ['meta-overfitting-v2-maml-cauchy'],
    'dataset': ['cauchy'],
    'n_threads': [N_THREADS],
    'seed': list(range(30, 45)),
    'data_seed': [28],
    'weight_decay': [0.0],
    'covar_module': ['NN'],
    'mean_module': ['NN'],
    'num_layers': [4],
    'layer_size': [32],
    'n_iter_fit': [40000],
    'n_train_tasks': [2, 4, 8, 16, 32, 64, 128, 256],
    'n_test_tasks': [200],
    'n_context_samples': [20, 40],
    'n_test_samples': [100],
}

command_list = generate_launch_commands(experiments.meta_overfitting_v2.meta_GPR_overfitting_base, exp_config)

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