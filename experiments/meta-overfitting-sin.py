import os
import sys
import hashlib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from experiments.util import AsyncExecutor, generate_launch_commands
import experiments.meta_GPR_base_exp

cluster = True

N_THREADS = 4

exp_config = {
    'n_threads': [N_THREADS],
    'seed': [25, 26, 27, 28, 29],
    'weight_decay': [0.0, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4],
    'covar_module': ['NN'],
    'mean_module': ['NN'],
    'n_iter_fit': [100000],
    'n_train_tasks': [2, 5, 10, 20],
    'n_train_samples': [5, 10, 20, 40]
}

command_list = generate_launch_commands(experiments.meta_GPR_base_exp, exp_config)

if cluster :
    for python_cmd in command_list:
        cmd_hash = hashlib.md5(str.encode(python_cmd)).hexdigest()

        bsub_cmd = 'bsub -oo /cluster/project/infk/krause/rojonas/stdout/gp-priors/meta-overfitting/%s.out' \
                   ' -W 12:00'\
                   ' -R "rusage[mem=2048]"' \
                   ' -n %i '% (cmd_hash, N_THREADS)
        cluster_cmd = bsub_cmd + ' ' + python_cmd
        os.system(cluster_cmd)
else:
    exec_fn = lambda cmd: os.system(cmd)
    executor = AsyncExecutor(n_jobs=-1)
    executor.run(exec_fn, command_list)