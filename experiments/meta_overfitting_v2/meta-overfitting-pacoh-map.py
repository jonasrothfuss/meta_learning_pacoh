import os
import sys
import hashlib
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from absl import flags
from absl import app


flags.DEFINE_integer('n_workers', default=-1, help='number of cpus to use')
flags.DEFINE_boolean('cluster', default=False, help='whether to submit jobs with bsub')
flags.DEFINE_string('datasets', default='sin,cauchy', help='specifies which dataset to use')

FLAGS = flags.FLAGS

N_THREADS = 1

def main(argv):
    from experiments.util import AsyncExecutor, generate_launch_commands
    import experiments.meta_overfitting_v2.meta_GPR_overfitting_base

    command_list = []

    for dataset in FLAGS.datasets.split(','):
        if dataset == 'sin':
            n_context_samples = [5, 10, 20]
        elif dataset == 'cauchy':
            n_context_samples = [20, 40]
        else:
            raise AssertionError('dataset must be either of [sin, cauchy]')

        exp_config = {
            'exp_name': ['meta-overfitting-v2-pacoh-map-%s'%dataset],
            'dataset': [dataset],
            'n_threads': [N_THREADS],
            'seed': list(range(30, 55)),
            'data_seed': [28],
            'weight_decay': list(np.logspace(-4, -0.25, num=10)),
            'covar_module': ['NN'],
            'mean_module': ['NN'],
            'num_layers': [4],
            'layer_size': [32],
            'n_iter_fit': [30000],
            'n_train_tasks': [2, 4, 8, 16, 32, 64, 128, 256, 512],
            'n_test_tasks': [200],
            'n_context_samples': n_context_samples,
            'n_test_samples': [100],
        }

        command_list.extend(
            generate_launch_commands(experiments.meta_overfitting_v2.meta_GPR_overfitting_base, exp_config))

    if FLAGS.cluster :
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
        answer = input("About to run %i compute jobs locally on %i workers. "
                       "Proceed? [yes/no]\n" %(len(command_list), FLAGS.n_workers))
        if answer == 'yes':
            exec_fn = lambda cmd: os.system(cmd)
            executor = AsyncExecutor(n_jobs=FLAGS.n_workers)
            executor.run(exec_fn, command_list)

if __name__ == '__main__':
    app.run(main)