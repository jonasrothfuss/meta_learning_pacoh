import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from experiments.util import generate_launch_commands

import experiments.hyperparam_search.meta_mll_hyperparm as meta_mll_hparam

import experiments.hyperparam_search.meta_svgd_hyperparam as meta_svgd_hparam
import experiments.hyperparam_search.meta_vi_hyperparam as meta_vi_hparam

from absl import flags
from absl import app


flags.DEFINE_integer('n_cpus', default=32, help='number of cpus to use')
flags.DEFINE_integer('n_gpus', default=1, help='number of gpus to use')
flags.DEFINE_string('dataset', default='sin', help='specifies which dataset to use')
flags.DEFINE_string('algos', default='map,vi,svgd', help='specifies which dataset to use')
flags.DEFINE_string('metric', default='test_rmse', help='specifies which metric to optimize')
flags.DEFINE_boolean('cluster', default=False, help='whether to submit jobs with bsub')
flags.DEFINE_boolean('dry', default=False, help='whether to only print launch commands')
flags.DEFINE_boolean('load_analysis', default=False, help='whether to load the analysis from existing dirs')
flags.DEFINE_boolean('resume', default=False, help='whether to resume checkpointed tune session')

FLAGS = flags.FLAGS


algo_map_dict = {'map': meta_mll_hparam,
                 'vi': meta_vi_hparam,
                 'svgd': meta_svgd_hparam}

def main(argv):
    assert FLAGS.dataset in ['sin', 'cauchy']

    hparam_search_modules = [algo_map_dict[algo_str] for algo_str in FLAGS.algos.split(',')]

    command_list = []

    for hparam_search_module in hparam_search_modules:

        exp_config = {
            'dataset': ['%s_%i'%(FLAGS.dataset, n_tasks) for n_tasks in reversed([5, 10, 20, 40, 80, 160, 320])],
            'covar_module': ['NN'],
            'num_cpus': [2 * FLAGS.n_cpus],
            'metric': [FLAGS.metric]
        }
        if FLAGS.load_analysis:
            exp_config['load_analysis'] = [True]
        if FLAGS.resume:
            exp_config['resume'] = [True]
        command_list += generate_launch_commands(hparam_search_module, exp_config, check_flags=False)

    print(command_list)


    if FLAGS.cluster:
        cluster_cmds = []
        for python_cmd in command_list:
            bsub_cmd = 'bsub' \
                       ' -W %i:59'%(3 if FLAGS.load_analysis else 23) + \
                       ' -R "rusage[mem=6000]"' + \
                       ' -R "rusage[ngpus_excl_p=%i]"'%FLAGS.n_gpus + \
                       ' -R "span[hosts=1]"' \
                       ' -n %i '% (FLAGS.n_cpus)
            cluster_cmds.append(bsub_cmd + ' ' + python_cmd)

        answer = input("About to submit %i compute jobs to the cluster. Proceed? [yes/no]\n"%len(cluster_cmds))
        if answer == 'yes':
            for cmd in cluster_cmds:
                if FLAGS.dry:
                    print(cmd)
                else:
                    os.system(cmd)

    else:
        answer = input("About to run %i compute jobs in a for loop. Proceed? [yes/no]\n"%len(command_list))
        if answer == 'yes':
            for cmd in command_list:
                if FLAGS.dry:
                    print(cmd)
                else:
                    os.system(cmd)

if __name__ == '__main__':
    app.run(main)