import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from experiments.util import generate_launch_commands

import experiments.hyperparam_search.meta_mll_hyperparm as meta_mll_hparam

import experiments.hyperparam_search.meta_svgd_hyperparam as meta_svgd_hparam
import experiments.hyperparam_search.meta_vi_hyperparam as meta_vi_hparam

CLUSTER = False
DRY = False

NUM_CPUS = 20

command_list = []

for hparam_search_module in [meta_mll_hparam, meta_svgd_hparam, meta_vi_hparam]:

    exp_config = {
        'dataset': ['cauchy_20', 'sin_20', 'physionet_0', 'physionet_2', 'swissfel'],
        'covar_module': ['NN'],
        'num_cpus': [NUM_CPUS]
    }
    command_list += generate_launch_commands(hparam_search_module, exp_config, check_flags=False)

print(command_list)


if CLUSTER:
    cluster_cmds = []
    for python_cmd in command_list:
        bsub_cmd = 'bsub' \
                   ' -W 23:00'\
                   ' -R "rusage[mem=4500]"' \
                   ' -R "rusage[ngpus_excl_p=1]"' \
                   ' -n %i '% (NUM_CPUS)
        cluster_cmds.append(bsub_cmd + ' ' + python_cmd)

    answer = input("About to submit %i compute jobs to the cluster. Proceed? [yes/no]\n"%len(cluster_cmds))
    if answer == 'yes':
        for cmd in cluster_cmds:
            if DRY:
                print(cmd)
            else:
                os.system(cmd)

else:
    answer = input("About to run %i compute jobs in a for loop. Proceed? [yes/no]\n"%len(command_list))
    if answer == 'yes':
        for cmd in command_list:
            if DRY:
                print(cmd)
            else:
                os.system(cmd)
