import os
import copy
import json
import hashlib
import collections
from absl import flags
from src.util import get_logger

DEFAULT_FLAGS = ['logtostderr', 'alsologtostderr', 'v', 'verbosity',
                  'stderrthreshold', 'showprefixforinfo', 'run_with_pdb', 'pdb_post_mortem',
                  'run_with_profiling', 'profile_file', 'use_cprofile_for_profiling',
                  'only_check_args', '?', 'help', 'helpshort', 'helpfull', 'helpxml']

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data')


def setup_exp_doc(exp_name, data_dir=None):

    # create dictionary of flags / hyperparams
    flags_dict = get_flags_dict()
    flags_dict['exp_name'] = exp_name

    # generate unique task identifier
    task_hash = hash_dict(flags_dict)
    flags_dict['task_hash'] = task_hash

    # create directory for experiment task, initialize logger and save the flags_dict
    exp_dir = create_exp_dir(exp_name, task_hash, data_dir=data_dir)
    logger = get_logger(log_dir=exp_dir, expname=exp_name)
    save_dict(flags_dict, os.path.join(exp_dir, 'config.json'))

    flags_table_str = dict_to_tabular_str(flags_dict)
    logger.info(" ------ Starting experiment: %s ------ \n"%exp_name+\
                "----------------------------------------\n"+\
                "             Configuration              \n"+\
                "----------------------------------------"+\
                "%s"%flags_table_str+\
                "----------------------------------------\n")

    return logger, exp_dir

def save_results(results_dict, exp_dir, log=True):
    results_file = os.path.join(exp_dir, 'results.json')
    save_dict(results_dict, results_file)

    if log:
        logger = get_logger(log_dir=exp_dir)
        results_table_str = dict_to_tabular_str(results_dict)

        logger.info("\n"+
                    "----------------------------------------\n" + \
                    "                   Results              \n" + \
                    "----------------------------------------" + \
                    "%s" % results_table_str + \
                    "----------------------------------------\n")


def create_exp_parent_dir(exp_name, data_dir=None):
    if data_dir is None:
        data_dir = DATA_DIR
    exp_parent_dir = os.path.join(data_dir, exp_name)
    if not os.path.isdir(exp_parent_dir):
        os.mkdir(exp_parent_dir)
    return exp_parent_dir

def create_exp_dir(exp_name, task_hash, data_dir=None):
    exp_parent_dir = create_exp_parent_dir(exp_name, data_dir=data_dir)
    exp_dir = os.path.join(exp_parent_dir, str(task_hash))
    if not os.path.isdir(exp_dir):
        os.mkdir(exp_dir)
    return exp_dir

def get_flags_dict():
    flags_dict = copy.deepcopy(flags.FLAGS.flag_values_dict())

    # remove absl's default flags from the dict
    list(map(flags_dict.__delitem__, DEFAULT_FLAGS))

    return flags_dict

def hash_dict(dict):
    return hashlib.md5(str.encode((json.dumps(dict, sort_keys=True)))).hexdigest()

def save_dict(dict, dump_path):
    with open(dump_path, 'w') as json_file:
        json.dump(dict, json_file, indent=4, sort_keys=True)

def dict_to_tabular_str(dict):
    s = "\n"
    format = "{:<25}{:<10}"
    for key, value in collections.OrderedDict(dict).items():
        s += format.format(key, value) + '\n'
    return s