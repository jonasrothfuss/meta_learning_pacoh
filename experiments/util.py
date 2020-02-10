import os
import copy
import json
import hashlib
import sys
import glob
import collections
import itertools
import multiprocessing
import pandas as pd
from absl import flags
from meta_learn.util import get_logger


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


def collect_exp_results(exp_name, verbose=True):
    exp_dir = os.path.join(DATA_DIR, exp_name)
    no_results_counter = 0

    exp_dicts = []
    for exp_sub_dir in glob.glob(exp_dir + '/*'):
        config_file = os.path.join(exp_sub_dir, 'config.json')
        results_file = os.path.join(exp_sub_dir, 'results.json')

        if os.path.isfile(config_file) and os.path.isfile(results_file):
            with open(config_file, 'r') as f:
                exp_dict = json.load(f)
            with open(results_file, 'r') as f:
                exp_dict.update(json.load(f))
            exp_dicts.append(exp_dict)
        else:
            no_results_counter += 1

    if verbose:
        logger = get_logger()
        logger.info('Parsed results %s - found %i folders with results and %i folders without results'
                    %(exp_name, len(exp_dicts), no_results_counter))

    return pd.DataFrame(data=exp_dicts)


def generate_launch_commands(module, exp_config, check_flags=True):
    # create base command without flags
    base_cmd = generate_base_command(module)

    if check_flags:
        allowed_flags = set(module.FLAGS.flag_values_dict().keys())
        for key, value in exp_config.items():
            assert hasattr(value, '__iter__')
            assert key in allowed_flags, "%s is not a flag in %s"%(key, str(module))


    config_product = list(itertools.product(*list(exp_config.values())))
    config_product_dicts = [(dict(zip(exp_config.keys(), conf))) for conf in config_product]

    # add flags to the base command
    cmds = []
    for config_dict in config_product_dicts:
        cmd = base_cmd
        for (key, value) in config_dict.items():
            cmd += " --%s=%s"%(str(key), str(value))
        cmds.append(cmd)

    return cmds


def generate_base_command(module):
    interpreter_script = sys.executable
    base_exp_script = os.path.abspath(module.__file__)
    base_cmd = interpreter_script + ' ' + base_exp_script
    return base_cmd


class AsyncExecutor:

    def __init__(self, n_jobs=1):
        self.num_workers = n_jobs if n_jobs > 0 else multiprocessing.cpu_count()
        self._pool = []
        self._populate_pool()

    def run(self, target, *args_iter, verbose=False):
        workers_idle = [False] * self.num_workers
        tasks = list(zip(*args_iter))
        n_tasks = len(tasks)

        while not all(workers_idle):
            for i in range(self.num_workers):
                if not self._pool[i].is_alive():
                    self._pool[i].terminate()
                    if len(tasks) > 0:
                        if verbose:
                          print(n_tasks-len(tasks))
                        next_task = tasks.pop(0)
                        self._pool[i] = _start_process(target, next_task)
                    else:
                        workers_idle[i] = True

    def _populate_pool(self):
        self._pool = [_start_process(_dummy_fun) for _ in range(self.num_workers)]


def _start_process(target, args=None):
    if args:
        p = multiprocessing.Process(target=target, args=args)
    else:
        p = multiprocessing.Process(target=target)
    p.start()
    return p

def _dummy_fun():
    pass