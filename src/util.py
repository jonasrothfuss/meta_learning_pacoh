import numpy as np
import os
import logging
from absl import flags


def _handle_input_dimensionality(x: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray):

    if x.ndim == 1:
        x = np.expand_dims(x, -1)
    if y.ndim == 1:
        y = np.expand_dims(y, -1)

    assert x.shape[0] == y.shape[0]
    assert x.ndim == 2
    assert y.ndim == 2

    return x, y

def get_logger(log_dir=None, log_file='output.log', expname=''):

    if log_dir is None and flags.FLAGS.is_parsed() and hasattr(flags.FLAGS, 'log_dir'):
        log_dir = flags.FLAGS.log_dir

    logger = logging.getLogger('gp-priors')
    logger.setLevel(logging.INFO)

    if len(logger.handlers) == 0:

        #formatting
        if len(expname) > 0:
            expname = ' %s - '%expname
        formatter = logging.Formatter('[%(asctime)s -' + '%s'%expname +  '%(levelname)s]  %(message)s')

        # Stream Handler
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        sh.setLevel(logging.INFO)
        logger.addHandler(sh)

        logger.propagate = False

        # File Handler
        if log_dir is not None and len(log_dir) > 0:
            fh = logging.FileHandler(os.path.join(log_dir, log_file))
            fh.setFormatter(formatter)
            fh.setLevel(logging.INFO)
            logger.addHandler(fh)
            logger.log_dir = log_dir
        else:
            logger.log_dir = None
    return logger



""" ------ Lightweight mltiprocessing utilities ------ """

from multiprocessing import Process
import multiprocessing
import numpy as np

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
                          print('task %i of %i'%(n_tasks-len(tasks), n_tasks))
                        next_task = tasks.pop(0)
                        self._pool[i] = _start_process(target, next_task)
                    else:
                        workers_idle[i] = True

    def _populate_pool(self):
        self._pool = [_start_process(_dummy_fun) for _ in range(self.num_workers)]

class LoopExecutor:

    def run(self, target, *args_iter, verbose=False):
        tasks = list(zip(*args_iter))
        n_tasks = len(tasks)


        for i, task in enumerate(tasks):
            target(*task)
            if verbose:
                print('task %i of %i'%(n_tasks-len(tasks), n_tasks))

def _start_process(target, args=None):
    if args:
        p = Process(target=target, args=args)
    else:
        p = Process(target=target)
    p.start()
    return p

def _dummy_fun():
    pass
