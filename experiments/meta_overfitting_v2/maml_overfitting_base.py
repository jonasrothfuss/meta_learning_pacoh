import os
import sys

# add project dir to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from absl import flags
from absl import app
import numpy as np
from pprint import pprint
from meta_learn.util import get_logger
from experiments.util import *
from experiments.data_sim import SinusoidNonstationaryDataset, MNISTRegressionDataset, \
    PhysionetDataset, GPFunctionsDataset, SinusoidDataset, CauchyDataset, provide_data
from meta_learn.MAML import MAMLRegression

import torch

flags.DEFINE_string('exp_name', default='meta-overfitting-v2-GPR-base-exp',
                    help='name of the folder in which to dump logs and results')

flags.DEFINE_integer('seed', default=28, help='random seed')
flags.DEFINE_integer('data_seed', default=2158, help='random seed')
flags.DEFINE_integer('n_threads', default=4, help='number of threads')

# Configuration for GP-Prior learning
flags.DEFINE_integer('num_layers', default=4, help='number of neural network layers for GP-prior NNs')
flags.DEFINE_integer('layer_size', default=128, help='number of neural network layers for GP-prior NNs')

flags.DEFINE_float('lr', default=1e-3, help='learning rate for AdamW optimizer')
flags.DEFINE_float('lr_inner', default=0.05, help='learning rate for AdamW optimizer')
flags.DEFINE_integer('batch_size', 5, help='batch size for meta training, i.e. number of tasks for computing grads')
flags.DEFINE_string('optimizer', default='Adam', help='type of optimizer to use - either \'SGD\' or \'ADAM\'')
flags.DEFINE_integer('n_iter_fit', default=50000, help='number of gradient steps')

# Configuration w.r.t. data
flags.DEFINE_boolean('normalize_data', default=True, help='whether to normalize the data')
flags.DEFINE_string('dataset', default='sin', help='meta learning dataset')
flags.DEFINE_integer('n_train_tasks', default=2, help='number of train tasks')
flags.DEFINE_integer('n_test_tasks', default=100, help='number of test tasks')
flags.DEFINE_integer('n_context_samples', default=20, help='number of test context points per task')
flags.DEFINE_integer('n_test_samples', default=500, help='number of test evaluation points per task')

FLAGS = flags.FLAGS


def main(argv):
    # setup logging

    logger, exp_dir = setup_exp_doc(FLAGS.exp_name)

    if FLAGS.dataset == 'swissfel':
        raise NotImplementedError
    else:
        if FLAGS.dataset == 'sin-nonstat':
            dataset = SinusoidNonstationaryDataset(random_state=np.random.RandomState(FLAGS.seed + 1))
        elif FLAGS.dataset == 'sin':
            dataset = SinusoidDataset(random_state=np.random.RandomState(FLAGS.seed + 1))
        elif FLAGS.dataset == 'cauchy':
            dataset = CauchyDataset(random_state=np.random.RandomState(FLAGS.seed + 1))
        elif FLAGS.dataset == 'mnist':
            dataset = MNISTRegressionDataset(random_state=np.random.RandomState(FLAGS.seed + 1))
        elif FLAGS.dataset == 'physionet':
            dataset = PhysionetDataset(random_state=np.random.RandomState(FLAGS.seed + 1))
        elif FLAGS.dataset == 'gp-funcs':
            dataset = GPFunctionsDataset(random_state=np.random.RandomState(FLAGS.seed + 1))
        else:
            raise NotImplementedError('Does not recognize dataset flag')

        meta_train_data = dataset.generate_meta_test_data(n_tasks=260, n_samples_context=FLAGS.n_context_samples,
                                                          n_samples_test=FLAGS.n_test_samples)
        meta_test_data = dataset.generate_meta_test_data(n_tasks=FLAGS.n_test_tasks, n_samples_context=FLAGS.n_context_samples,
                                                    n_samples_test=FLAGS.n_test_samples)

    nn_layers = tuple([FLAGS.layer_size for _ in range(FLAGS.num_layers)])
    torch.set_num_threads(FLAGS.n_threads)

    # only take meta-train context for training
    data_train = [(context_x, context_y) for context_x, context_y, _, _ in meta_train_data]
    data_train = data_train[:FLAGS.n_train_tasks]

    gp_meta = MAMLRegression(data_train,
                              num_iter_fit=FLAGS.n_iter_fit,
                              layer_sizes=nn_layers,
                              task_batch_size=FLAGS.batch_size,
                              lr_inner=FLAGS.lr_inner,
                              lr_meta=FLAGS.lr,
                              random_seed=FLAGS.seed,
                              optimizer=FLAGS.optimizer,
                              normalize_data=FLAGS.normalize_data
                              )

    gp_meta.meta_fit(log_period=1000)

    test_rmse_meta_train = gp_meta.eval_datasets(meta_train_data)
    test_rmse_meta_test = gp_meta.eval_datasets(meta_test_data)

    # save results
    results_dict = {
        'test_rmse_meta_train': test_rmse_meta_train,
        'test_rmse_meta_test': test_rmse_meta_test,
    }

    pprint(results_dict)

    save_results(results_dict, exp_dir, log=True)

if __name__ == '__main__':
    app.run(main)