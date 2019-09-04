import os
import sys

# add project dir to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)


from absl import flags
from absl import app
import numpy as np
from src.util import get_logger
from experiments.util import *
from experiments.data_sim import SinusoidMetaDataset, SinusoidNonstationaryDataset, MNISTRegressionDataset
from src.GPR_meta_mll import GPRegressionMetaLearned

import torch


EXP_NAME = 'meta-overfitting-sin'


flags.DEFINE_integer('seed', default=28, help='random seed')
flags.DEFINE_integer('n_threads', default=8, help='number of threads')

# Configuration for GP-Prior learning
flags.DEFINE_float('weight_decay', default=0.0, help='weight decay for meta-learning the prior')
flags.DEFINE_integer("n_iter_fit", default=100000, help='number of gradient steps')
flags.DEFINE_string('learning_mode', default='both', help='specifies what to use as mean function of the GP prior')
flags.DEFINE_string('mean_module', default='NN', help='specifies what to use as mean function of the GP prior')
flags.DEFINE_string('covar_module', default='SE', help='specifies what to use as kernel function of the GP prior')
flags.DEFINE_integer('num_layers', default=5, help='number of neural network layers for GP-prior NNs')
flags.DEFINE_integer('layer_size', default=128, help='number of neural network layers for GP-prior NNs')
flags.DEFINE_float('lr', default=1e-3, help='learning rate for AdamW optimizer')
flags.DEFINE_integer('batch_size', 20, help='batch size for meta training, i.e. number of tasks for computing grads')

# Configuration w.r.t. data
flags.DEFINE_integer('n_train_tasks', default=10, help='number of train tasks')
flags.DEFINE_integer('n_train_samples', default=20, help='number of train samples per task')

flags.DEFINE_integer('n_test_tasks', default=100, help='number of test tasks')
flags.DEFINE_integer('n_context_samples', default=20, help='number of test context points per task')
flags.DEFINE_integer('n_test_samples', default=500, help='number of test evaluation points per task')


FLAGS = flags.FLAGS



def main(argv):
    # setup logging

    logger, exp_dir = setup_exp_doc(EXP_NAME)

    rds = np.random.RandomState(FLAGS.seed)

    dataset = SinusoidNonstationaryDataset(random_state=rds)
    data_train = dataset.generate_meta_train_data(n_tasks=FLAGS.n_train_tasks, n_samples=FLAGS.n_train_samples)
    data_test = dataset.generate_meta_test_data(n_tasks=FLAGS.n_test_tasks, n_samples_context=FLAGS.n_context_samples,
                                                n_samples_test=FLAGS.n_test_samples)

    nn_layers = tuple([FLAGS.layer_size for _ in range(FLAGS.num_layers)])

    torch.set_num_threads(FLAGS.n_threads)

    gp_meta = GPRegressionMetaLearned(data_train,
                                      learning_mode=FLAGS.learning_mode,
                                      num_iter_fit=FLAGS.n_iter_fit,
                                      covar_module=FLAGS.covar_module,
                                      mean_module=FLAGS.mean_module,
                                      kernel_nn_layers=nn_layers,
                                      mean_nn_layers=nn_layers,
                                      weight_decay=FLAGS.weight_decay,
                                      lr_params=FLAGS.lr,
                                      random_seed=FLAGS.seed,
                                      task_batch_size=FLAGS.batch_size
                                      )

    gp_meta.meta_fit(valid_tuples=data_test[:100], log_period=500)

    test_ll, rmse = gp_meta.eval_datasets(data_test)

    # save results
    results_dict = {
        'test_ll': test_ll,
        'test_rmse': rmse
    }
    save_results(results_dict, exp_dir, log=True)

if __name__ == '__main__':
  app.run(main)