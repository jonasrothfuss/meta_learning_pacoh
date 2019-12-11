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
from experiments.data_sim import SinusoidNonstationaryDataset, MNISTRegressionDataset, PhysionetDataset, \
    GPFunctionsDataset, SinusoidDataset, CauchyDataset
from src.GPR_meta_svgd import GPRegressionMetaLearnedSVGD

import torch


flags.DEFINE_string('exp_name', default='meta-GPR-VI-base-exp', help='name of the folder in which to dump logs and results')

flags.DEFINE_integer('seed', default=28, help='random seed')
flags.DEFINE_integer('n_threads', default=8, help='number of threads')

# Configuration for GP-Prior learning
flags.DEFINE_string('mean_module', default='NN', help='specifies what to use as mean function of the GP prior')
flags.DEFINE_string('covar_module', default='NN', help='specifies what to use as kernel function of the GP prior')
flags.DEFINE_integer('num_layers', default=4, help='number of neural network layers for GP-prior NNs')
flags.DEFINE_integer('layer_size', default=32, help='number of neural network layers for GP-prior NNs')

flags.DEFINE_float('lr', default=1e-3, help='learning rate for AdamW optimizer')
flags.DEFINE_float('lr_decay', 0.98, help='multiplicative learning rate decay factor applied after every 1000 steps')
flags.DEFINE_string('optimizer', default='Adam', help='type of optimizer to use - either \'SGD\' or \'ADAM\'')
flags.DEFINE_integer('task_batch_size', 2, help='batch size for meta training, i.e. number of tasks for computing grads')
flags.DEFINE_integer('n_iter_fit', default=100000, help='number of gradient steps')

flags.DEFINE_float('weight_prior_std', default=0.5, help='scale of hyper-prior distribution on NN weights')
flags.DEFINE_float('prior_factor', default=0.1, help='factor weighting the importance of the hyper-prior relative to '
                                                     'the meta-likelihood')
flags.DEFINE_string('kernel', default='RBF', help='SVGD kernel function')
flags.DEFINE_float('bandwidth', default=0.1, help='Bandwidth of SVGD kernel function')
flags.DEFINE_integer('num_particles', default=10, help='number of SVGD particles')

# Configuration w.r.t. data
flags.DEFINE_boolean('normalize_data', default=True, help='whether to normalize the data')
flags.DEFINE_string('dataset', default='sin', help='meta learning dataset')
flags.DEFINE_integer('n_train_tasks', default=20, help='number of train tasks')
flags.DEFINE_integer('n_train_samples', default=20, help='number of train samples per task')

flags.DEFINE_integer('n_test_tasks', default=100, help='number of test tasks')
flags.DEFINE_integer('n_context_samples', default=20, help='number of test context points per task')
flags.DEFINE_integer('n_test_samples', default=500, help='number of test evaluation points per task')


FLAGS = flags.FLAGS


def main(argv):
    # setup logging

    logger, exp_dir = setup_exp_doc(FLAGS.exp_name)

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

    data_train = dataset.generate_meta_train_data(n_tasks=FLAGS.n_train_tasks, n_samples=FLAGS.n_train_samples)
    data_test = dataset.generate_meta_test_data(n_tasks=FLAGS.n_test_tasks, n_samples_context=FLAGS.n_context_samples,
                                                n_samples_test=FLAGS.n_test_samples)

    nn_layers = tuple([FLAGS.layer_size for _ in range(FLAGS.num_layers)])

    torch.set_num_threads(FLAGS.n_threads)

    gp_meta = GPRegressionMetaLearnedSVGD(data_train,
                                        weight_prior_std=FLAGS.weight_prior_std,
                                        prior_factor=FLAGS.prior_factor,
                                        covar_module=FLAGS.covar_module,
                                        mean_module=FLAGS.mean_module,
                                        kernel_nn_layers=nn_layers,
                                        mean_nn_layers=nn_layers,
                                        random_seed=FLAGS.seed,
                                        optimizer=FLAGS.optimizer,
                                        lr=FLAGS.lr,
                                        lr_decay=FLAGS.lr_decay,
                                        num_iter_fit=FLAGS.n_iter_fit,
                                        kernel=FLAGS.kernel,
                                        bandwidth=FLAGS.bandwidth,
                                        num_particles=FLAGS.num_particles,
                                        normalize_data=FLAGS.normalize_data,
                                        task_batch_size=FLAGS.task_batch_size
                                      )

    gp_meta.meta_fit(valid_tuples=data_test[:100], log_period=1000)

    test_ll, rmse, calib_err = gp_meta.eval_datasets(data_test)

    # save results
    results_dict = {
        'test_ll': test_ll,
        'test_rmse': rmse,
        'calib_err': calib_err
    }
    print(results_dict)
    save_results(results_dict, exp_dir, log=True)

if __name__ == '__main__':
  app.run(main)