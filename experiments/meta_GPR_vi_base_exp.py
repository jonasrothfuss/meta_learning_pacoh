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
from experiments.data_sim import SinusoidNonstationaryDataset, MNISTRegressionDataset, PhysionetDataset, GPFunctionsDataset
from src.GPR_meta_vi import GPRegressionMetaLearnedVI

import torch


flags.DEFINE_string('exp_name', default='meta-GPR-VI-base-exp', help='name of the folder in which to dump logs and results')

flags.DEFINE_integer('seed', default=28, help='random seed')
flags.DEFINE_integer('n_threads', default=8, help='number of threads')

# Configuration for GP-Prior learning
flags.DEFINE_float('weight_prior_scale', default=0.1, help='scale of hyper-prior distribution on NN weights')
flags.DEFINE_float('prior_factor', default=0.1, help='factor weighting the importance of the hyper-prior relative to '
                                                     'the meta-likelihood')
flags.DEFINE_string('mean_module', default='NN', help='specifies what to use as mean function of the GP prior')
flags.DEFINE_string('covar_module', default='NN', help='specifies what to use as kernel function of the GP prior')
flags.DEFINE_integer('num_layers', default=5, help='number of neural network layers for GP-prior NNs')
flags.DEFINE_integer('layer_size', default=32, help='number of neural network layers for GP-prior NNs')
flags.DEFINE_float('lr', default=1e-3, help='learning rate for AdamW optimizer')
flags.DEFINE_integer('n_iter_fit', default=100000, help='number of gradient steps')
flags.DEFINE_string('optimizer', default='Adam', help='type of optimizer to use - either \'SGD\' or \'ADAM\'')
flags.DEFINE_integer('svi_batch_size', default=1000, help='number of posterior samples to estimate grads')
#flags.DEFINE_integer('batch_size', 20, help='batch size for meta training, i.e. number of tasks for computing grads') #TODO

# Configuration w.r.t. data
flags.DEFINE_boolean('normalize_data', default=True, help='whether to normalize the data')
flags.DEFINE_string('dataset', default='sin-nonstat', help='meta learning dataset')
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
        dataset = SinusoidNonstationaryDataset(random_state=np.random.RandomState(FLAGS.seed+1))
    elif FLAGS.dataset == 'mnist':
        dataset = MNISTRegressionDataset(random_state=np.random.RandomState(FLAGS.seed+1))
    elif FLAGS.dataset == 'physionet':
        dataset = PhysionetDataset(random_state=np.random.RandomState(FLAGS.seed+1))
    elif FLAGS.dataset == 'gp-funcs':
        dataset = GPFunctionsDataset(random_state=np.random.RandomState(FLAGS.seed+1))
    else:
        raise NotImplementedError('Does not recognize dataset flag')

    data_train = dataset.generate_meta_train_data(n_tasks=FLAGS.n_train_tasks, n_samples=FLAGS.n_train_samples)
    data_test = dataset.generate_meta_test_data(n_tasks=FLAGS.n_test_tasks, n_samples_context=FLAGS.n_context_samples,
                                                n_samples_test=FLAGS.n_test_samples)

    nn_layers = tuple([FLAGS.layer_size for _ in range(FLAGS.num_layers)])

    torch.set_num_threads(FLAGS.n_threads)

    gp_meta = GPRegressionMetaLearnedVI(data_train,
                                        weight_prior_scale=FLAGS.weight_prior_scale,
                                        prior_factor=FLAGS.prior_factor,
                                        covar_module=FLAGS.covar_module,
                                        mean_module=FLAGS.mean_module,
                                        kernel_nn_layers=nn_layers,
                                        mean_nn_layers=nn_layers,
                                        random_seed=FLAGS.seed,
                                        optimizer=FLAGS.optimizer,
                                        lr_params=FLAGS.lr,
                                        num_iter_fit=FLAGS.n_iter_fit,
                                        svi_batch_size=FLAGS.svi_batch_size,
                                        normalize_data=FLAGS.normalize_data,
                                      )

    gp_meta.meta_fit(valid_tuples=data_test[:100], log_period=100)

    test_ll_bayes, rmse_bayes = gp_meta.eval_datasets(data_test, mode='Bayes')
    test_ll_map, rmse_map = gp_meta.eval_datasets(data_test, mode='MAP')

    # save results
    results_dict = {
        'test_ll_bayes': test_ll_bayes,
        'test_rmse_bayes': rmse_bayes,
        'test_ll_map': test_ll_map,
        'rmse_map': rmse_map
    }
    save_results(results_dict, exp_dir, log=True)

if __name__ == '__main__':
  app.run(main)