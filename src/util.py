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

    if log_dir is None and flags.FLAGS.is_parsed():
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

    return logger