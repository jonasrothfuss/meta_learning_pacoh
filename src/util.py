import numpy as np


def _handle_input_dimensionality(x: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray):

    if x.ndim == 1:
        x = np.expand_dims(x, -1)
    if y.ndim == 1:
        y = np.expand_dims(y, -1)

    assert x.shape[0] == y.shape[0]
    assert x.ndim == 2
    assert y.ndim == 2

    return x, y