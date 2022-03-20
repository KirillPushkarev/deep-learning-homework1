import numpy as np


def l1_regularization(x):
    return np.sum(np.abs(x))


def l1_regularization_d(x):
    return np.sign(x)


def l2_regularization(x):
    return np.sum(x ** 2)


def l2_regularization_d(x):
    return 2 * x
