import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_d(x):
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def tanh_d(x):
    return 1 - tanh(x) ** 2


def re_lu(x):
    return np.maximum(0, x)


def re_lu_d(x):
    return np.int64(x > 0)
