import numpy as np


def sigmoid(z):
    """
    Compute the sigmoid of z.
    """
    return 1.0 / (1.0 + np.exp(-z))
