import os
import numpy as np
from matplotlib import pyplot
from scipy import optimize


def sigmoid(z):
    """
    Computes the sigmoid of z.
    """
    return 1.0 / (1.0 + np.exp(-z))
