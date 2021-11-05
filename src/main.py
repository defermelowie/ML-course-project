import os
from PIL import Image
import numpy as np
from matplotlib import pyplot
from scipy import optimize


def sigmoid(z):
    """
    Computes the sigmoid of z.
    """
    return 1.0 / (1.0 + np.exp(-z))


img = Image.open(
    './data/Garbage classification/cardboard/cardboard384.jpg')

#print('height: %s, width: %s' % img.size)

(R_chan, G_chan, B_chan) = img.split()

R_chan.show()
