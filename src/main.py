from typing import List
import numpy as np
from csv import DictReader
import logging

from imageLoader import ImageLoader
from neuralNetwork import NeuralNetwork

# Set up logging
logging.basicConfig(
    format='[%(asctime)s | %(levelname)s] %(message)s', level=logging.DEBUG)


###############################################################
#                         Load images                         #
###############################################################

# Image constants
IMG_RESCALE_FACTOR = 15
DATA_PATHS = './data/path_list.csv'

# Load path_list from csv file
path_list = []
with open(DATA_PATHS, mode='r') as fd:
    dictReader = DictReader(fd)
    for dict in dictReader:
        path_list.append(dict)

# Load X & y from path_list
imageLoader = ImageLoader(path_list, 'cv0', scale_factor=IMG_RESCALE_FACTOR)
(X_, Y) = imageLoader.load_images()

# Normalize X
X = X_*(1.0/255.0)


###############################################################
#                       Neural network                        #
###############################################################

# Neural network constants
NN_EPSILON_INIT = 0.12
NN_LAMBDA = 0.0

# Build neural network
nn = NeuralNetwork((X.shape[1], X.shape[1], len(np.unique(Y))),
                   epsilon_init=NN_EPSILON_INIT,
                   lambda_=NN_LAMBDA
                   )
nn.feed_forward(X)
