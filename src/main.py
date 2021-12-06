from typing import List
import numpy as np
from csv import DictReader
import logging

from imageLoader import ImageLoader
from neuralNetwork import NeuralNetwork

# Set up logging
logging.basicConfig(
    format='[%(asctime)s | %(levelname)s] %(message)s', level=logging.INFO)

# General constants
classes = {'papier': 0, 'glas': 1, 'pmd': 2, 'restafval': 3}

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
# Possibilities: {'test', 'cv0', 'cv1', 'cv2', 'cv3'}
sets = {'cv0', 'cv1', 'cv2', 'cv3'}
imageLoader = ImageLoader(path_list, sets, classes,
                          scale_factor=IMG_RESCALE_FACTOR)
(X_, Y) = imageLoader.load_images()

# Normalize X
X = X_*(1.0/255.0)

# Load & normalize test set
imageLoader = ImageLoader(
    path_list, {'test'}, classes, scale_factor=IMG_RESCALE_FACTOR)
(T_, Y_t) = imageLoader.load_images()
T = T_*(1.0/255.0)


###############################################################
#                       Neural network                        #
###############################################################

# Neural network constants
NN_EPSILON_INIT = 0.12
NN_LAMBDA = 0.1

# Build neural network
nn = NeuralNetwork((X.shape[1], X.shape[1]//2, len(np.unique(Y))),
                   epsilon_init=NN_EPSILON_INIT,
                   lambda_=NN_LAMBDA
                   )

# Untrained performance
P = nn.predict(T)
logging.info(f'Untrained performance: {np.sum(np.equal(P, Y_t))/Y_t.shape[0]}')

###############################
#    Train Neural Network     #
###############################

initial_J = nn.cost_function(X, Y)
logging.info(f'Initial J: {initial_J}')
J_list = [initial_J]  # List to hold costs
while True:
    initial_J = J_list[-1]
    nn.train(X, Y, 100)
    trained_J = nn.cost_function(X, Y)
    if initial_J == trained_J:
        break
    else:
        logging.info(f'New J: {trained_J}')
        J_list.append(trained_J)

logging.info(f'No improvement for J')
logging.info(f'Training J\'s: {J_list}')

###############################
#   Measure NN Performance    #
###############################

# Predict classes of test set
P = nn.predict(T)

logging.info(f'Trained performance: {np.sum(np.equal(P, Y_t))/Y_t.shape[0]}')

###############################
#    Export NN Parameters     #
###############################

print(nn.get_network_parameters_as_dict())
