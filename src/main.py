from typing import List
import numpy as np
from csv import DictReader
import logging

from imageLoader import ImageLoader
from neuralNetwork import NeuralNetwork

# Set up logging
logging.basicConfig(
    format='[%(asctime)s | %(levelname)s] %(message)s', level=logging.DEBUG)

# General constants
#classes = {'papier': 0, 'glas': 1, 'pmd': 2, 'restafval': 3}
classes = {'papier': 0, 'pmd': 1}

###############################################################
#                         Load images                         #
###############################################################

# Image constants
IMG_RESCALE_FACTOR = 20
DATA_PATHS = './data/path_list.csv'

# Load path_list from csv file
path_list = []
with open(DATA_PATHS, mode='r') as fd:
    dictReader = DictReader(fd)
    for dict in dictReader:
        path_list.append(dict)

# Load X & y from path_list
training_sets = {'cv0', 'cv1', 'cv2'}
imageLoader = ImageLoader(path_list, training_sets, classes,
                          scale_factor=IMG_RESCALE_FACTOR)
(X_, Y_x) = imageLoader.load_images()

# Normalize X
X = X_*(1.0/255.0)

# Load & normalize cross validation set
imageLoader = ImageLoader(
    path_list, {'cv3'}, classes, scale_factor=IMG_RESCALE_FACTOR)
(CV_, Y_cv) = imageLoader.load_images()
CV = CV_*(1.0/255.0)


###############################################################
#                       Neural network                        #
###############################################################

# Neural network constants
NN_EPSILON_INIT = 0.12
NN_LAMBDA = 1.0

# Build neural network
nn = NeuralNetwork((X.shape[1], X.shape[1]//2, len(np.unique(Y_x))),
                   epsilon_init=NN_EPSILON_INIT,
                   lambda_=NN_LAMBDA
                   )

# ONLY FOR DEBUGGING
# theta_list = nn.get_theta_list()
# thetas = nn._ravel_theta(theta_list)
# nn._cost_function(thetas, X, Y_x)


###############################
#    Train Neural Network     #
###############################

initial_P = np.sum(np.equal(nn.predict(CV), Y_cv))/Y_cv.shape[0]
logging.info(f'Initial cross validation accuracy: {initial_P}')
nn.train(X, Y_x, 5000)
trained_P = np.sum(np.equal(nn.predict(CV), Y_cv))/Y_cv.shape[0]
logging.info(f'Trained cross validation accuracy: {trained_P}')

# P_list = [initial_P]  # List to hold performances
# for _i in range(0, 3):
#     nn.train(X, Y_x, 500)
#     new_P = np.sum(np.equal(nn.predict(CV), Y_cv))/Y_cv.shape[0]
#     logging.info(f'New cross validation performance: {new_P}')
#     P_list.append(new_P)
#
# logging.info(f'No improvement for in cross validation performance')
# logging.info(f'Cross validation performance list: {P_list}')

###############################
#   Measure NN Performance    #
###############################

# Load & normalize test set
imageLoader = ImageLoader(
    path_list, {'test'}, classes, scale_factor=IMG_RESCALE_FACTOR)
(T_, Y_t) = imageLoader.load_images()
T = T_*(1.0/255.0)

# Predict classes of test set
P = nn.predict(T)
logging.info(f'Test accuracy: {np.sum(np.equal(P, Y_t))/Y_t.shape[0]}')

###############################
#    Export NN Parameters     #
###############################

print(nn.get_network_parameters_as_dict())
