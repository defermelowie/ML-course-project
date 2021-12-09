from os import mkdir
from datetime import datetime
from typing import List
import json
import numpy as np
from csv import DictReader
import logging
import matplotlib.pyplot as plt

from imageLoader import ImageLoader
from neuralNetwork import NeuralNetwork

# Set up logging
result_dir_path = f'./results/{datetime.now().strftime("%d-%m-%Y_%Hu%M")}'
mkdir(result_dir_path)
logging.basicConfig(
    format='[%(asctime)s | %(levelname)s] %(message)s',
    level=logging.INFO,
    filename=f'{result_dir_path}/main.log',
    filemode='w'
)

# General constants
classes = {'papier': 0, 'glas': 1, 'pmd': 2, 'restafval': 3}
#classes = {'papier': 0, 'pmd': 1}

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

sets = {'cv0', 'cv1', 'cv2', 'cv3'}
CV_accuracies = []

for cv_set in sets:
    logging.info(
        '---------------------------------------------------------------')

    # Define training set
    training_sets = sets.copy()
    training_sets.remove(cv_set)

    # Load & normalize training sets
    imageLoader = ImageLoader(path_list, training_sets, classes,
                              scale_factor=IMG_RESCALE_FACTOR)
    (X_, Y_x) = imageLoader.load_images()
    X = X_*(1.0/255.0)

    # Load & normalize cross validation set
    imageLoader = ImageLoader(
        path_list, cv_set, classes, scale_factor=IMG_RESCALE_FACTOR)
    (CV_, Y_cv) = imageLoader.load_images()
    CV = CV_*(1.0/255.0)

    ###############################################################
    #                       Neural network                        #
    ###############################################################

    # Neural network constants
    NN_EPSILON_INIT = 0.12
    NN_LAMBDA = 0.5

    # Build neural network
    nn = NeuralNetwork((X.shape[1], X.shape[1]//2, len(np.unique(Y_x))),
                       epsilon_init=NN_EPSILON_INIT,
                       lambda_=NN_LAMBDA
                       )

    ###############################
    #    Train Neural Network     #
    ###############################
    initial_cv_accuracy = np.sum(np.equal(nn.predict(CV), Y_cv))/Y_cv.shape[0]
    logging.info(f'Initial cross validation accuracy: {initial_cv_accuracy}')

    initial_train_accuracy = np.sum(np.equal(nn.predict(X), Y_x))/Y_x.shape[0]
    logging.info(f'Initial training accuracy: {initial_train_accuracy}')

    cv_accuracy_list = [initial_cv_accuracy]
    train_accuracy_list = [initial_train_accuracy]
    iteration_list = [0]

    train_step_iter_count = 200
    train_steps = 15
    for i in range(1, train_steps+1):
        nn.train(X, Y_x, train_step_iter_count)

        trained_cv_accuracy = np.sum(
            np.equal(nn.predict(CV), Y_cv))/Y_cv.shape[0]
        logging.info(f'Cross validation accuracy: {trained_cv_accuracy}')

        trained_train_accuracy = np.sum(
            np.equal(nn.predict(X), Y_x))/Y_x.shape[0]
        logging.info(f'Training accuracy: {trained_train_accuracy}')

        cv_accuracy_list.append(trained_cv_accuracy)
        train_accuracy_list.append(trained_train_accuracy)
        iteration_list.append(i*train_step_iter_count)

    CV_accuracies.append({
        'cv': cv_set,
        'cv_accuracy': cv_accuracy_list,
        'train_accuracy': train_accuracy_list,
        'iterations': iteration_list
    })

    ###############################
    #    Export NN Parameters     #
    ###############################

    with open(f'{result_dir_path}/model_{cv_set}.json', 'w') as fd:
        model = {'model': nn.get_network_parameters_as_dict()}
        json.dump(model, fd)

    theta = nn.get_theta()
    np.save(f'{result_dir_path}/theta_{cv_set}.npy', theta)

###############################
#       Plot CV results       #
###############################

line_styles = ['-', '--', '-.', ':']

for i, accuracy in enumerate(CV_accuracies):
    plt.plot(
        accuracy['iterations'],
        accuracy['cv_accuracy'],
        f'k{line_styles[i]}',
        label=f'cross validation accuracy ({accuracy["cv"]})'
    )
    plt.plot(
        accuracy['iterations'],
        accuracy['train_accuracy'],
        f'b{line_styles[i]}',
        label=f'training accuracy ({accuracy["cv"]})'
    )

plt.legend()
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.grid(visible=True, which='both', axis='both')
plt.savefig(f'{result_dir_path}/accuracy_iterations.pdf')
# plt.show()

###############################
#   Measure NN Performance    #
###############################

# Load & normalize test set
# imageLoader = ImageLoader(
#     path_list, {'test'}, classes, scale_factor=IMG_RESCALE_FACTOR)
# (T_, Y_t) = imageLoader.load_images()
# T = T_*(1.0/255.0)
#
# # Predict classes of test set
# P = nn.predict(T)
# logging.info(f'Test accuracy: {np.sum(np.equal(P, Y_t))/Y_t.shape[0]}')
