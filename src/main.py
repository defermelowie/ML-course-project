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
results = []

for cv_number, cv_set in enumerate(sets):
    logging.info(
        '---------------------------------------------------------------')

    # Create dir for this cross validation set
    cv_result_dir_path = f'{result_dir_path}/{cv_set}'
    mkdir(cv_result_dir_path)

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
    cv_accuracy_list = []
    train_accuracy_list = []
    iterations_list = []

    # Neural network constants
    NN_EPSILON_INIT = 0.12
    NN_LAMBDA = 0.75
    NN_TRAINING_ITERATIONS = 5000

    # Build neural network
    nn = NeuralNetwork((X.shape[1], X.shape[1]//2, len(np.unique(Y_x))),
                       epsilon_init=NN_EPSILON_INIT,
                       lambda_=NN_LAMBDA
                       )

    ###############################
    #    Train Neural Network     #
    ###############################
    initial_cv_accuracy = np.sum(
        np.equal(nn.predict(CV), Y_cv))/Y_cv.shape[0]
    logging.info(
        f'Initial cross validation accuracy: {initial_cv_accuracy}')

    initial_train_accuracy = np.sum(
        np.equal(nn.predict(X), Y_x))/Y_x.shape[0]
    logging.info(f'Initial training accuracy: {initial_train_accuracy}')

    iterations_list.append(0)
    cv_accuracy_list.append(initial_cv_accuracy)
    train_accuracy_list.append(initial_train_accuracy)

    iterations_per_step = 500
    for _i in range(0, NN_TRAINING_ITERATIONS//iterations_per_step):

        nn.train(X, Y_x, iterations_per_step)

        trained_cv_accuracy = np.sum(
            np.equal(nn.predict(CV), Y_cv))/Y_cv.shape[0]
        logging.info(f'Cross validation accuracy: {trained_cv_accuracy}')

        trained_train_accuracy = np.sum(
            np.equal(nn.predict(X), Y_x))/Y_x.shape[0]
        logging.info(f'Training accuracy: {trained_train_accuracy}')

        iterations_list.append((_i+1)*iterations_per_step)
        cv_accuracy_list.append(trained_cv_accuracy)
        train_accuracy_list.append(trained_train_accuracy)

    ###############################
    #    Export NN Parameters     #
    ###############################

    with open(f'{cv_result_dir_path}/model.json', 'w') as fd:
        model = {'model': nn.get_network_parameters_as_dict()}
        json.dump(model, fd)

    theta = nn.get_theta()
    np.save(f'{cv_result_dir_path}/theta.npy', theta)

    results.append({
        'cv': cv_number,
        'cv_accuracy': cv_accuracy_list,
        'train_accuracy': train_accuracy_list,
        'iterations': iterations_list
    })

###############################
#        Save results         #
###############################

with open(f'{result_dir_path}/results.json', 'w') as fd:
    results = {'results': results}
    json.dump(results, fd)

###############################
#       Plot CV results       #
###############################

fig, (train_plt, cv_plt) = plt.subplots(2, sharex=True)

train_plt.set(xlabel='Iterations', ylabel='Accuracy')
train_plt.grid(visible=True, which='both', axis='both')
train_plt.label_outer()

cv_plt.set(xlabel='Iterations', ylabel='Accuracy')
cv_plt.grid(visible=True, which='both', axis='both')
cv_plt.label_outer()

line_styles = ['-', '--', '-.', ':']

for i, result in enumerate(results):
    cv_plt.plot(
        result['iterations'],
        result['cv_accuracy'],
        f'k{line_styles[i]}',
        label=f'Cross validation {result["cv"]}'
    )
    train_plt.plot(
        result['iterations'],
        result['train_accuracy'],
        f'k{line_styles[i]}',
        label=f'Training {result["cv"]}'
    )

train_plt.legend()
cv_plt.legend()

fig.tight_layout()

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
