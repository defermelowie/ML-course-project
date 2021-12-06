import numpy as np
from typing import Dict, List, Tuple
import logging
from numpy.lib.function_base import gradient
from scipy import optimize


class NeuralNetwork:
    # Data members
    lambda_: float
    layer_sizes: List[int]
    epsilon_init: float
    theta_list: List[np.ndarray]
    training_iterations: int = 0

    def __init__(self, model_layer_sizes: Tuple[int], lambda_: float = 0.0, epsilon_init: float = 0.12):
        """Create new neural network instance"""
        # Save parameters
        self.lambda_ = lambda_
        self.epsilon_init = epsilon_init
        self.layer_sizes = model_layer_sizes

        # Create randomized theta list
        layer_count = len(model_layer_sizes)
        self.theta_list = [np.random.rand(model_layer_sizes[i] + 1, model_layer_sizes[i+1])
                           * 2 * self.epsilon_init - self.epsilon_init for i in range(layer_count-1)]
        logging.debug(
            f'Thetas are of shape: {[theta.shape for theta in self.theta_list]}')

        # Logging
        logging.info(
            f'Neural network with shape {model_layer_sizes} was created')

    def cost_function(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Get cost function for model theta's"""
        thetas = np.concatenate([theta.ravel() for theta in self.theta_list])
        (J, _gradient) = self._cost_function(thetas, X, Y)
        return J

    def _cost_function(self, thetas: np.ndarray, X: np.ndarray, Y: np.ndarray) -> Tuple[float, np.ndarray]:
        """Calculate cost function using X {shape: (set_size, input_layer_size)} and Y {shape: (set_size,)}

        Returns:
            (J, grad)

            J : float
            grad : 1d array
        """
        # Reshape thetas back into theta_list
        theta_list = self._unravel_theta(thetas)

        # Feed forward the network and calculate layers
        a_layers = self.feed_forward(X, theta_list)

        # Create usefull vars
        exit_layer = a_layers[-1]  # Get exit layer
        y_matrix = (np.eye(exit_layer.shape[1])[Y])  # Make matrix of Y
        set_size = X.shape[0]

        # Calculate cost function
        #######################################################

        # Calculate cumulative cost
        j = np.sum(y_matrix*np.log(exit_layer) +
                   (1-y_matrix)*np.log(1-exit_layer))

        # Calculate regularization term
        reg_term = (self.lambda_ / (2 * set_size)) * (np.sum([np.sum(np.square(theta[1:, :]))
                                                              for theta in theta_list]))

        # Calculate actual cost function
        J = -j/set_size + reg_term

        # Calculate gradients
        #######################################################

        # Calculate delta's
        delta_list = []  # Empty list to hold delta's
        delta_list.append(exit_layer - y_matrix)
        logging.debug(f'delta2 is of shape {delta_list[0].shape}')
        # TODO: Generalize for other NN architectures
        delta1 = np.dot(delta_list[0], theta_list[1].T)[:, 1:] * \
            self._sigmoid_gradient(np.dot(a_layers[0], theta_list[0]))
        delta_list.append(delta1)
        logging.debug(f'delta1 is of shape {delta1.shape}')

        Delta_list = [np.dot(delta.T, a_layers[i])
                      for i, delta in enumerate(reversed(delta_list))]
        logging.debug(
            f'Deltas are of shape {[delta.shape for delta in Delta_list]}')

        # Calculate gradient list
        gradient_list = [delta/set_size for delta in Delta_list]
        for Theta in gradient_list:
            Theta[:, 1:] += (self.lambda_/set_size) * Theta[:, 1:]

        logging.debug(
            f'Theta gradients are of shape {[grad.shape for grad in gradient_list]}')

        # Convert gradients to 1 dimensional array
        gradients = np.concatenate([grad.ravel() for grad in gradient_list])

        # Return cost and gradients
        return (J, gradients)

    def feed_forward(self, X: np.ndarray, theta_list: List[np.ndarray]) -> List[np.ndarray]:
        """Feed forward of an input X of shape: (set_size, input_layer_size)

        Returns
            a_list: ndarray
        """
        # Usefull vars
        set_size = X.shape[0]  # Number of examples

        # Calculate layers
        a_list = [X]
        for i, theta in enumerate(theta_list):
            # Add bias to layer
            a_list[i] = np.c_[a_list[i], np.ones(set_size)]
            logging.debug(f'a{i} with bias is of shape: {a_list[i].shape}')

            # Calculate next layer
            logging.debug(f'theta{i} is of shape: {theta.shape}')
            a = self._sigmoid(np.dot(a_list[i], theta))
            a_list.append(a)

        logging.debug(f'output layer is of shape: {a_list[-1].shape}')

        return a_list  # Return all layers

    def predict(self, X: np.ndarray) -> np.ndarray:
        layers = self.feed_forward(X, self.theta_list)
        return np.argmax(layers[-1], axis=1)

    def get_network_parameters_as_dict(self) -> Dict:
        """Export network params as dict"""
        # Initialize empty dictionary
        params = {}

        # Model parameters
        layer_sizes = {}
        for i, layer_size in enumerate(self.layer_sizes):
            layer_sizes[f'layer_{i}'] = layer_size
        params['layer_sizes'] = layer_sizes

        # Learning parameters
        params['lambda'] = self.lambda_
        params['training_iterations'] = self.training_iterations

        return params

    def train(self, X: np.ndarray, Y: np.ndarray, max_iterations: int = 100):
        # options
        options = {'maxiter': max_iterations}

        # Initial thetas
        theta_init = self._ravel_theta(self.theta_list)

        # Create "short hand" for the cost function to be minimized
        def costFunction(thetas): return self._cost_function(thetas, X, Y)

        # Logging
        logging.info(f'Optimize with max {max_iterations} iterations ...')

        # Minimize cost function
        res = optimize.minimize(
            costFunction,
            theta_init,
            jac=True,
            method='TNC',
            options=options
        )

        # Update model with solution of the optimization
        self.theta_list = self._unravel_theta(res.x)

        # Update training_iterations counter
        self.training_iterations += max_iterations

    def _ravel_theta(self, theta_list: List[np.ndarray]) -> np.ndarray:
        return np.concatenate([theta.ravel() for theta in theta_list])

    def _unravel_theta(self, theta_raveled: np.ndarray) -> List[np.ndarray]:
        # First theta
        theta_list = [np.reshape(
            theta_raveled[:self.layer_sizes[1] * (self.layer_sizes[0] + 1)],
            (self.layer_sizes[0] + 1, self.layer_sizes[1])
        )]

        # TODO: Calculate middel theta's

        # Last theta
        theta_list.append(np.reshape(
            theta_raveled[self.layer_sizes[-2] * (self.layer_sizes[-3] + 1):],
            (self.layer_sizes[-2] + 1, self.layer_sizes[-1])
        ))

        return theta_list

    def _get_layer_size(self, layer: int, bias: bool) -> int:
        if (not bias or layer == len(self.layer_sizes)):  # Without bias or if last layer
            return self.layer_sizes[layer]
        else:
            return self.layer_sizes[layer] + 1

    def _sigmoid(self, z):
        """
        Computes the sigmoid of z.
        """
        return 1.0 / (1.0 + np.exp(-z))

    def _sigmoid_gradient(self, z: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the sigmoid function evaluated at z.
        """
        return self._sigmoid(z) * (1 - self._sigmoid(z))
