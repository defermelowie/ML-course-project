import numpy as np
from typing import Dict, List, Tuple
import logging


class NeuralNetwork:
    # Data members
    lambda_: float
    layer_sizes: List[int]
    epsilon_init: float
    theta_list: List[np.ndarray]

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

    def cost_function(self) -> Tuple[float, np.ndarray]:
        """Calculate cost function

        Returns:
            (J, grad)

            J : float
            grad : ndarray
        """
        # TODO: Calculate cost
        pass

    def feed_forward(self, X: np.ndarray) -> np.ndarray:
        """Feed forward of an input X

        Returns
            a_n: ndarray
        """
        # Usefull vars
        set_size = X.shape[0]  # Number of examples

        # Calculate layers
        a_list = [X]
        for i, theta in enumerate(self.theta_list):
            # Add bias to layer
            a_list[i] = np.c_[a_list[i], np.ones(set_size)]
            logging.debug(f'a{i} with bias is of shape: {a_list[i].shape}')

            # Calculate next layer
            logging.debug(f'theta{i} is of shape: {theta.shape}')
            a = self._sigmoid(np.dot(a_list[i], theta))
            a_list.append(a)

        logging.debug(f'output layer is of shape: {a_list[-1].shape}')

        pass

    def get_network_parameters_as_dict(self) -> Dict:
        # TODO: Export network params as dict
        pass

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
