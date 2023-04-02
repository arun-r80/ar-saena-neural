"""
A modification to neural network, to use L1 regularization.
Prepared in response to Chapter 3, Section "Handwritten recogniztion revisited: the code", problem 1
"""

from process import neural_network2_3 as neural
import numpy as np


class neural_2_4(neural.neural_2_3):
    def __init__(self, training_data, no_of_neural_layers, no_of_training_set_members=50000,
                 no_of_validation_data_members=10000, eta=0.25, l_regularize=0.15, m=9000):
        """Initialise neural network - This initialisation does not add anything new to super class.
        """
        super().__init__(training_data, no_of_neural_layers,
                         no_of_training_set_members=no_of_training_set_members,
                         no_of_validation_data_members=no_of_validation_data_members,
                         eta=eta,
                         l_regularize=l_regularize, m=m)

    def _regularize_w(self, eta, lmbda, m, batch_size, w_network, nabla_w):
        """
        Employ L1 regularization
        :param eta: learning rate
        :param lmbda: regularization parameter
        :param m: training set size
        :param batch_size: batch size
        :param w_network: weights for the network
        :param nabla_w: difference in weights
        :return: regularized weight based on hyper-parameters and regularization logic
        """
        sgn_w = [w/np.abs(w) for w in w_network]

        return [w - (eta * (lmbda / m)) * sgnw - (eta / batch_size) * nw for w, nw, sgnw in
                zip(w_network, nabla_w, sgn_w)]
