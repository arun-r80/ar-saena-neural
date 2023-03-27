"""
Neural Network fine-tuned with cross entropy, mini batches and yet all
"""

import numpy as np, gzip, os
from pathlib import Path
from process import neural_network2
import pickle
import matplotlib.pyplot
import random


def vectorize_y(y):
    """
    Create vectorized training output for a given training
    :return: vectorized training output
    """
    vector_base = np.zeros((len(y), 10))
    for output in range(len(y)):
        vector_base[output][y[output]] = 1
    return vector_base.T


def devectorize(a):
    """
    Provide mapping for activation vector of final layer, ie., the actual category this vector maps to in the real mumber space.
    This is mothing but argmax for the numpy vector.
    :param x: Un-transposed activation vector
    :return:
    """
    a_t = a.T
    return [np.argmax(x) for x in a_t]


class neural_2_1(neural_network2.Neural_2):
    def __init__(self, training_data, no_of_neural_layers, no_of_training_set_members=50000,
                 no_of_validation_data_members=10000, eta=0.25, l_regularize=0.15, m=9000):
        """
        Initialize class with size as input.
        :param no_of_neural_layers: a list which contains no of neurons for each layer.So, len(size) will provide total
        no of layers in this neural schema, including input(which contains features or "X" values)
        and output layers.
        """
        # super().__init__(training_data, no_of_neural_layers, no_of_training_set_members, eta=eta)
        super().__init__(training_data, no_of_neural_layers, no_of_training_set_members=no_of_training_set_members,
                         no_of_validation_data_members=no_of_validation_data_members, eta=eta,
                         l_regularize=l_regularize, m=m)
        self.length_validation_data = no_of_validation_data_members
        # self.m = no_of_training_set_members
        # self.v = no_of_validation_data_members
        # self.cost_function = []
        # self.success_rate = []
        # self.eta = eta
        # self.lmbda = l_regularize
        # self.batch_size = m
        # print("In derived class init")
        # print("m is", self.m)
        # # random assignment of weights for each layer
        # self.W = [np.random.randn(x, y) for x, y in zip(self.size[1:], self.size[:-1])]
        # # random assignment of bias for each layer
        # self.B = [np.random.randn(x, 1) for x in self.size[1:]]
        # # Open and populate training data into object instance variables.
        # # training_datafile = gzip.open(training_data, mode="rb")
        # # training_datafile_set = pickle.load(training_datafile)
        # training_x, training_y = self._load_training_data(training_data)
        # # training data file contains a tuple of training dataset and corresponding categories
        # # The training data set is a single dimensional array which contains color RGB for
        # # each of 60000 image, with each image being represented as an array of 784 pixels,
        # # and these 784 pixels, in turn, refer to 28x28 pixels.
        # training_data_transposed = training_x.T
        # self.X = training_data_transposed[:no_of_training_set_members].T  # X => (784, 60000)
        # self.Validation_Data = training_data_transposed[-no_of_validation_data_members:].T # Validation_Data => (784, 10000)
        # print("In Init")
        # # Modify training results - y - to be a vector
        # self.Y = vectorize_y(training_y[:no_of_training_set_members])
        # self.RAW_Y = training_y[:no_of_training_set_members]
        # self.Validation_Y = vectorize_y(training_y[-no_of_validation_data_members:])
        # self.epochs = 10  # initialize epochs for the training model
        # # Load training data
        # test_data_file = gzip.open(os.path.join(training_data, "data", "t10k-images-idx3-ubyte.gz"), mode="rb")
        # test_data_bytes = bytearray(10000 * 784 + 16)
        # test_data_file.readinto(test_data_bytes)
        # self.test_data = np.array(test_data_bytes)[16:].reshape(784, 10000)
        # test_label_file = gzip.open(os.path.join(training_data, "data", "t10k-labels-idx1-ubyte.gz"), mode="rb")
        # test_label_bytes = bytearray(10000 + 4)
        # test_label_file.readinto(test_data_bytes)
        # self.test_label = np.array(test_label_bytes)[4:]

    def _calculate_loss__(self, a, y, lmbda, batchsize=50000):
        """
        Calculate cost and loss function for the epoch, for the selected training subset
        :param y: output values for training
        :return:
        """
        l_across_outputneurons = self._loss_fn_(a, y)
        regularized_cost = np.sum(l_across_outputneurons, axis=0, keepdims=True)  # => (n[l],self.m)
        self.J = np.sum(regularized_cost) / batchsize + 0.5*(lmbda/batchsize)*sum(
            np.linalg.norm(w)**2 for w in self.W)

    def _evaluate(self, a, y, count):
        """
        Calculate successful predictions of given dataset as a percentage
        :param a: predicted results for the validation dataset, this is typically a function of activated outputs of output neuron layer
        :param y: categorised results for the given dataset
        :return: percentage of successful results
        """
        return (np.count_nonzero(a == y) / count) * 100

    def train(self, epochs=10):
        """ This is the externally exposed class, which is just a wrapper
                on forward and backward propagation functions.
                epochs: No of epochs to train the data
            """
        self.epochs = epochs
        x_transposed = self.X.T
        y_transposed = self.Y.T
        training_data_transposed = list(zip(x_transposed, y_transposed))
        # initialize weighted output(Z) and activation function output for this epoch
        for i in range(self.epochs):
            print("Epoch ", i, end=" ")
            random.shuffle(training_data_transposed)
            training_shuffled_minibatches = [training_data_transposed[k:k + self.batch_size] for k in
                                             range(0, self.m, self.batch_size)]

            for minibatch in training_shuffled_minibatches:
                x = np.array([a[0] for a in minibatch]).T
                y = np.array([a[1] for a in minibatch]).T
                self._prepare_epoch__(x)
                self._propagate_forward__()
                self._prep_backward_propagation__()
                # self._backward_propagate_2__()
                self._backward_propagate__(self.A[-1], y)
            self._calculate_loss__(self._process_feedforward(self.X), self.Y, self.lmbda, batchsize=self.m)
            print("J", round(self.J, 5), end=" ")
            rate = self._evaluate(devectorize(self._process_feedforward(self.Validation_Data)), self.RAW_VALIDATION_Y, self.length_validation_data)
            print("Success results =", round(rate, 2), "%")
            self.cost_function.append(self.J)
            self.success_rate.append(rate)
        # pickle the file for plotting in JupyterHub
        a = (self.cost_function, self.success_rate)
        results = open(os.path.join(Path.cwd(), "data", "plot.pkl"), mode="wb")
        pickle.dump(a, results)

        fig, (ax1, ax2) = matplotlib.pyplot.subplots(2, 1)
        ax1.plot(range(self.epochs), self.cost_function, "g")
        matplotlib.pyplot.title("Cost Function")
        matplotlib.pyplot.xlabel("Epochs")
        matplotlib.pyplot.ylabel("Cost")
        ax2.plot(range(self.epochs), self.success_rate, "r")
        matplotlib.pyplot.title("Success Rate")
        matplotlib.pyplot.xlabel("Epochs")
        matplotlib.pyplot.ylabel("Success Rate(%)")
        matplotlib.pyplot.show()
