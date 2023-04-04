"""
This class uses dataset used by Michael Nielsen, to explore if the change in success percentage is due to a different dataset
"""
import pickle

from process import neural_network_2_2 as neural
import numpy as np, os, pathlib, gzip
from process import neural_network2
import random
from process import neural_network2, neural_network_2_1


def load_data_wrapper(tr_d, va_d, te_d):
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.

    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.

    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.

    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return (training_data, validation_data, test_data)


def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


class neural_2_3(neural.neural_2_2):
    """
    This extension class has essentially the same functionality as other Neural classes,
    but the dataset to be used will be different.
    """

    def __init__(self, training_data, no_of_neural_layers, no_of_training_set_members=50000,
                 no_of_validation_data_members=10000, eta=0.25, l_regularize=0.15, m=9000):
        """
        A Modification of init function to use different dataset
        """
        # load training, validation and test data
        mnist_data = gzip.open(os.path.join(training_data, "data", "mnist.pkl.gz"), mode="rb")
        self.training_data_raw, self.validation_data_raw, self.test_data_raw = pickle.load(mnist_data,
                                                                                           encoding="latin1")
        self.training_data, self.v_data, self.test_data = load_data_wrapper(self.training_data_raw,
                                                                            self.validation_data_raw,
                                                                            self.test_data_raw)
        print("Shape of X in self.training data", np.shape(self.training_data_raw[0][0]))
        self.m = no_of_training_set_members
        self.size = no_of_neural_layers
        self.v = no_of_validation_data_members
        self.cost_function = []
        self.success_rate = []
        self.eta = eta
        self.lmbda = l_regularize
        self.batch_size = m
        self.length_validation_data = no_of_validation_data_members
        self.X = self.training_data_raw[0].T
        self.Y = neural_network2.vectorize_y(self.training_data_raw[1])
        self.Validation_Data = self.validation_data_raw[0].T
        self.RAW_VALIDATION_Y = self.validation_data_raw[1]

        print("Shape of X", np.shape(self.X),
              "Shape of Y", np.shape(self.Y),
              "Shape of Validation Data", np.shape(self.Validation_Data),
              "Shape of Raw Validation Data", np.shape(self.RAW_VALIDATION_Y),
              "Shape of Test Data", np.shape(self.test_data_raw[0].T),
              "Shape of y in training data", np.shape(self.training_data[0][1]))

        # random assignment of weights for each layer
        self.W = [np.random.randn(x, y) for x, y in zip(self.size[1:], self.size[:-1])]
        # random assignment of bias for each layer
        self.B = [np.random.randn(x, 1) for x in self.size[1:]]
        # Open and populate training data into object instance variables.
        # training_datafile = gzip.open(training_data, mode="rb")
        # training_datafile_set = pickle.load(training_datafile)

    def train(self, epochs=10):
        """ This is the externally exposed class, which is just a wrapper
                on forward and backward propagation functions.
                epochs: No of epochs to train the data
            """
        self.epochs = epochs
        self.cost_function = []
        self.success_rate = []
        for e in range(epochs):
            print("Epoch ", e, end=" ")
            random.shuffle(self.training_data)
            minibatches = [self.training_data[k: k + self.batch_size] for k in range(0, self.m, self.batch_size)]

            for minibatch in minibatches:
                self._run_minibatch(minibatch)
            self._calculate_loss__(self._process_feedforward(self.X), self.Y, self.lmbda, batchsize=50000)
            self.cost_function.append(self.J)
            rate = self._evaluate(neural_network_2_1.devectorize(self._process_feedforward(self.test_data_raw[0].T)),
                                  self.test_data_raw[1], self.length_validation_data)
            self.success_rate.append(rate)
            print("Cost = ", round(self.J, 5), "Success rate %f %", rate)
