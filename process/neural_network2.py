"""
Neural Network fine tuned with cross entropy, mini batches and yet all
"""

import numpy as np, gzip, os, pathlib
from process import neural_network
import matplotlib.pyplot


def vectorize_y(y):
    """
    Create vectorized training output for a given training
    :return: vectorized training output
    """
    vector_base = np.zeros((len(y), 9))
    for output in range(len(y)):
        vector_base[output][y[output] - 1] = 1
    return vector_base.T


class Neural_2(neural_network.Neural):
    def __init__(self, training_data, no_of_neural_layers, no_of_training_set_members=60000, eta=0.25):
        """
        Initialize class with size as input.
        :param size: a list which contains no of neurons for each layer.So, len(size) will provide total
        no of layers in this neural schema, including input(which contains features or "X" values)
        and output layers.
        """
        self.size = no_of_neural_layers
        self.m = no_of_training_set_members
        self.cost_function = []
        self.success_rate = []
        self.eta = eta
        print("In derived class init")
        # random assignment of weights for each layer
        self.W = [np.random.randn(*z) for z in list(zip([x for x in self.size[1:]], [y for y in self.size[:-1]]))]
        # random assignment of bias for each layer
        self.B = [np.random.randn(x, 1) for x in self.size[1:]]
        # Open and populate training data into object instance variables.
        # training_datafile = gzip.open(training_data, mode="rb")
        # training_datafile_set = pickle.load(training_datafile)
        training_x, training_y = self._load_training_data(training_data)
        # training data file contains a tuple of training dataset and corresponding categories
        # The training data set is a single dimensional array which contains color RGB for
        # each of 60000 image, with each image being represented as an array of 784 pixels,
        # and these 784 pixels, in turn, refer to 28x28 pixels.
        training_data_transposed = training_x.T
        self.X = training_data_transposed[:50000].T  # X => (784, 60000)
        print("shape of X", np.shape(self.X))
        self.Validation_Data = training_data_transposed[50000:].T
        print("Shape of validation data", np.shape(self.Validation_Data))
        # Modify training results - y - to be a vector
        self.Y = vectorize_y(training_y[:50000])
        self.RAW_Y = training_y[:50000]
        print("Shape of Y", np.shape(self.Y))

        self.Validation_Y = vectorize_y(training_y[50000:])
        self.epochs = 10  # initialize epochs for the training model
        # Load training data
        test_data_file = gzip.open(os.path.join(training_data, "data", "t10k-images-idx3-ubyte.gz"), mode="rb")
        test_data_bytes = bytearray(10000 * 784 + 16)
        test_data_file.readinto(test_data_bytes)
        self.test_data = np.array(test_data_bytes)[16:].reshape(784, 10000)
        print("test data read bytes", np.shape(self.test_data))
        test_label_file = gzip.open(os.path.join(training_data, "data", "t10k-labels-idx1-ubyte.gz"), mode="rb")
        test_label_bytes = bytearray(10000 + 4)
        test_label_file.readinto(test_data_bytes)
        self.test_label = np.array(test_label_bytes)[4:]
        print("Shape of Test labels", np.shape(self.test_label))

    def _prepare_epoch__(self, x):
        """
        Prepare an epoch, but for a subset of training dataset
        :param x: a subset of training dataset, which is used as activation output for first neural layer(input layer)
        :return:
        """
        self.Z = []
        self.A = []
        self.A.append(x)
        self.L = 0  # initialize Loss function to be zero, for the entiretity of dataset.
        self.J = 0  # so initialize the cost function as well.

    def _process_feedforward(self, x):
        """
        Calculate output of neural layer for given input dataset x
        :param x: Dataset to evaluate the neural network for
        :return: Activations from output layer
        """
        a = x
        for layer in range(len(self.W)):
            z = np.dot(self.W[layer], a) + self.B[layer]
            a = 1 / (1 + np.exp(-z))
        return a

    def _calculate_loss__(self, y, batchsize=50000):
        """
        Calculate cost and loss function for the epoch, for the selected training subset
        :param y: output values for training
        :return:
        """
        self.L = -1 * (y * np.log(self.A[-1]) + (1 - y) * np.log(1 - self.A[-1]))
        self.J = np.sum(self.L) / batchsize

    def _backward_propagate__(self, x, y):
        """
        Update weights and biases for the epoch, using backward propagation.
        :return:
        """
        self._calculate_loss__(y)
        delta = self._moment_lossonoutput__(self.A[-1], y)
        db = np.sum(delta, axis=1, keepdims=True) / 50000
        dw = np.dot(delta, self.A[-2].T) / 50000
        self.W[-1] -= self.eta * dw
        self.B[-1] -= self.eta * db
        for layer in range(len(self.W) - 1, 0, -1):
            delta = np.dot(delta.T, self.W[layer]).T
            db_population = delta * self._moment_of_activation_function_on_weighted_output__(layer)
            dw = np.dot(db_population, self.A[layer - 1].T) / 50000
            self.W[layer - 1] -= self.eta * dw
            self.B[layer - 1] -= self.eta * np.sum(db_population, axis=1, keepdims=True) / 50000

    def _evaluate(self, a, y):
        """
        Calculate successful predictions of given dataset as a percentage
        :param a: predicted results for the validation dataset, this is typically a function of activated outputs of output neuron layer
        :param y: categorised results for the given dataset
        :return: percentage of successful results
        """
        # result = np.isclose(a.T, y.T, atol=0.1, rtol=0.01)
        # print("Shape of resuelt, ", np.shape(result), end = " ")
        # compress_result = np.sum(result.T, axis=0, keepdims=True)
        # eval = np.where(compress_result == 1, np.ones((1, len(a.T))), np.zeros((1, len(a.T))))
        # evaluation = np.count_nonzero(eval)
        a = np.array([np.isclose(np.argmax(x), np.argmax(y), atol=0.2, rtol=0.01) for x, y in zip(a.T, y.T)])
        r = np.count_nonzero(a)
        return r / len(a.T) * 100

    def train(self, epochs=10):
        """ This is the externally exposed class, which is just a wrapper
                on forward and backward propagation functions.
                epochs: No of epochs to train the data
            """
        self.epochs = epochs
        # initialize weighted output(Z) and activation function output for this epoch
        for i in range(self.epochs):
            print("Epoch ", i, end=" ")
            self._prepare_epoch__(self.X)
            self._propagate_forward__()
            self._prep_backward_propagation__()
            # self._backward_propagate_2__()
            self._backward_propagate__(self.A[-1], self.Y)
            print("J", round(self.J, 5), end=" ")
            rate = self._evaluate(self._process_feedforward(self.Validation_Data), self.Validation_Y)
            print("Success results =", round(rate, 2))
            self.cost_function.append(self.J)
            self.success_rate.append(rate)

        fig, (ax1, ax2) = matplotlib.pyplot.subplots(2, 1)
        ax1.plot(range(self.epochs), self.cost_function)
        ax2.plot(range(self.epochs), self.success_rate)
        matplotlib.pyplot.show()
