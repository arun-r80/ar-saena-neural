"""
Neural Network fine-tuned with cross entropy, mini batches and yet all
"""

import numpy as np, gzip, os
from pathlib import Path
from process import neural_network
import pickle
import matplotlib.pyplot


def vectorize_y(y):
    """
    Create vectorized training output for a given training
    :return: vectorized training output
    """
    vector_base = np.zeros((len(y), 10))
    for output in range(len(y)):
        vector_base[output][y[output]] = 1
    return vector_base.T


class Neural_2(neural_network.Neural):
    def __init__(self, training_data, no_of_neural_layers,  no_of_training_set_members=50000,
                 no_of_validation_data_members=10000, eta=0.25, l_regularize=0.15, m=9000):
        """
        Initialize class with size as input.
        :param no_of_neural_layers: a list which contains no of neurons for each layer.So, len(size) will provide total
        no of layers in this neural schema, including input(which contains features or "X" values)
        and output layers.
        """
        super().__init__(training_data, no_of_neural_layers, no_of_training_set_members, eta=eta)
        self.m = no_of_training_set_members
        self.v = no_of_validation_data_members
        self.cost_function = []
        self.success_rate = []
        self.eta = eta
        self.lmbda = l_regularize
        self.batch_size = m
        print("In derived class init")
        print("m is", self.m)
        # random assignment of weights for each layer
        self.W = [np.random.randn(x, y) for x, y in zip(self.size[1:], self.size[:-1])]
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
        self.X = training_data_transposed[:no_of_training_set_members].T  # X => (784, 60000)
        self.Validation_Data = training_data_transposed[-no_of_validation_data_members:].T # Validation_Data => (784, 10000)
        print("In Init")
        # Modify training results - y - to be a vector
        self.Y = vectorize_y(training_y[:no_of_training_set_members])
        self.RAW_Y = training_y[:no_of_training_set_members]
        self.RAW_VALIDATION_Y = training_y[-no_of_validation_data_members:]
        self.Validation_Y = vectorize_y(self.RAW_VALIDATION_Y)
        self.epochs = 10  # initialize epochs for the training model
        # Load training data
        test_data_file = gzip.open(os.path.join(training_data, "data", "t10k-images-idx3-ubyte.gz"), mode="rb")
        test_data_bytes = bytearray(10000 * 784 + 16)
        test_data_file.readinto(test_data_bytes)
        self.test_data = np.array(test_data_bytes)[16:].reshape(784, 10000)
        test_label_file = gzip.open(os.path.join(training_data, "data", "t10k-labels-idx1-ubyte.gz"), mode="rb")
        test_label_bytes = bytearray(10000 + 4)
        test_label_file.readinto(test_data_bytes)
        self.test_label = np.array(test_label_bytes)[4:]

    def _prepare_epoch__(self, x):
        """
        Prepare an epoch, but for a subset of training dataset
        :param x: a subset of training dataset, which is used as activation output for first neural layer(input layer)
        :return:
        """
        self.Z = []
        self.A = []
        self.A.append(x)
        self.L = 0  # initialize Loss function to be zero, for the entirety of dataset.
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

    def _loss_fn_(self, a, y):
        """
        Calculate the loss function as a vector, for given activated outputs from the neural network,
        and corresponding sample size results y.
        This function uses Cross Entropy function as the basis to calculate loss.
        :param a: activated outputs from neural network( activation functions from last layer)
        :param y: expected results for the sample size
        :return: cross entropy based loss function as a vector
        """
        return  np.nan_to_num((-y * np.log(a) - (1 - y) * np.log(1 - a)))

    def _calculate_loss__(self,a, y, lmbda, batchsize=50000):
        """
        Calculate cost and loss function for the epoch, for the selected training subset
        :param y: output values for training
        :return:
        """
        print("Shape of a ang y in calculate loss", np.shape(a), np.shape(y))
        l_across_outputneurons = self._loss_fn_(a, y)
        regularized_cost = np.sum(l_across_outputneurons, axis=0,keepdims=True) # => (n[l],self.m)
        l_across_samplesize = regularized_cost + (lmbda * 0.5) * sum(np.linalg.norm(w)**2 for w in self.W)
        self.J = np.sum(l_across_samplesize) / batchsize


    def _backward_propagate__(self, a, y):
        """
        Update weights and biases for the epoch, using backward propagation.
        :return:
        """
        delta = self._moment_lossonoutput__(a, y) # delta => n[l],b where b is batch size
        db = np.sum(delta, axis=1, keepdims=True) # db => n[l],1
        dw = np.dot(delta, self.A[-2].T) # delta => n[l], b; A[-2].T => b,n[l-1]; dw => n[l], n[l-2]
        self.W[-1] = self.W[-1]*(1 - self.eta * (self.lmbda/self.m)) - (self.eta/self.batch_size) * dw
        self.B[-1] -= (self.eta/self.batch_size) * db
        for layer in range(len(self.W) - 1, 0, -1):
            delta = np.dot(self.W[layer].T, delta) * self._moment_of_activation_function_on_weighted_output__(layer)
            # np.dot(W[layer].T => (n[l-1],n[l], delta => (n[l],b)) => n[l-1],b
            #delta = n[l-1],b
            db_population = delta
            dw = np.dot(db_population, self.A[layer - 1].T)
            self.W[layer - 1] = self.W[layer - 1] * (1 - self.eta * (self.lmbda/self.m)) - (self.eta/self.batch_size) * dw
            self.B[layer - 1] -= (self.eta/self.batch_size) * np.sum(db_population, axis=1, keepdims=True)

    def _evaluate(self, a, y):
        """
        Calculate successful predictions of given dataset as a percentage
        :param a: predicted results for the validation dataset, this is typically a function of activated outputs of output neuron layer
        :param y: categorised results for the given dataset
        :return: percentage of successful results
        """

        b = np.array([np.isclose(np.argmax(x), np.argmax(y), atol=0.3, rtol=0.01) for x, y in zip(a.T, y.T)])
        print(y.T[2000], a.T[2000], np.argmax(y.T[2000]), np.argmax(a.T[2000]), b[2000], round(y.T[2000][np.argmax(y.T[2000])] - a.T[2000][np.argmax(a.T[2000])],2))
        r = np.count_nonzero(b)
        return r / len(a.T) * 100

    def train(self, epochs=10):
        """ This is the externally exposed class, which is just a wrapper
                on forward and backward propagation functions.
                epochs: No of epochs to train the data
            """
        self.epochs = epochs
        x_transpose = self.X.T
        y_transpose = self.Y.T
        # initialize weighted output(Z) and activation function output for this epoch
        for i in range(self.epochs):
            print("Epoch ", i, end=" ")
            batch_index = np.random.randint(0, self.m - self.batch_size, 1, dtype=int)[0]
            x = x_transpose[batch_index: batch_index + self.batch_size].T
            y = y_transpose[batch_index: batch_index + self.batch_size].T
            self._prepare_epoch__(x)
            self._propagate_forward__()
            self._prep_backward_propagation__()
            # self._backward_propagate_2__()
            self._calculate_loss__(self.A[-1], y, self.lmbda, batchsize=self.m)
            self._backward_propagate__(self.A[-1], y)
            print("J", round(self.J, 5), end=" ")
            rate = self._evaluate(self._process_feedforward(self.Validation_Data), self.Validation_Y)
            print("Success results =", round(rate, 2))
            self.cost_function.append(self.J)
            self.success_rate.append(rate)
        # pickle the file for plotting in JupyterHub
        a = (self.cost_function, self.success_rate)
        results = open(os.path.join(Path.cwd(), "data", "plot.pkl"), mode="wb")
        pickle.dump(a, results)

        fig, (ax1, ax2) = matplotlib.pyplot.subplots(2, 1)
        ax1.plot(range(self.epochs-3), self.cost_function[3:], "g")
        matplotlib.pyplot.title("Cost Function")
        matplotlib.pyplot.xlabel("Epochs")
        matplotlib.pyplot.ylabel("Cost")
        ax2.plot(range(self.epochs-3), self.success_rate[3:], "r")
        matplotlib.pyplot.title("Success Rate")
        matplotlib.pyplot.xlabel("Epochs")
        matplotlib.pyplot.ylabel("Success Rate(%)")
        matplotlib.pyplot.show()
