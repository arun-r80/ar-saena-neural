"""
This is the module which creates a class to
train - ie., to forward and backpropagate on training data.
"""

import numpy as np
import os, \
    matplotlib.pyplot, \
    gzip \



class Neural:
    """
    This is the core class that does forward and backward propagation
    This class defines initialization function, forward and backward propagation functions.
    The following notation guide will help understand the logic
    W,B(w,b) = represents weights and biases respectively in a neuron
    Z = the weighted output of a neuron, ie.,  = W * X + B
    A, a = Activation output of a layer.
    m = total no of training data points

    """

    def __init__(self, training_data, no_of_neyral_layers, no_of_training_set_members=60000, eta=0.0001):
        """
        Initialize class with size as input.
        :param size: a list which contains no of neurons for each layer.So, len(size) will provide total
        no of layers in this neural schema, including input(which contains features or "X" values)
        and output layers.
        """
        self.size = no_of_neyral_layers
        self.m = no_of_training_set_members
        self.cost_function = []
        self.success_rate = []
        self.eta = eta
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
        self.X = training_x  # X => (784, 60000)
        self.Y = training_y  # Y => (60000,1)
        self.epochs = 10  # initialize epochs for the training model

    def _load_training_data(self, projectrootfolder):
        """
        This function wraps the process of training data extraction, as an API,
        to abstract different methods and sources for data extraction
        :param projectrootfolder:
        :return: a tuple containing training data and training label
        """
        training_image_file = gzip.open(os.path.join(projectrootfolder, "data", "train-images-idx3-ubyte.gz"),
                                        mode="rb")
        training_image_bytes = bytearray(self.m * 784 + 16)
        training_image_bytes_read = training_image_file.readinto(training_image_bytes)
        training_image_set = np.array(training_image_bytes[16:]).reshape((60000, 784))
        # extract training labels
        training_label_file = gzip.open(os.path.join(projectrootfolder, "data", "train-labels-idx1-ubyte.gz"),
                                        mode="rb")
        training_label = bytearray(self.m + 8)
        training_label_read = training_label_file.readinto(training_label)
        training_label_set = np.array(training_label[8:])
        return (training_image_set.T, training_label_set)

    def _moment_lossOnOutput__(self, output, y):
        """
        This function returns the first moment of loss function(L) on classifier output(a)
        when the Loss function is chosen as the standard deviation, ie:
        L = 1/2 * (a - Y)^2
        :param OutPut: The output of neural network a
        :param Y: Test data labels corresponding to the input data
        :return: The first derivative of loss function on the output value
        """

        return output - y

    def _moment_of_activation_function_on_weighted_output__(self, layer=None):
        if layer == 0:
            return self.A[layer]
        k = np.multiply(self.A[layer], (1 - self.A[layer]))

        return np.multiply(self.A[layer], (1 - self.A[layer]))

    def _prepare_epoch__(self):
        self.Z = []
        self.A = []
        self.A.append(self.X)
        self.L = 0  # initialize Loss function to be zero, for the entiretity of dataset.
        self.J = 0  # so initialize the cost function as well.

    def _propagate_forward__(self):
        """
        This function does the forward propagation, which entails following the below steps
        in the order given:
        a. Calculate weighted output(Z) for each layer, for all the input dataset.
        b. Calculate activation function output(A) for each layer, and again for each and every datapoint in training dataset
        c. Calculate loss function, based on activated output above, for all datapoints in dataset.
        :return:
        """
        for i in range(len(self.W)):
            z_next = np.dot(self.W[i], self.A[i]) + self.B[i]
            # W[i]      => (size(i), size(i-1))
            # A[i-1]    => (size(i-1),60000)
            # B[i]      => (size(i))
            a_next = 1 / (1 + np.exp(-1 * z_next))
            self.A.append(a_next)
            self.Z.append(z_next)
        self.A_OUTPUT_LAYER = np.sum(self.A[-1], axis=0)

    def _calculate_loss__(self):
        """
        This function preps data for backward propagation to start, mainly by calculating
        Loss function, Cost function and derivative of Loss function with last
        :return:
        """
        self.L = 0.5 * np.square(self.A_OUTPUT_LAYER - self.Y)
        self.J = np.sum(self.L) / self.m
        self.LossMomentOnOutput = self._moment_lossOnOutput__(self.A_OUTPUT_LAYER, self.Y)


    def _prep_backward_propagation__(self):
        """
        This function prepares backward propagation,
        by resetting necessary variables.
        :return:
        """

        # The above is loss differential for last layer, for the loss
        # function which is a standard deviation. This needs to be replaced with
        # a first differential of Loss function on output function.
        self.LossDifferential = [np.ones([x, self.m]) for x in self.size[1:]]
        # The loss differential is an important entity with special properties,
        # that helps immensenly in back propagation. It is defined as follows
        # loss differential, mu(i,l) = d(a_output)/d(a_l)
        # where l denotes the layer in neural schema, and i is the ith neuron in the layer
        # This has a unique property that mu(l,i) = Sum( mu(l+1,j) * w(j,i))
        # where j takes value from 1 to no of neurons in layer l+1.
        # for the last layer(output layer), mu(l,i) = 1.
        self.dW = list(range(len(self.size) - 1))  # Store the gradient of Weights against Loss function for each layer
        self.db = list(range(len(self.size) - 1))  # Store gradient of bias against Loss function for each layer.

    def _backward_propagate__(self):
        """
        This function completes the backward propagation across all layers.
        :return:
        """
        self._calculate_loss__()
        # db_over_training_data = self.LossMomentOnOutput * self._moment_of_activation_function_on_weighted_output__(len(self.size) - 1)
        # db = np.sum(db_over_training_data, axis=0, keepdims=True)
        # dw = np.dot(db_over_training_data, self.A[-2].T)
        # self.W[-1] = 0.0001 * dw
        # self.B[-1] = 0.0001 * db
        for layer in range(len(self.W), 0, -1):
            w_layer_index = layer - 1  # calibrate the iteration counter to remove input layer in weights and biases
            if layer == len(self.W):
                mu_layer = np.ones((self.size[-1], self.m))
            else:
                mu_layer = np.dot(np.multiply(self.LossDifferential[w_layer_index + 1],
                                              self._moment_of_activation_function_on_weighted_output__(
                                                  w_layer_index + 2)).T, self.W[w_layer_index + 1]).T

            self.LossDifferential[w_layer_index] = mu_layer
            k = self.LossMomentOnOutput * mu_layer
            moment_of_layer = self._moment_of_activation_function_on_weighted_output__(w_layer_index + 1)
            db_spread_over_training_data = k * moment_of_layer
            dW = np.dot(db_spread_over_training_data, self.A[w_layer_index].T)
            db = np.sum(db_spread_over_training_data, axis=1, keepdims=True) / self.m
            self.W[w_layer_index] -= self.eta * dW  # a hardcoded learning rate of 1/1,00,000
            self.B[w_layer_index] -= self.eta * db

    def _backward_propagate_2__(self):
        """
        This function completes the backward propagation across all layers.
        :return:
        """
        self._calculate_loss__()
        db_over_training_data = self.LossMomentOnOutput * self._moment_of_activation_function_on_weighted_output__(len(self.size) - 1)
        db = np.sum(db_over_training_data, axis=1, keepdims=True)
        dw = np.dot(db_over_training_data, self.A[-2].T)
        self.W[-1] -= self.eta * dw
        self.B[-1] -= self.eta * db
        for layer in range(len(self.W) - 1, 0, -1):
            db_over_training_data = np.dot(db_over_training_data.T, self.W[layer]).T * self._moment_of_activation_function_on_weighted_output__(layer)
            db = np.sum(db_over_training_data, axis=1, keepdims=True)
            dw = np.dot(db_over_training_data, self.A[layer - 1].T)
            self.W[layer - 1] -= self.eta * dw # a hardcoded learning rate of 1/1,00,000
            self.B[layer - 1] -= self.eta * db

    def _evaluate(self):
        """
        Evaluate percentage of succussful prediction in every epoch
        :return: return the percentage of successful prediction
        """
        a = np.isclose(self.A_OUTPUT_LAYER, self.Y, atol=0.01, rtol=0.01)
        print("Shape of a", np.shape(a), end=" ")
        b = np.count_nonzero(np.isclose(self.A_OUTPUT_LAYER, self.Y, 0.001, atol=0.01))
        print("No of non zero = ", b, end=" ")
        return b/self.m

    def train(self, epochs=10):
        """ This is the externally exposed class, which is just a wrapper
            on forward and backward propagation functions.
            epochs: No of epochs to train the data
        """
        self.epochs = epochs
        # initialize weighted output(Z) and activation function output for this epoch
        for i in range(self.epochs):
            print("Epoch ", i, end=" ")
            self._prepare_epoch__()
            self._propagate_forward__()
            self._prep_backward_propagation__()
            self._backward_propagate_2__()
            # self._backward_propagate__()
            print("J", self.J, end= " " )
            rate = self._evaluate()
            print("Success results =", rate)
            self.cost_function.append(self.J)
            self.success_rate.append(rate)

        print(self.success_rate)
        fig, (ax1,ax2) = matplotlib.pyplot.subplots(2,1)
        ax1.plot(range(self.epochs), self.cost_function)
        ax2.plot(range(self.epochs), self.success_rate)
        matplotlib.pyplot.show()
        print("m ", self.m)
        print("Shape of A", np.shape(self.A[-1]))
        print("Shape of Y", np.shape(self.Y))
        print("A values", self.A[-1].T[0])
        print("A out put ", self.A_OUTPUT_LAYER[:10])
        print("Y", self.Y.T[:10])
