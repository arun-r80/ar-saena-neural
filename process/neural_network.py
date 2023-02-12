"""
This is the module which creates a class to
train - ie., to forward and backpropagate on training data.
"""


import numpy as np
import os, \
    matplotlib, \
    gzip, \
    pickle, \
    time



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

    def __init__(self, training_data, no_of_neyral_layers, no_of_training_set_members=60000 ):
        """
        Initialize class with size as input.
        :param size: a list which contains no of neurons for each layer.So, len(size) will provide total
        no of layers in this neural schema, including input(which contains features or "X" values)
        and output layers.
        """
        self.size = no_of_neyral_layers
        self.m = no_of_training_set_members
        # random assignment of weights for each layer
        self.W = [np.random.rand(*z)for z in list(zip([x for x in self.size[1:]], [y for y in self.size[:-1]]))]
        # random assignment of bias for each layer
        self.B = [np.random.rand(x,1) for x in self.size[1:]]
        for i in range(0,len(self.size)-1):
            print("Shape of B", np.shape(self.B[i]))
        # Open and populate training data into object instance variables.
        training_datafile = gzip.open(training_data, mode="rb")
        training_datafile_set = pickle.load(training_datafile)
        training_x, training_y = training_datafile_set
        # training data file contains a tuple of training dataset and corresponding categories
        # The training data set is a single dimensional array which contains color RGB for
        # each of 60000 image, with each image being represented as an array of 784 pixels,
        # and these 784 pixels, in turn, refer to 28x28 pixels.
        self.X = np.reshape(training_x, [self.m, 784]).T  # X => (784, 60000)

        self.Y = training_y   # Y => (60000,1)
        print(np.shape(self.Y))
        self.epochs = 10 # initialize epochs for the training model


    def _moment_lossOnOutput__(self, OutPut, Y):
        """
        This function returns the first moment of loss function(L) on classifier output(a)
        when the Loss function is chosen as the standard deviation, ie:
        L = 1/2 * (a - Y)^2
        :param OutPut: The output of neural network a
        :param Y: Test data labels corresponding to the input data
        :return: The first derivative of loss function on the output value
        """
        a = OutPut - Y
        print("Shape of moment of loss on a output ", np.shape(a))
        return OutPut - Y


    def _moment_Of_Activation_Function_On_Output__(self, layer=None):
        k = np.multiply(self.A[layer-1], (1 - self.A[layer-1]))
        print("Shape of first moment ", np.shape(k))
        return np.multiply(self.A[layer-1], (1 - self.A[layer-1]))


    def _prepare_epoch__(self):
        self.Z = []
        self.A = []
        self.A.append(self.X)
        self.L = 0       # initialize Loss function to be zero, for the entiretity of dataset.
        self.J = 0       # so initialize the cost function as well.

    def _propagate_forward__(self):
        """
        This function does the forward propagation, which entails following the below steps
        in the order given:
        a. Calculate weighted output(Z) for each layer, for all the input dataset.
        b. Calculate activation function output(A) for each layer, and again for each and every datapoint in training dataset
        c. Calculate loss function, based on activated output above, for all datapoints in dataset.
        :return:
        """
        t_start = time.time()
        for i in range(len(self.W)):
            t_zStart = time.time()
            z_next = np.dot(self.W[i], self.A[i]) + self.B[i]
            t_zEnd = time.time()
            # W[i]      => (size(i), size(i-1))
            # A[i-1]    => (size(i-1),60000)
            # B[i]      => (size(i))
            t_a_start = time.time()
            a_next = 1/(1 + np.exp(-1 * z_next))
            t_a_end = time.time()
            print("Time taken for Z calculation in layer ", i+1, ":", t_zEnd - t_zStart)
            print("Time taken for A calcuation in layer ", i+1 , ":", t_a_end - t_a_start)
            print("Shape of A ", np.shape(a_next))
            self.A.append(a_next)
        self.A_OUTPUT_LAYER = np.sum(self.A[-1], axis=0)
        t_end = time.time()
        print("Time taken for forward propagation ", t_end - t_start)
        print("Shape of A_OutPut Layer ", np.shape(self.A_OUTPUT_LAYER))

    def _calculate_loss__(self):
        """
        This function preps data for backward propagation to start, mainly by calculating
        Loss function, Cost function and derivative of Loss function with last
        :return:
        """
        self.L = 0.5 * np.square(self.A_OUTPUT_LAYER - self.Y)
        self.J = np.sum(self.L, axis=0) / self.m

    def _prep_backward_propagation(self):
        """
        This function prepares backward propagation,
        by resetting necessary variables.
        :return:
        """
        self.LossMomentOnOutput = self._moment_lossOnOutput__(self.A_OUTPUT_LAYER, self.Y)
        # The above is loss differential for last layer, for the loss
        # function which is a standard deviation. This needs to be replaced with
        # a first differential of Loss function on output function.
        self.LossDifferential = [np.ones(x) for x in self.size[1:]]
        print("Shape of Loss Diff 1 ", np.shape(self.LossDifferential[0]), np.shape(self.LossDifferential[1]))

        # The loss differential is an important entity with special properties,
        # that helps immensenly in back propagation. It is defined as follows
        # loss differential, mu(i,l) = d(a_output)/d(a_l)
        # where l denotes the layer in neural schema, and i is the ith neuron in the layer
        # This has a unique property that mu(l,i) = Sum( mu(l+1,j) * w(j,i))
        # where j takes value from 1 to no of neurons in layer l+1.
        # for the last layer(output layer), mu(l,i) = 1.
        self.dW = list(range(len(self.size) -1 ))  # Store the gradient of Weights against Loss function for each layer
        self.db = list(range(len(self.size) - 1 ))  # Store gradient of bias against Loss function for each layer.



    def _backward_propagate__(self):
        """
        This function completes the backward propagation across all layers.
        :return:
        """
        self._calculate_loss__()
        # calculate dW and db for output layer,
        # as it is a special case
        db_spread_over_training_data = self._moment_Of_Activation_Function_On_Output__(len(self.size)) * self._moment_lossOnOutput__(self.A_OUTPUT_LAYER, self.Y)
        db = np.sum(db_spread_over_training_data, axis=1) / self.m
        db_test_npsum = np.sum(db_spread_over_training_data, axis=1)

        print("dB = ", db[0], db_test_npsum[0] / 60000)
        print("Shape of db = product of two moments ", np.shape(db) )
        print("Shape of B in outermost layer ", np.shape(self.B[-1]))
        print("Shape of A pervious player ", np.shape(self.A[-2]))
        dW = np.dot(db_spread_over_training_data, self.A[-2].T) / self.m
        print("Shape of dW in the outermost layer ", np.shape(dW))





    def train(self, epochs=10):
        """ This is the externally exposed class, which is just a wrapper
            on forward and backward propagation functions.
            epochs: No of epochs to train the data
        """
        self.epochs = epochs
        # initialize weighted output(Z) and activation function output for this epoch

        self._prepare_epoch__()
        self._propagate_forward__()
        self._prep_backward_propagation()
        self._backward_propagate__()










