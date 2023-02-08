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
        for i in range(1,len(self.size)-1):
            print("Shape of B", np.shape(self.B[i]))
        # Open and populate training data into object instance variables.
        training_datafile = gzip.open(os.path.join(os.getcwd(), "data", training_data), mode="rb")
        training_datafile_set = pickle.load(training_datafile)
        training_x, training_y = training_datafile_set
        # training data file contains a tuple of training dataset and corresponding categories
        # The training data set is a single dimensional array which contains color RGB for
        # each of 60000 image, with each image being represented as an array of 784 pixels,
        # and these 784 pixels, in turn, refer to 28x28 pixels.
        self.X = np.reshape(training_x, [self.m, 784]).T  # X => (784, 60000)

        self.Y = np.reshape(training_y, [self.m, 1])   # Y => (60000,1)
        print(np.shape(self.Y))
        self.epochs = 10 # initialize epochs for the training model

    def __prepare_epoch__(self):
        self.Z = []
        self.A = []
        self.A. append( self.X)
        self.L = 0       # initialize Loss function to be zero, for the entiretity of dataset.
        self.J = 0       # so initialize the cost function as well.

    def propagate_forward(self):
        """
        This function does the forward propagation, which entails following the below steps
        in the order given:
        a. Calculate weighted output(Z) for each layer, for all the input dataset.
        b. Calculate activation function output(A) for each layer, and again for each and every datapoint in training dataset
        c. Calculate loss function, based on activated output above, for all datapoints in dataset.
        :return:
        """
        for i in range(len(self.W)):
            Z_next = np.dot(self.W[i], self.A[i]) + self.B[i]
            # W[i]      => (size(i), size(i-1))
            # A[i-1]    => (size(i-1),60000)
            # B[i]      => (size(i))
            A_next = 1/(1 + np.exp(-1 * Z_next))
            self.A.append(A_next)
        self.A_OUTPUT_LAYER = np.sum(self.A[-1],axis=0).reshape([self.m,1])
        print("Shape of Output layer activation function ", np.shape(self.A_OUTPUT_LAYER))



    def train(self, epochs=10):
        """ This is the externally exposed class, which is just a wrapper
            on forward and backward propagation functions.
            epochs: No of epochs to train the data
        """
        self.epochs = epochs
        # initialize weighted output(Z) and activation function output for this epoch

        self.__prepare_epoch__()
        self.propagate_forward()










