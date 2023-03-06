# This is a sample Python script.
import configparser
import os
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from process import neural_network as neural
import numpy as np
# Read configuration file for important parameters
training_data_file = os.path.join(os.getcwd())#, "data",training_data_redimensioned_file)
# The following lines of code create training data file which ultimately serves as input
# for neural network. These can be uncommented if need be, but currently the gzip file with training data
# is available to be fed into neural network.
# initialize neuron structure for neural schema.
handwritten_digits = neural.Neural(training_data_file, [784,10, 9], 60000)
handwritten_digits.train(300)



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
