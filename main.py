# This is a sample Python script.
import configparser
import os
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from process import neural_network as neural
from process import neural_network2 as neural2
from process import neural_network_2_1 as neural2_1
from process import neural_network_2_2 as neural2_2
from process import neural_network2_3 as neural2_3
import numpy as np

# Read configuration file for important parameters
training_data_file = os.path.join(os.getcwd())  # , "data",training_data_redimensioned_file)
# The following lines of code create training data file which ultimately serves as input
# for neural network. These can be uncommented if need be, but currently the gzip file with training data
# is available to be fed into neural network.
# initialize neuron structure for neural schema.
# handwritten_digits = neural.Neural(training_data_file, [784,10, 9], 60000)
# handwritten_digits.train(500)
#
handwritten_digits = neural2.Neural_2(training_data_file, [784, 100, 9],
                                      no_of_training_set_members=50000,
                                      no_of_validation_data_members=10000,
                                      eta=0.25,
                                      l_regularize=0.15,
                                      m=1000)
handwritten_digits.train(100)

handwritten_digits = neural2_3.neural_2_3(training_data_file, [784, 30, 10],
                                          no_of_training_set_members=50000,
                                          no_of_validation_data_members=10000,
                                          eta=0.5,
                                          l_regularize=0.15,
                                          m=10)

handwritten_digits.train(10)

handwritten_digits = neural2_2.neural_2_2(training_data_file, [784, 30, 10],
                                          no_of_training_set_members=50000,
                                          no_of_validation_data_members=10000,
                                          eta=0.01,
                                          l_regularize=0.15,
                                          m=10)
handwritten_digits.train(10)
#
handwritten_digits = neural2_1.neural_2_1(training_data_file, [784, 30, 10],
                                          no_of_training_set_members=50000,
                                          no_of_validation_data_members=10000,
                                          eta=0.5,
                                          l_regularize=.15,
                                          m=10)
handwritten_digits.train(10)






# See PyCharm help at https://www.jetbrains.com/help/pycharm/
