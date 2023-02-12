# This is a sample Python script.
import configparser
import os
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import sys
from load import load_mnist, extract_mnist
from process import neural_network as neural
import numpy as np



# Read configuration file for important parameters
configfile = open(os.path.join(os.getcwd(),"config", "config.cfg"), mode="r")
config = configparser.ConfigParser()
config.read_file(configfile)
training_data_redimensioned_file = config.get("Populate Data", "training_set_file")
print(training_data_redimensioned_file)



def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')


training_data_file = os.path.join(os.getcwd(), "data",training_data_redimensioned_file)
# The following lines of code create training data file which ultimately serves as input
# for neural network. These can be uncommented if need be, but currently the gzip file with training data
# is available to be fed into neural network.
#load_mnist.load_mnist(projectrootfolder)
#extract_mnist.redim_training_dataset(projectrootfolder)
# initialize neuron structure for neural schema.
handwritten_digits = neural.Neural( training_data_file, [784, 9, 10], 60000)
handwritten_digits.train(10)



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
