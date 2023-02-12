from process import neural_network
import configparser, pathlib, os, math, numpy

config_parser = configparser.ConfigParser()
configfile = open(os.path.join(pathlib.PurePath(os.getcwd()).parent, "config", "config.cfg" ), mode="r")
print(os.path.join(pathlib.PurePath(os.getcwd()).parent))

config_parser.read_file(configfile)
training_data_filename = config_parser.get("Populate Data", "training_set_file")
training_data_file = os.path.join(pathlib.PurePath(os.getcwd()).parent, "data", training_data_filename )

test_neural = neural_network.Neural(training_data_file, [784, 9, 10], 60000)
test_neural._prepare_epoch__()
test_neural._propagate_forward__()
c=0
w = test_neural.W[0]

for i in range(784):

    c = c + test_neural.W[0][0][i]  * test_neural.X[i][0]
c = c + test_neural.B[0][0][0]
print(c, math.exp(-1*c))
a = 1/(1+math.exp(-1 * c))
print("Activation function from testing", a)
print("Calculated activation function", test_neural.A[1][0][0],"Shape", numpy.shape(test_neural.A[1][0]) )