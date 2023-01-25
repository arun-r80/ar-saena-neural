import numpy as np


def g2_first_moment(a, y):
    derivative = (a - y) / (a * (1 - a))
    return derivative

def activation_sigmoid_moment(A, Z):
    A_Squared = np.square(A)
    Z_Exp_Inverse = np.exp(-1 * Z)
    return A_Squared * Z_Exp_Inverse


m = 20          # no of training sets
nx = 4          # no of features
n1 = 15         # No of neurons in first layer
n2 = 1          # no of neurons in second layer
alpha = 0.05    # learning rate

# Training set data - simulated
X = np.random.random(size=(nx, m))
Y = np.random.random(size=(1, m))
# Neuron architecture
# Layer 1 neurons
W1 = np.random.random(size=(n1, nx))
b1 = np.random.random(size=(n1, 1))
W2 = np.random.random(size=(n2, n1))
b2 = np.random.random(size=(n2, 1))
#for i in range(10):
# Compute increments for Weight and Bases
# in Layer 2 of the neural network
Z1 = np.dot(W1, X) + b1         # shape (n1,m)
# print("X = ", X)
# print("W1 = ", W1)
print("Shape of Z1", np.shape(Z1))
A1 = 1/(1 + np.exp(-1 * Z1))    # Shape (n1,m)
print("Shape of A1 ", np.shape(A1))
# print("Shape of W2 ", np.shape(W2))
Z2 = np.dot(W2, A1) + b2         # Shape (n2,m)
print("Shape of Z2", np.shape(Z2))
A2 = 1/(1+np.exp(-1 * Z2))       # Shape (n2,m)
# print("A2 is ", A2)
print("Shape of A2", np.shape(A2))
#Backward propagation
dZ2 = A2 - Y                                # Shape (n2,m)
print("Shape of dZ2 ", np.shape(dZ2))
d = g2_first_moment(A2, Y)                  # Shape (n2, m)
df = dZ2                              # Shape (n2,m)
dW2_across_training_data = df          # Shape (n2,m)
dW2_summation = np.sum(dW2_across_training_data, axis=1, keepdims=True )  # Shape (n2,1)
dW2 = dW2_summation/m       #  Shape(n2,1)
db2 = np.sum(dZ2, axis=1, keepdims=True)  # Shape(n2,1)
print("dW2", dW2)
print("Shape of dW2", np.shape(dW2))
print(db2,"Shape of db2 ", np.shape(db2))  # Shape (n2,m)
W2 -= alpha * dW2     # Shape (n2,n1)
b2 -= alpha * db2     # Shape (n2, 1)

# Calculate incremental changes for Layer 1
# The variable df1 defined below, is to get dZ1 as follows
# dL/dZ1 = dL/dA2 * dA2/dZ2 * dZ2/dA1 * dA1/dZ1
# Here dZ2/dA1 will be W2, the array of weights on Neuron layer 2
# dA1/dZ1 is nothing but first moment (first derivative) of g1 - the activation function
# network layer 1.
# Now, dL/dA2 * dA2/dZ2, the first two factors, is nothing but dZ2.
# So dL/dZ1 = dZ2 * W2 * activation_sigmoid_moment(A1, Z1)
df1 = W2 * dZ2  # Shape (n2,m), elementwise
activation_function_moment_layer1 = activation_sigmoid_moment(A1, Z1) # Shape(n1,m)
dZ1_individual_training_samples = df1 * W2 * activation_function_moment_layer1
# Shape (n2,m) * (n2,




# a = np.array([[1, 2,3], [3, 2, 1]])
# print(a)
# b = np.array([[1,1],[2,2],[3,3]])
# print(b)
# c = np.array([[5,5]])
# print(c)
# print(np.dot(a, b))
# print(np.dot(a, b) + c)
