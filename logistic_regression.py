import numpy as np
import math

# update on dev
def sigmoid(z):
    return 1 / (1 + math.exp(-z))

m = 10
X = np.random.random(m)
print(X)
Y = np.random.random(m)
W = np.random.random(m)
print(W)
J = 0
w = 0
b = 0
loss = 0
ycap = 0
learning_rate = 0.001

C = W * X + b
Sigma = 1 / (1 + np.exp(-1 * C))
print(C)
print("Sigmoid Function of C ", Sigma)
# for i in range(10):
#     Z = np.dot(W, X) + b
#     A = 1 / (1 + math.exp(-1 * Z))
#
#     loss = -1 * (Y[i] * math.log(ycap)+(1-Y[i])*math.log(1-ycap))
#
#     J += loss
# J /= m
# print(J)
# print(Z)
# print(A)


