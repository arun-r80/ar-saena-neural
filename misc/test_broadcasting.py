import numpy as np

a = np.array([[5, 4], [3, 2]])
b = np.array([[0.5, 0.25], [0.5, 0.25]])

X = 1/(1 + np.exp(-1 * a))
Y = np.square(X)

x = np.random.randn(4,2,2)
print(x)
y = np.reshape(x, (8,2))
print(y)
# c = a / (b * (1 - b))
# print(c)

# print("a= ", a)
# d = np.sum(a, axis=1, keepdims=False)
# e = np.sum(a, axis=1, keepdims=True)
# print(d)
# print(e)

