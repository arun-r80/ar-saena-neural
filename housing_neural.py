# basic housing neural network
# built in PyCharm editor than in Google Colab

import numpy as np
import tensorflow as tf
from tensorflow import keras
import time as time

print("In Program")
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# model.fit(xs, ys, epochs=500)
#
# print(model.predict([30.0, 20.0]))

a = np.random.rand(1000000)
b = np.random.rand(1000000)

before_proccessing = time.time()
c = np.dot(a,b)
after_processing = time.time()


before_processing_nonvec = time.time()
c= 0
for i in range(1000000):
    c += a[i] * b[i]
after_processing_nonvec = time.time()
print("Time taken for vectorized dot product is ",1000 * (after_processing - before_proccessing), "ms" )
print("Time taken for non vectorized dot product is ", 1000 * (after_processing_nonvec - before_processing_nonvec), "ms")




