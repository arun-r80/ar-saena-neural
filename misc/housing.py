import tensorflow as tf
from tensorflow import keras
import numpy as np
import sys
print('In Program', sys.path)


# GRADED FUNCTION: house_model
def house_model():
    ### START CODE HERE

    # Define input and output tensors with the values for houses with 1 up to 6 bedrooms
    # Hint: Remember to explictly set the dtype as float
    xs = np.array([0, 1, 2, 3, 4, 5, 6], dtype=int)
    ys = np.array([0.50, 1.00, 1.50, 2.00, 2.50, 3.00, 3.50], dtype=float)

    # Define your model (should be a model with 1 dense layer and 1 unit)
    model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

    # Compile your model
    # Set the optimizer to Stochastic Gradient Descent
    # and use Mean Squared Error as the loss function
    model.compile(optimizer='sgd', loss='mean_squared_error')

    # Train your model for 1000 epochs by feeding the i/o tensors
    model.fit(xs, ys, epochs=1000)

    ### END CODE HERE
    return model


model = house_model()
new_y = 7.0
prediction = model.predict([new_y])[0]
print(prediction)


