import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt;


class LossCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('loss') < 0.4:
            print("Cancelling training as loss limit is adequate")
            self.model.stop_training = True


loss_callback = LossCallback()
# Load MNIST data
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_label), (test_images, test_label) = fashion_mnist.load_data()

# ##Print one of the training data
# np.set_printoptions(linewidth=250)
# # set the index
# index = 32
#
# print(f'LABEL: {train_label[index]}')
# print(f'\n Training Image Array\n {train_images[index]}')
#
# #print image
# print("Printing Image")
# plt.imshow(train_images[index])
# plt.show()
#
# # plt.imshow(train_images[0])
# # plt.show()

# create the model
model = tf.keras.Sequential([tf.keras.layers.Flatten(),
                             tf.keras.layers.Dense(128, activation=tf.nn.relu),
                             tf.keras.layers.Dense(10, activation=tf.nn.softmax)
                             ])

#compile the model
model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy']

              )

model.fit(train_images, train_label, epochs=25, callbacks=[loss_callback])

#evaluate model
print("Evaluating model")
model.evaluate(test_images, test_label)