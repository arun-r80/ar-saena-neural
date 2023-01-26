import numpy as np
import os, pathlib, gzip, pickle
import matplotlib.pyplot

print("Starting unpickling training images")
training_image_gzip_extract = gzip.open(os.path.join(os.path.dirname(os.getcwd()), "data", "train_image_pkl_gz.gz"))
training_image_pickle = pickle.load(training_image_gzip_extract)
training_image_np = np.array(training_image_pickle)
training_image = [ np.reshape(x, (28,28)) for x in np.reshape(training_image_np, (60000, 784))]
print("Unpickling complete!!!")
print("Shape of Unpickled training image object", np.shape(training_image_np))


print("Starting unpickling testing images")
training_label_gzip = gzip.open(os.path.join(os.path.dirname(os.getcwd()), "data", "train_label_pkl_gz.gz"))
training_label_pkl = pickle.load(training_label_gzip)
training_label = np.array(training_label_pkl)
print("Shape of unpickled training label object", np.shape(training_label))


training_dataset = (training_image, training_label)


