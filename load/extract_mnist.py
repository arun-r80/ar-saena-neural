import numpy as np
import os, pathlib, gzip, pickle
import matplotlib.pyplot

print("Starting unpickling")
training_image_gzip_extract = gzip.open(os.path.join(pathlib.PurePath(), "../data", "train_image_pkl_gz.gz"))
training_image_pickle = pickle.load(training_image_gzip_extract)
training_image_np = np.array(training_image_pickle)
print("Unpickling complete!!!")
print("Shape of Unpickled object", np.shape(training_image_np))


training_image = [ np.reshape(x, (28,28)) for x in np.reshape(training_image_np, (60000, 784))]
print("Printing the first ten images extracted from unpickled object!!")
for i in range(10):
    matplotlib.pyplot.imshow(training_image[i])
    matplotlib.pyplot.show()

training_image_gzip_extract.close()



