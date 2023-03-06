"""
This module loads training data and pickles them.
"""

import gzip, os, pathlib
import numpy as np
import matplotlib.pyplot as plot

test_bytes = bytearray(784*10000 +16)
test_file = gzip.open(os.path.join(pathlib.PurePath(os.getcwd()).parent, "data", "t10k-images-idx3-ubyte.gz"), mode="rb")
testing_image_set = test_file.readinto(test_bytes)
testing_image_sample = [np.reshape(x, (28,28)) for x in np.reshape(np.array(test_bytes[16:]), (10000, 784))]
print("Shape of testing data", np.shape(testing_image_sample))
for i in range(10):
    plot.imshow(testing_image_sample[i])
    plot.show()

