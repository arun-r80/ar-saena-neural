import gzip
import numpy as np
import matplotlib.pyplot as pyplot

def __bytes2int__(b):
    return int(bytes.hex(b), base=16)


train_object = gzip.open('.\\data\\train-images-idx3-ubyte.gz', mode='rb')
print("Magic Number : ", __bytes2int__(train_object.read(4)))
print("No of Images : ", __bytes2int__(train_object.read(4)))
print("No of rows   : ", __bytes2int__(train_object.read(4)))
print("No of Columns: ", __bytes2int__(train_object.read(4)))
a = np.zeros(784,dtype=np.uint8)

for i in range(784):
    a[i] = __bytes2int__(train_object.read(1))

print(np.shape(a))
b = np.reshape(a,(28, 28))

pyplot.imshow(b)
pyplot.show()
#train_images = np.zeros((60000,768), dtype='int')
#
# train_images = np.fromfile('.\\data\\train-images-idx3-ubyte.gz', dtype=np.hex, count=768)
# print((train_images[1]))
# print(np.shape(train_images))
# print(train_images)
