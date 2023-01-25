import gzip
import numpy as np
import matplotlib.pyplot as pyplot
import os
import pathlib
import pickle


def __bytes2int__(b):
    return int(bytes.hex(b), base=16)

train_object = gzip.open(str(os.path.join(pathlib.PurePath(), "data","train-images-idx3-ubyte.gz" )), mode='rb')
# #train_object = gzip.open('.\\data\\train-images-idx3-ubyte.gz', mode='rb')
# print("Magic Number : ", __bytes2int__(train_object.read(4)))
# print("No of Images : ", __bytes2int__(train_object.read(4)))
# print("No of rows   : ", __bytes2int__(train_object.read(4)))
# print("No of Columns: ", __bytes2int__(train_object.read(4)))
# a = np.zeros(784,dtype=np.uint8)
train_image_pixels = []
train_image_pixels.append(1)
print(train_image_pixels)
train_image_pixels.append(__bytes2int__(train_object.read(1)))
read_bytes = train_object.read(1)
i=0
while read_bytes != b'':
    train_image_pixels.append(__bytes2int__(read_bytes))
    i=i+1
    print(i)
    read_bytes = train_object.read(1)
    if i == 47040015:
        print("Final  Byte" ,read_bytes)
        break

# test_eof = open(os.path.join(pathlib.PurePath(), "data", "testdata.txt"))
# i = 0
# eof = []
# read_byte = test_eof.read(1)
# while read_byte != '':
#     i = i + 1
#     print(i)
#     eof.append(read_byte)
#     read_byte = test_eof.read(1)
#
# print("Object before pickling ", eof)
# print("Starting test pickling")
# test_pickle = open(os.path.join(pathlib.PurePath(), "data", "test_pkl"), mode="wb")
# pickle.dump(eof,test_pickle)
# test_pickle.close()
# print("Pickled!!!!!")
#
# print("GEt the pickle....")
# unpkl_file = open(os.path.join(pathlib.PurePath(), "data", "test_pkl"), mode="rb")
# eof_unpkld = pickle.load(unpkl_file)
# print("Unpickeld ", eof_unpkld)

print("Reading complete")
print("Starting pickling....")

train_image_pixels_pkl = open(os.path.join(pathlib.PurePath(), "data", "train_image_pkl"), mode="wb")
pickle.dump(train_image_pixels, train_image_pixels_pkl )

print("Pickling complete!")




# print(np.shape(a))
# b = np.reshape(a,(28, 28))
#
# pyplot.imshow(b)
# pyplot.show()
#train_images = np.zeros((60000,768), dtype='int')
#
# train_images = np.fromfile('.\\data\\train-images-idx3-ubyte.gz', dtype=np.hex, count=768)
# print((train_images[1]))
# print(np.shape(train_images))
# print(train_images)
