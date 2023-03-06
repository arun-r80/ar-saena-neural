"""
Load MNIST dataset using boiler-plate Python code, and pickle/compress the
dataset for easier retrieval later.
For details on MNIST dataset, please visit: http://yann.lecun.com/exdb/mnist/
To understand the logic and offsets (refer to code comments),
review the file structure for each of the files that contain training data and testing data.

============================================================================================
============================================================================================
TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000803(2051) magic number
0004     32 bit integer  60000            number of images
0008     32 bit integer  28               number of rows
0012     32 bit integer  28               number of columns
0016     unsigned byte   ??               pixel
0017     unsigned byte   ??               pixel
........
xxxx     unsigned byte   ??               pixel
Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

"""

import gzip
import numpy as np
import matplotlib.pyplot as pyplot
import os
import pathlib
import pickle
import shutil
import sys, configparser



def __bytes2int__(b):
    """
    :param b: bytes object returned as result of read() function on a file object. The file object
    referred here is compressed gzip file of the dataset - one of the four training/testing dataset
    :return: the string form of the bytes object (parameter) b
    """
    return int(bytes.hex(b), base=16)

def load_mnist(projectrootfolder):
    """
    This function loads training and testing dataset from MNIST database, which
    have been loaded into "data" folder.
    Please note that the database source files may not be part of code repo in the future,
    or when the redimensioned dataset has been stored as gzip files.
    :return:
    """
    config_file = open(os.path.join(projectrootfolder, "config", "config.cfg"))
    config = configparser.ConfigParser()
    config.read_file(config_file)
    print(config.get("Load data","dataset_to_load" ))
    data_set_to_load = config.get("Load data","dataset_to_load")
    if data_set_to_load == "1" or data_set_to_load == "0":

        #
        # The gzip files being opened here may not be part of "data" subfolder in future commits.
        # These gzip files are the actual MNIST dataset that can be downloaded from http://yann.lecun.com/exdb/mnist/
        print("Starting extraction for training images.......")
        print("==============================================")
        train_object = gzip.open(str(os.path.join(projectrootfolder, "data", "train-images-idx3-ubyte.gz")), mode='rb')
        # Header information in training image dataset is printed below, to help the user in understanding.
        print("Magic Number : ", __bytes2int__(train_object.read(4)))
        print("No of Images : ", __bytes2int__(train_object.read(4)))
        print("No of rows   : ", __bytes2int__(train_object.read(4)))
        print("No of Columns: ", __bytes2int__(train_object.read(4)))

        train_image_pixels = []
        read_bytes = train_object.read(1)
        i = 0
        while read_bytes != b'':
            train_image_pixels.append(__bytes2int__(read_bytes))
            i = i + 1
            print("Reading byte: ",i, end='\r')
            read_bytes = train_object.read(1)

        train_object.close()
        print("Reading complete")
        print("Starting pickling....")

        train_image_pixels_pkl = open(os.path.join(pathlib.PurePath(), "data", "train_image_pkl"), mode="wb")
        pickle.dump(train_image_pixels, train_image_pixels_pkl)
        train_image_pixels_pkl.close()
        print("Pickling complete!")

        print("Compressing the pickle(Why would Github not allow files above 50MB size??)")
        with open(os.path.join(projectrootfolder, "data", "train_image_pkl"), mode="rb") as pkl:
            with gzip.open(os.path.join(projectrootfolder, "data", "train_image_pkl_gz.gz"), mode="wb") as pkl_gzip:
                shutil.copyfileobj(pkl, pkl_gzip)

        pkl.close()
        pkl_gzip.close()

        print("Now compressed the pickle!!")

    if data_set_to_load == "2" or data_set_to_load == "0":
        print("Starting extraction for training labels.......")
        print("==============================================")

        train_image_labels_gzip = gzip.open(os.path.join(projectrootfolder, "data", "train-labels-idx1-ubyte.gz"), mode="rb")
        print("TRAINING LABELS FILE STRUCTURE:")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("Magic Number     :", __bytes2int__(train_image_labels_gzip.read(4)))
        print("No of items      :", __bytes2int__(train_image_labels_gzip.read(4)))

        print("Starting reading from the file:")
        train_label_set = []
        i = 0
        read_train_label_bytes = train_image_labels_gzip.read(1)
        print("")
        while read_train_label_bytes != b'':
            train_label_set.append(__bytes2int__(read_train_label_bytes))
            i = i + 1
            read_train_label_bytes = train_image_labels_gzip.read(1)
            print("Reading byte: ", i, end='\r')

        print("Completed reading training label file")
        for i in range(10):
            print("Image Label for element  ", i, "is", train_label_set[i])

        print("Starting pickling for training labels")
        train_labels_pkl_file = open(os.path.join(projectrootfolder, "data", "train_label_pkl.pkl"), mode="wb" )
        pickle.dump(train_label_set, train_labels_pkl_file)
        train_labels_pkl_file.close()
        print("Pickling complete")

        print("Compressing train label")

        with gzip.open(os.path.join(projectrootfolder,"data","train_label_pkl_gz.gz"), mode="w" ) as pkl_gz:
            with open(os.path.join(os.getcwd(), "data", "train_label_pkl.pkl"), mode="rb") as pkl:
                shutil.copyfileobj(pkl, pkl_gz)

        pkl.close()
        pkl_gz.close()
        print("Completed compressing train label")
    # # print(np.shape(a))
    # # b = np.reshape(a,(28, 28)
    #
    # pyplot.imshow(b)
    # pyplot.show()
    # train_images = np.zeros((60000,768), dtype='int')
    #
    # train_images = np.fromfile('.\\data\\train-images-idx3-ubyte.gz', dtype=np.hex, count=768)
    # print((train_images[1]))
    # print(np.shape(train_images))
    # print(train_images)
