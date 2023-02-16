
import numpy as np
import os, pathlib, gzip, pickle, sys
import matplotlib.pyplot


def redim_training_dataset(projectrootfolder):
    print("Starting unpickling training images")
    training_image_gzip_extract = gzip.open(os.path.join(projectrootfolder, "data", "train_image_pkl_gz.gz"), mode="rb")
    training_image_pickle = pickle.load(training_image_gzip_extract)
    training_image_np = np.array(training_image_pickle)
    training_image =  np.reshape(training_image_np, (60000, 784))
    print("Unpickling complete!!!")
    print("Shape of Unpickled training image object", np.shape(training_image_np))


    print("Starting unpickling testing images")
    training_label_gzip = gzip.open(os.path.join(projectrootfolder, "data", "train_label_pkl_gz.gz"), mode="rb")
    training_label_pkl = pickle.load(training_label_gzip)
    training_label = np.array(training_label_pkl)
    print("Shape of unpickled training label object", np.shape(training_label))

    print("Starting pickling of re-dimensioned training dataset")
    training_dataset = (training_image, training_label)
    #gzip training re-dimensioned training data set.
    training_dataset_redimensioned_pkl = gzip.open(os.path.join(projectrootfolder, "data", "training_dataset_redimensioned_pkl_gz.gz"), mode="wb")
    pickle.dump(training_dataset, training_dataset_redimensioned_pkl)
    training_dataset_redimensioned_pkl.close()

    # with gzip.open(os.path.join(projectrootfolder, "data", "training_dataset_redimensioned_pkl_gz.gz"), mode="wb") as pkl_gz:
    #     with open(os.path.join(projectrootfolder, "data", "training_dataset_redimensioned_pkl.pkl"), mode="rb") as pkl:
    #         shutil.copyfileobj(pkl, pkl_gz)


    print("Pickling of re-dimensioned training dataset complete!")



