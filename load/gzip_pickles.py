import gzip
import os
import pathlib
import shutil

with open(os.path.join(pathlib.PurePath(), "../data", "train_image_pkl"), mode="rb") as pkl:
    with gzip.open(os.path.join(pathlib.PurePath(), "../data", "train_image_pkl_gz.gz"), mode="wb") as pkl_gzip:
        shutil.copyfileobj(pkl,pkl_gzip)

pkl.close()
pkl_gzip.close()