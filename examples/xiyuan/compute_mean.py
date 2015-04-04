#!/usr/bin/python
import cv2
import caffe.io
import os
import lmdb
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LMDB = os.path.join(CURRENT_DIR, 'db_train')
OUTPUT = os.path.join(CURRENT_DIR, 'mean.npy')
C, H, W = 3, 227, 227

env = lmdb.open(LMDB)
# img_mean = np.zeros((C, H, W))
img_mean = None
count = 0
with env.begin() as ctx:
    cursor = ctx.cursor()
    cursor.first()
    datum = caffe.io.caffe_pb2.Datum()
    while True:
        count += 1
        d_str = cursor.value()
        datum.ParseFromString(d_str)
        img = caffe.io.datum_to_array(datum)

        C, H, W = img.shape
        if img_mean is None:
            img_mean = np.zeros((C, H, W))
        else:
            img_mean += img

        if not cursor.next():
            break

img_mean /= count
np.save(OUTPUT, img_mean)
