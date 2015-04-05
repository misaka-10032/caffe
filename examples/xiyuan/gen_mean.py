#!/usr/bin/python
import os
import cv2
import caffe.io
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


W, H = 30, 30
MEAN_NPY = os.path.join(CURRENT_DIR, 'mean.npy')
MEAN_BIN = os.path.join(CURRENT_DIR, 'mean.binaryproto')

MEAN_NPY_OUT = os.path.join(CURRENT_DIR, 'mean_uni-%dx%d.npy' % (W, H))
MEAN_BIN_OUT = os.path.join(CURRENT_DIR, 'mean_uni-%dx%d.binaryproto' % (W, H))

def main():
    m = np.load(MEAN_NPY)
    m = m.transpose()
    r, g, b = np.mean(m[0]), np.mean(m[1]), np.mean(m[2])

    np_out = np.zeros((3, H, W))
    np_out[0] = np.array([r] * (W*H)).reshape((H, W))
    np_out[1] = np.array([g] * (W*H)).reshape((H, W))
    np_out[2] = np.array([b] * (W*H)).reshape((H, W))

    bp_out = np_out.reshape((1,) + np_out.shape)
    bp_out = caffe.io.array_to_blobproto(bp_out)
    with open(MEAN_BIN_OUT, 'wb') as output:
        output.write(bp_out.SerializeToString())

    np.save(MEAN_NPY_OUT, np_out)


if __name__ == '__main__':
    main()
