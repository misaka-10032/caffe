#!/usr/bin/python
import os
from argparse import ArgumentParser
import cv2
import caffe.io
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


W, H = 30, 30
# W, H = 512, 288

# Make sure mean is of shape (c, h, w)
MEAN_NPY = os.path.join(CURRENT_DIR, 'mean.npy')
MEAN_BIN = os.path.join(CURRENT_DIR, 'mean.binaryproto')

PREFIX_OUT = 'mean_uni'

def main(args):
    size = args.size.split('x')
    assert len(size) == 2
    w, h = int(size[0]), int(size[1])

    in_npy = args.input
    prefix = args.prefix or PREFIX_OUT
    out_npy = os.path.join(CURRENT_DIR, '%s-%dx%d.npy' % (prefix, w, h))
    out_bin = os.path.join(CURRENT_DIR, '%s-%dx%d.binaryproto' % (prefix, w, h))

    gen_mean(h, w, in_npy, out_npy, out_bin)

def gen_mean(h, w, in_npy, out_npy, out_bin):
    m = np.load(in_npy)
    m = m.transpose()

    r, g, b = np.mean(m[0]), np.mean(m[1]), np.mean(m[2])

    np_out = np.zeros((3, h, w))
    np_out[0] = np.array([r] * (w*h)).reshape((h, w))
    np_out[1] = np.array([g] * (w*h)).reshape((h, w))
    np_out[2] = np.array([b] * (w*h)).reshape((h, w))

    bp_out = np_out.reshape((1,) + np_out.shape)
    bp_out = caffe.io.array_to_blobproto(bp_out)
    with open(out_bin, 'wb') as output:
        output.write(bp_out.SerializeToString())

    np.save(out_npy, np_out)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-s', '--size', type=str, default='%dx%d' % (W, H),
                        help='Size of generated mean image. Format: WxH. Default: %dx%d' % (W, H))
    parser.add_argument('-i', '--input', type=str, default=str(MEAN_NPY),
                        help='Input to be generated from.')
    parser.add_argument('-p', '--prefix', type=str, default=None,
                        help='Prefix of the generated mean image')
    args = parser.parse_args()
    main(args)
