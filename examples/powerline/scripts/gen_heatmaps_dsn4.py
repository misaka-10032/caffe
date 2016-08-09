#!/usr/bin/env python
# encoding: utf-8
"""
Created by misaka-10032 (longqic@andrew.cmu.edu).
All rights reserved.

TODO: purpose
"""

import argparse
import os
import time
import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../../../python')
import caffe

fmt_in = 'final_{}.png'
n_imgs = 64
w, h, c = 800, 800, 3  # patch size
fmt_hmap = 'hmap_{}.png'
fmt_lmap = 'lmap_{}.npz'


def main(args):
    t_all, t_fwd, t_io = 0, 0, 0
    t_all1 = time.time()

    caffe.set_mode_gpu()
    caffe.set_device(0)
    net = caffe.Net(args.model_def, args.model_weight, caffe.TEST)
    for i in xrange(n_imgs):
        fname = fmt_in.format(i)
        path = os.path.join(args.dir_in, fname)

        t_io1 = time.time()
        img = imread(path, mode='RGB')
        t_io2 = time.time()
        t_io += t_io2 - t_io1

        H, W, _ = img.shape
        assert H%h==0 and W%w==0, \
            'patch size must be divisor of img size'
        nh, nw = H/h, W/w
        patches = img[:, :, ::-1] - \
                  np.array([108.353, 115.294, 114.578])
        patches = patches.reshape(nh, h, nw, w, -1).transpose([0, 2, 4, 1, 3])

        t1 = time.time()
        probs = np.zeros([nh, nw, h, w], dtype=float)
        labels = np.zeros([nh, nw, h, w], dtype=int)
        for ho in xrange(nh):
            for wo in xrange(nw):
                net.blobs['data'].reshape(1, c, h, w)
                net.blobs['data'].data[...] = patches[ho, wo, ...]
                net.forward()
                probs[ho, wo] = net.blobs['sigmoid-dsn4'].data[0]
                labels[ho, wo] = probs[ho, wo] > .5
        t2 = time.time()
        t = t2 - t1
        t_fwd += t
        print '[{}] forward takes {}s.'.format(fname, t)

        probs = probs.transpose([0, 2, 1, 3]).reshape([H, W])
        fname = fmt_hmap.format(i)
        path = os.path.join(args.dir_out, fname)
        t_io1 = time.time()
        plt.imsave(path, probs)
        t_io2 = time.time()
        t_io += t_io2 - t_io1

        labels = labels.transpose([0, 2, 1, 3]).reshape([H, W])
        labels = labels.astype(bool)
        fname = fmt_lmap.format(i)
        path = os.path.join(args.dir_out, fname)
        t_io1 = time.time()
        np.savez_compressed(path, labels=labels)
        t_io2 = time.time()
        t_io += t_io2 - t_io1

    t_all2 = time.time()
    t_all = t_all2 - t_all1
    t_oth = t_all - t_fwd - t_io
    print
    print '-- Summary'
    print 'fwd takes:  \t{0:.2f}s'.format(t_fwd)
    print 'io takes:   \t{0:.2f}s'.format(t_io)
    print 'others take:\t{0:.2f}s'.format(t_oth)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('arg', help='...')
    # parser.add_argument('-o', '--optional', action='store_true', help='...')
    parser.add_argument('-din', '--dir_in',
                        type=str, default='../eval/bg',
                        help='dir of input maps')
    parser.add_argument('-dhmap', '--dir_out',
                        type=str, default='../eval/pred',
                        help='dir of output hmaps and lmaps')
    # parser.add_argument('-doutsm', '--dir_out_small',
    #                     type=str, default='../results-1k',
    #                     help='dir of output heat maps')
    parser.add_argument('-md', '--model_def',
                        type=str, default='../models/hough_v8.deploy',
                        help='caffe model deploy file')
    parser.add_argument('-mw', '--model_weight',
                        type=str, default='../models/hough_v8.caffemodel',
                        help='caffe model deploy file')
    main(parser.parse_args())

