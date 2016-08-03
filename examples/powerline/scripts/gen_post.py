#!/usr/bin/env python
# encoding: utf-8

import os
import numpy as np
from scipy.misc import imread, imsave
import matplotlib.pyplot as plt
import cv2
from hough import Hough

import sys
sys.path.insert(0, '../../../python')
import caffe

root = '../eval'
range_in = range(64)
fmt_hmap = 'hmap_{}.png'
fmt_lmap = 'lmap_{}.npz'

H = W = 400             # post-processing window size
H_STEP = W_STEP = 200   # post-processing window step
THETA, RHO = 180, 240   # hough transform range
HH = WW = 8000          # size of the entire image

thresh_bp = .1  # back projection threshold


def filter_peaks(hftn, sp_theta=8, sp_rho=4, sp_thresh=.003, sp_avg=.001):
    hftn = hftn.copy()
    THETA, RHO = hftn.shape
    peaks = np.zeros_like(hftn)
    while True:
        idx = np.argmax(hftn)
        theta, rho = idx/RHO, idx%RHO
        theta_min = max(0, int(theta-sp_theta/2))
        theta_max = min(THETA, int(theta+sp_theta/2))
        rho_min = max(0, int(rho-sp_rho/2))
        rho_max = min(RHO, int(rho+sp_rho/2))

        #print hftn[theta, rho]
        #sys.stdout.flush()

        if hftn[theta, rho] < sp_thresh:
            break
        if np.sum(hftn[theta_min:theta_max, rho_min:rho_max]) < sp_avg*sp_theta*sp_rho:
            break
        hftn[theta_min:theta_max, rho_min:rho_max] = 0.
        peaks[theta_min:theta_max, rho_min:rho_max] = 1.
    return peaks


def backpj(hb, hftn):
    r = hb.backward(filter_peaks(hftn))
    if r.max() > 0:
        r /= r.max()
    return r


def main():
    assert os.path.isdir(os.path.join(root, 'pred'))
    assert os.path.isdir(os.path.join(root, 'post'))
    hb = Hough(H, W, THETA, RHO)

    for i in range_in:
        print 'Processing {}...'.format(i)
        fpath = os.path.join(root, 'pred', fmt_hmap.format(i))
        hmap_pre = imread(fpath)[:, :, 0].astype(float) / 255.
        hmap_post = np.zeros_like(hmap_pre)
        for h_start in xrange(0, HH-H+1, H_STEP):
            for w_start in xrange(0, WW-W+1, W_STEP):
                pre_small = hmap_pre[h_start:h_start+H, w_start:w_start+W]
                post_small = hmap_post[h_start:h_start+H, w_start:w_start+W]
                hftn = hb.forward(pre_small)
                post_small += backpj(hb, hftn)
        lmap_post = hmap_post > thresh_bp
        # TODO: don't use hard threshold?
        hmap_post = lmap_post
        fpath = os.path.join(root, 'post', fmt_hmap.format(i))
        plt.imsave(fpath, hmap_post)
        fpath = os.path.join(root, 'post', fmt_lmap.format(i))
        np.savez_compressed(fpath, labels=lmap_post)


if __name__ == '__main__':
    main()

