#!/usr/bin/env python
# encoding: utf-8
"""
Created by misaka-10032 (longqic@andrew.cmu.edu).
All rights reserved.

TODO: purpose
"""

import argparse
import os
import numpy as np
from scipy.misc import imread, imrotate, imsave
import svg

start_idx = 0
max_idx = 20000     # max # of synthesized images
range_in = range(64)
w, h, c = 800, 800, 3            # patch size
ws, hs = 400, 400   # stride
fmt_in_png = 'final_{}.png'
fmt_out_png = '{:0%d}.png' % len(str(max_idx))
fmt_out_ppm = '{:0%d}.ppm' % len(str(max_idx))


def main(args):
    # sanity check
    assert os.path.isdir(os.path.join(args.dir, 'bg')), \
        'dir must contain bg/ as input'
    assert os.path.isdir(os.path.join(args.dir, 'lines-svg')), \
        'dir must contain lines-svg/ as input'
    assert os.path.isdir(os.path.join(args.dir, 'lines-png')), \
        'dir must contain lines-png/ as input'
    assert os.path.isdir(os.path.join(args.dir, 'tiles')), \
        'dir must contain tiles/ as output'


    idx = start_idx
    for i in range_in:
        print 'Processing final_{}...'.format(i)
        if idx > max_idx:
            print 'Terminates at final_{}'.format(i)
        img_bg = imread(os.path.join(args.dir, 'bg', fmt_in_png.format(i)), mode='RGBA')
        img_ln = imread(os.path.join(args.dir, 'lines-png', fmt_in_png.format(i)), mode='RGBA')[:, :, 3]
        H, W, _ = img_bg.shape
        for h_start in xrange(0, H, hs):
            for w_start in xrange(0, W, ws):
                if idx > max_idx:
                    break
                bg_small = img_bg[h_start:h_start+h, w_start:w_start+w]
                ln_small = img_ln[h_start:h_start+h, w_start:w_start+w]

                fname = fmt_out_png.format(idx)
                imsave(os.path.join(args.dir, 'tiles', fname), bg_small)
                fname = fmt_out_ppm.format(idx)
                imsave(os.path.join(args.dir, 'tiles', fname), ln_small)
                idx += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('arg', help='...')
    # parser.add_argument('-o', '--optional', action='store_true', help='...')
    parser.add_argument('-d', '--dir',
                        type=str, default='../data/fp2',
                        help='dir which contains bg/, lines-svg/, '
                             'lines-png,/ and names.txt as input;'
                             'also empty tiles/ as output')
    main(parser.parse_args())
