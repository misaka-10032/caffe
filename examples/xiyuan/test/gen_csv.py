#!/usr/bin/python
import cv2
import pandas as pd
from pandas import DataFrame
import caffe

import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
#INPUT = os.path.join(CURRENT_DIR, 'gray.jpg')
INPUT = os.path.join(CURRENT_DIR, '../../../data/xiyuan/positive_images/Subject01_B_01_0.jpg')
OUTPUT = os.path.join(CURRENT_DIR, 'win.csv')

STRIDE = 10
H_I, W_I = 50, 50  # H_INPUT
Hs = [30, 40, 50, 60, 70, 80, 90, 100]  # H_WINDOW

pdata = {}
filenames, ymin, xmin, ymax, xmax = [], [], [], [], []

img = cv2.imread(INPUT)
h, w = img.shape[:2]
for H in Hs:
    W = H
    for h_start in xrange(0, h-H+1, H):
        for w_start in xrange(0, w-W+1, W):
            ymin.append(h_start)
            xmin.append(w_start)
            ymax.append(h_start+H)
            xmax.append(w_start+W)
            filenames.append(INPUT)
pdata['filename'] = filenames
pdata['ymin'] = ymin
pdata['xmin'] = xmin
pdata['ymax'] = ymax
pdata['xmax'] = xmax

df = DataFrame(pdata, columns=['filename', 'ymin', 'xmin', 'ymax', 'xmax'])
df.set_index('filename', inplace=True)
df.to_csv(OUTPUT, sep=',')
