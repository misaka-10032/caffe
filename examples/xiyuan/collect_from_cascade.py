#!/usr/bin/python
import os
import time
import re
import tarfile
import cv2
import numpy as np

from utils import (oversample_img, need_tmp_dir, put_imgs_to_tar)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CASCADE = os.path.join(CURRENT_DIR, '../../data/xiyuan/cascade/haarcascade_frontalface_alt.xml')
INPUT_DIR = os.path.join(CURRENT_DIR, '../../data/xiyuan/class')
INPUT_VIDEO = os.path.join(INPUT_DIR, '8.mp4')

LABEL = 14
TARGET = 'face_opencv'
OUTPUT_DIR = os.path.join(CURRENT_DIR, '../../data/xiyuan/arranged')
OUTPUT_SUFFIX = 'tar.gz'
OUTPUT_TAR = os.path.join(OUTPUT_DIR, '%s-%s.%s' % (LABEL, TARGET, OUTPUT_SUFFIX))
MAX_COUNT = 50000
INTERVAL = 5000

TMP_DIR = '/tmp/cascade'

C, H, W = 3, 30, 30
angles = xrange(-20, 30, 10)


@need_tmp_dir(TMP_DIR)
def main():
    count = 0
    reporter = 0
    cascade = cv2.CascadeClassifier(CASCADE)
    cap = cv2.VideoCapture(INPUT_VIDEO)

    with tarfile.open(OUTPUT_TAR, 'w') as tar:
        while cap.isOpened():
            should_break = False
            ret, frame = cap.read()
            faces = cascade.detectMultiScale(frame)
            for x, y, w, h in faces:
                try:
                    imgs = oversample_img(frame, (y, x, h, w),
                                          angles, H, W)
                    put_imgs_to_tar(imgs, tar, '%06d' % count, TMP_DIR)
                    count += len(imgs)
                    reporter += len(imgs)

                    if count > MAX_COUNT:
                        should_break = True

                    if reporter > INTERVAL:
                        print count
                        reporter -= INTERVAL

                except:
                    pass
            if should_break:
                break

    renamed = os.path.join(OUTPUT_DIR, '%s-%s-%d.%s' % (LABEL, TARGET, count, OUTPUT_SUFFIX))
    os.rename(OUTPUT_TAR, renamed)


if __name__ == '__main__':
    main()