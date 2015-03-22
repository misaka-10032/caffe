#!/usr/bin/python
import os, shutil
import time
import tarfile
import cv2
import random
import numpy as np

from utils import (need_tmp_dir, display_imgs_in_tar)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(CURRENT_DIR, '../../data/xiyuan/classroom')
output_dir = os.path.join(CURRENT_DIR, '../../data/xiyuan')
tmp_dir = '/tmp/classroom'
tar_out_file = 'neg.tar.gz'

W, H = 30, 30
scales = [.4, .6, .8, 1., 1.2, 1.6, 2.4]
n_patches = 40000


@need_tmp_dir(tmp_dir)
def main():
    '''
    generate 40,000 patches out of each image (28 in all) with different scales
    '''
    with tarfile.open(os.path.join(output_dir, tar_out_file), 'w') as tar_out:
        for _file in os.listdir(input_dir):
            if not _file.lower().endswith('.jpg'):
                continue
            tic = time.time()

            name = _file[:-4]
            img = cv2.imread(os.path.join(input_dir, _file))
            h_img, w_img = img.shape[:2]
            imgs = [cv2.resize(img, (int(scale*h_img), int(scale*w_img)))
                    for scale in scales]

            for i in xrange(n_patches):
                scale_idx = random.randint(0, len(scales)-1)
                img = imgs[scale_idx]
                h_img, w_img = img.shape[:2]
                h_start = random.randint(0, h_img - H)
                w_start = random.randint(0, w_img - W)
                roi = img[h_start:h_start+H, w_start:w_start+W, :]
                tmp_file = os.path.join(tmp_dir, 'tmp.jpg')

                cv2.imwrite(tmp_file, roi)
                tar_out.add(tmp_file, '%s_%05d.jpg' % (name, i))
                os.remove(tmp_file)

            toc = time.time()
            print "%s spends %d" % (_file, toc-tic)

        shutil.rmtree(tmp_dir)


def show():
    display_imgs_in_tar(os.path.join(output_dir, tar_out_file))


if __name__ == '__main__':
    main()