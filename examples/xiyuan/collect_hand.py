#!/usr/bin/python
import os
import time
import re
import tarfile
import cv2
import matlab.engine

from utils import (dist, oversample_img, need_tmp_dir)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_DIR = os.path.join(CURRENT_DIR, '../../data/xiyuan/hand/hand_dataset')
ANN_DIRS = [os.path.join(INPUT_DIR, 'training_dataset', 'training_data', 'annotations'),
            os.path.join(INPUT_DIR, 'validation_dataset', 'validation_data', 'annotations'),
            os.path.join(INPUT_DIR, 'test_dataset', 'test_data', 'annotations')]
IMG_DIRS = [os.path.join(INPUT_DIR, 'training_dataset', 'training_data', 'images'),
            os.path.join(INPUT_DIR, 'validation_dataset', 'validation_data', 'images'),
            os.path.join(INPUT_DIR, 'test_dataset', 'test_data', 'images')]
FOLDERS = ['training_data', 'validation_data', 'test_data']

LABEL = 1
TARGET = 'hand'
OUTPUT_DIR = os.path.join(CURRENT_DIR, '../../data/xiyuan/arranged')
OUTPUT_SUFFIX = 'tar.gz'
OUTPUT_TAR = os.path.join(OUTPUT_DIR, '%s-%s.%s' % (LABEL, TARGET, OUTPUT_SUFFIX))

TMP_DIR = '/tmp/hand'

W, H = 227, 227
angles = xrange(-20, 30, 10)

matlab_eng = matlab.engine.start_matlab()

MAX_COUNT = 25000
INTERVAL = 5000

@need_tmp_dir(TMP_DIR)
def main():
    count = 0
    reporter = 0
    with tarfile.open(OUTPUT_TAR, 'w') as tar:
        for ann_dir, img_dir, folder in zip(ANN_DIRS, IMG_DIRS, FOLDERS):
            for ann_file in os.listdir(ann_dir):
                if count > MAX_COUNT:
                    break
                if reporter > INTERVAL:
                    reporter -= INTERVAL
                    print count

                try:
                    name = re.match('^(.*)\.mat$', ann_file).group(1)
                except AttributeError:
                    continue
                img_file = name + '.jpg'

                boxes = load_annotations(os.path.join(ann_dir, ann_file))
                img = cv2.imread(os.path.join(img_dir, img_file))

                for box in boxes:
                    try:
                        patches = oversample_img(img, box, angles, H, W)
                    except:
                        print '*' * 10 + img_dir + '/' + img_file + '*' * 10
                        print

                    for patch, idx in zip(patches, xrange(len(patches))):
                        count += 1
                        reporter += 1
                        tmp_file = os.path.join(TMP_DIR, 'tmp.jpg')
                        out_name = '%s_%s_%s.jpg' % (folder, name, idx)
                        cv2.imwrite(tmp_file, patch)
                        tar.add(tmp_file, out_name)
                        os.remove(tmp_file)

    new_name = '%s-%s-%d.%s' % (LABEL, TARGET, count, OUTPUT_SUFFIX)
    os.rename(OUTPUT_TAR, os.path.join(OUTPUT_DIR, new_name))


def load_annotations(filename):
    mat = matlab_eng.load(filename, nargout=1)
    anns = []
    for box in mat['boxes']:
        a_y, a_x = box['a'][0][0], box['a'][0][1]
        b_y, b_x = box['b'][0][0], box['b'][0][1]
        c_y, c_x = box['c'][0][0], box['c'][0][1]
        d_y, d_x = box['d'][0][0], box['d'][0][1]
        x = int((a_x + b_x + c_x + d_x) / 4)
        y = int((a_y + b_y + c_y + d_y) / 4)
        d = [dist((a_x, a_y), (b_x, b_y)),
             dist((b_x, b_y), (c_x, c_y)),
             dist((c_x, c_y), (d_x, d_y)),
             dist((d_x, d_y), (a_x, a_y))]
        w = int(max(d))
        h = w
        anns.append((y-h/2, x-w/2, h, w))
    return anns



if __name__ == '__main__':
    main()