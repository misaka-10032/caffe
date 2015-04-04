#!/usr/bin/python
import os
import time
import re
import tarfile
import xml.etree.ElementTree as ET
import cv2

from utils import (need_tmp_dir, oversample_img)

TMP_DIR = '/tmp/ilsvrc'
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_DIR = os.path.join(CURRENT_DIR, '../../data/xiyuan/ilsvrc/candidate')
# TODO: modify this section for different cls
INPUTS = [
    (2, 'book', 'book-annotation'),
    (3, 'desk', 'desk-annotation'),
    (4, 'jersey', 'jersey-annotation'),
    (5, 'phone', 'phone-annotation'),
    (6, 'switch', 'switch-annotation'),
    (7, 'toilet_tissue', 'toilet_tissue-annotation'),
    (8, 'wall_clock', 'wall_clock-annotation'),
    (9, 'water_bottle', 'water_bottle-annotation'),
]

# LABEL, TARGET = 2, 'book'
# INPUT_DIR = os.path.join(CURRENT_DIR, '../../data/xiyuan/ilsvrc/book')
# INPUT_TAR_IMAGE = os.path.join(INPUT_DIR, '%s.tar' % TARGET)
# # INPUT_TAR_ANNOTATION = None
# INPUT_TAR_ANNOTATION = os.path.join(INPUT_DIR, '%s-annotation.tar.gz' % TARGET)

OUTPUT_DIR = os.path.join(CURRENT_DIR, '../../data/xiyuan/arranged')
OUTPUT_SUFFIX = 'tar.gz'
# OUTPUT_TAR = os.path.join(OUTPUT_DIR, '%s-%s.%s' % (LABEL, TARGET, OUTPUT_SUFFIX))

W, H = 227, 227
angles = xrange(-20, 30, 10)

MAX_COUNT = 25000
REPORT_INTERVAL = 5000

@need_tmp_dir(TMP_DIR)
def main():
    '''
    Given:
     label(default 2, start from 2),
     image tar,
     annotation tar(xml inside), if None, crop center
     output location
    Output:
     label-output-number.tar.gz
    :return:
    '''

    class open_res:
        def __init__(self, input_tar_image, input_tar_annotation):
            self.input_tar_image = input_tar_image
            self.input_tar_annotation = input_tar_annotation

        def __enter__(self):
            self.input_tar_image = tarfile.open(self.input_tar_image)
            self.input_tar_annotation = tarfile.open(self.input_tar_annotation) if self.input_tar_annotation else None
            self.output_tar = tarfile.open(OUTPUT_TAR, 'w')
            return self.input_tar_image, self.input_tar_annotation, self.output_tar

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.input_tar_image.close()
            if self.input_tar_annotation:
                self.input_tar_annotation.close()
            self.output_tar.close()
            return False

    class extract_res:
        def __init__(self, tar_image, tar_annotation, ti_image, ti_annotation):
            self.tar_image = tar_image
            self.tar_annotation = tar_annotation
            self.ti_image = ti_image
            self.ti_annotation = ti_annotation

        def __enter__(self):
            self.image_file = os.path.join(TMP_DIR, self.ti_image.name)
            self.tar_image.extract(self.ti_image, TMP_DIR)

            if self.tar_annotation and self.ti_annotation:
                self.annotation_file = os.path.join(TMP_DIR, self.ti_annotation.name)
                self.tar_annotation.extract(self.ti_annotation, TMP_DIR)
            else:
                self.annotation_file = None

            return self.image_file, self.annotation_file

        def __exit__(self, exc_type, exc_val, exc_tb):
            os.remove(self.image_file)
            if self.annotation_file:
                os.remove(self.annotation_file)
            return False

    class write_img:
        def __init__(self, img_file, img):
            self.img_file = img_file
            self.img = img

        def __enter__(self):
            cv2.imwrite(self.img_file, self.img)
            return self.img_file

        def __exit__(self, exc_type, exc_val, exc_tb):
            os.remove(self.img_file)
            return False

    for LABEL, TARGET, ANNOTATION in INPUTS:
        print TARGET
        OUTPUT_TAR = os.path.join(OUTPUT_DIR, '%s-%s.%s' % (LABEL, TARGET, OUTPUT_SUFFIX))
        INPUT_TAR_IMAGE = os.path.join(INPUT_DIR, '%s.tar' % TARGET)
        INPUT_TAR_ANNOTATION = os.path.join(INPUT_DIR, '%s.tar.gz' % ANNOTATION) if ANNOTATION else None
        with open_res(INPUT_TAR_IMAGE, INPUT_TAR_ANNOTATION) as (tar_input_image, tar_input_annotation, tar_output):
            count = 0
            reporter = 0
            while True:
                if count > MAX_COUNT:
                    break
                if reporter > REPORT_INTERVAL:
                    reporter -= REPORT_INTERVAL
                    print count

                # prepare resources
                ti_image = tar_input_image.next()
                if not ti_image:
                    break
                m = re.search('^(.*)_(\d*).JPEG$', ti_image.name)
                if not m:
                    continue
                name, idx = m.group(1), int(m.group(2))
                try:
                    ti_annotation = tar_input_annotation.getmember(
                        os.path.join('Annotation', name, '%s_%s.xml' % (name, idx)))
                except KeyError:  # No such member annotation file
                    ti_annotation = None
                except AttributeError:  # tar_input_annotation is None
                    ti_annotation = None

                # extract and process them
                with extract_res(tar_input_image, tar_input_annotation, ti_image, ti_annotation) as\
                        (image_file, annotation_file):
                    img = cv2.imread(image_file)

                    if annotation_file:
                        annotation_tree = ET.parse(annotation_file)
                        bndbox = annotation_tree.getroot()\
                            .find('object')\
                            .find('bndbox')
                        x = int(bndbox.find('xmin').text)
                        y = int(bndbox.find('ymin').text)
                        w = int(bndbox.find('xmax').text) - x
                        h = int(bndbox.find('ymax').text) - y
                        box = (y, x, h, w)
                    else:
                        box = (0, 0) + img.shape[:2]

                    try:
                        imgs = oversample_img(img, box, angles, H, W)
                    except:
                        print '*main'
                        print image_file
                        print annotation_file
                        print box
                        print img.shape
                        # raise

                    l = len(imgs)
                    count += l
                    reporter += l
                    for img, i in zip(imgs, xrange(l)):
                        filename = '%s_%s_%s.jpg' % (name, idx, i)
                        with write_img(os.path.join(TMP_DIR, filename), img) as img_file:
                            tar_output.add(img_file, filename)

        print 'total: %d' % count
        _output = os.path.join(OUTPUT_DIR, '%s-%s-%s.%s' % (LABEL, TARGET, count, OUTPUT_SUFFIX))
        os.rename(OUTPUT_TAR, _output)


if __name__ == '__main__':
    main()