#!/usr/bin/python
import os
import time
import re
import tarfile
import xml.etree.ElementTree as ET
import cv2

from utils import (need_tmp_dir)

TMP_DIR = '/tmp/arranged'
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# TODO
LABEL, TARGET = 22, 'wall_clock'
INPUT_DIR = os.path.join(CURRENT_DIR, '../../data/xiyuan/ilsvrc/clock')
INPUT_TAR_IMAGE = os.path.join(INPUT_DIR, '%s.tar' % TARGET)
# INPUT_TAR_ANNOTATION = None
INPUT_TAR_ANNOTATION = os.path.join(INPUT_DIR, '%s-annotation.tar.gz' % TARGET)

OUTPUT_DIR = os.path.join(CURRENT_DIR, '../../data/xiyuan/ilsvrc_arranged')
OUTPUT_SUFFIX = 'tar.gz'
OUTPUT_TAR = os.path.join(OUTPUT_DIR, '%s-%s.%s' % (LABEL, TARGET, OUTPUT_SUFFIX))

W, H = 30, 30
angles = xrange(-40, 50, 10)

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
        def __init__(self):
            pass

        def __enter__(self):
            self.input_tar_image = tarfile.open(INPUT_TAR_IMAGE)
            self.input_tar_annotation = tarfile.open(INPUT_TAR_ANNOTATION) if INPUT_TAR_ANNOTATION else None
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

    with open_res() as (tar_input_image, tar_input_annotation, tar_output):
        count = 0
        while True:
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
                    imgs = oversample_img(img, box)
                except:
                    print '*main'
                    print image_file
                    print annotation_file
                    print box
                    print img.shape
                    raise

                l = len(imgs)
                count += l
                for img, i in zip(imgs, xrange(l)):
                    filename = '%s_%s_%s.jpg' % (name, idx, i)
                    with write_img(os.path.join(TMP_DIR, filename), img) as img_file:
                        tar_output.add(img_file, filename)

    print 'total: %d' % count
    _output = os.path.join(OUTPUT_DIR, '%s-%s-%s.%s' % (LABEL, TARGET, count, OUTPUT_SUFFIX))
    os.rename(OUTPUT_TAR, _output)


def oversample_img(img, box):
    '''
    :param img:
    :return: list of rotated & cropped (to square) & resized (to unisize) images
    '''
    y, x, h, w = box
    imgs = []
    for angle in angles:
        r = cv2.getRotationMatrix2D((x+w/2, y+h/2), angle, 1.0)
        img_rot = cv2.warpAffine(img, r, (img.shape[1], img.shape[0]))
        target = img_rot[y:y+h, x:x+w]
        try:
            imgs.extend(crop_resize_image(target))
        except:
            print '**oversample_img'
            print box
            print img.shape
            print img_rot.shape
            print target.shape
            raise
    return imgs


def crop_resize_image(img):
    h_whole, w_whole = img.shape[:2]
    if abs(w_whole - h_whole) < 3:
        return [cv2.resize(img, (W, H))]

    imgs = []
    if w_whole > h_whole:
        h_start, h, w = 0, h_whole, h_whole
        # left
        w_start = 0

        try:
            imgs.append(cv2.resize(
                img[h_start:h_start+h, w_start:w_start+w],
                (W, H)))
        except:
            print '***crop_resize_image'
            print img.shape
            print h_start, w_start, h, w
            raise

        # middle
        w_start = (w_whole - w) / 2
        imgs.append(cv2.resize(
            img[h_start:h_start+h, w_start:w_start+w],
            (W, H)))
        # right
        w_start = w_whole - w
        imgs.append(cv2.resize(
            img[h_start:h_start+h, w_start:w_start+w],
            (W, H)))
        # whole
        imgs.append(cv2.resize(img, (W, H)))
    else:
        w_start, h, w = 0, w_whole, w_whole
        # top
        h_start = 0
        imgs.append(cv2.resize(
            img[h_start:h_start+h, w_start:w_start+w],
            (W, H)))
        # middle
        h_start = (h_whole - h) / 2
        imgs.append(cv2.resize(
            img[h_start:h_start+h, w_start:w_start+w],
            (W, H)))
        # bottom
        h_start = h_whole - h
        imgs.append(cv2.resize(
            img[h_start:h_start+h, w_start:w_start+w],
            (W, H)))
        # whole
        imgs.append(cv2.resize(img, (W, H)))

    return imgs


if __name__ == '__main__':
    main()