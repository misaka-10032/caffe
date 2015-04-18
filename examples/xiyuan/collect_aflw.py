#!/usr/bin/python
import os
import shutil
import time
import cv2
from math import pi
import tarfile
import re
import sqlite3

from utils import (need_tmp_dir, oversample_img, display_imgs_in_tar)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

input_dir = os.path.join(CURRENT_DIR, '../../data/aflw')
output_dir = os.path.join(CURRENT_DIR, '../../data/xiyuan/arranged')
tmp_dir = '/tmp/aflw'
res_prefix = 'aflw/data/flickr'

tar_in_files = ['aflw-images-3.tar.gz', 'aflw-images-2.tar.gz', 'aflw-images-0.tar.gz']
tar_out_name = '0-face_aflw_-45_to_45'
tar_out_suffix = 'tar.gz'
tar_out_file = '%s.%s' % (tar_out_name, tar_out_suffix)

W, H = 30, 30
ANGLE_MIN = -20  # inclusive
ANGLE_MAX = 30  # exclusive
ANGLE_DELTA = 20

# MAX_COUNT = 25000
REPORT_INTERVAL = 5000

SELECT_FACE_ID_FROM_FILE_ID = "select face_id from Faces where file_id='%s'"
SELECT_FACE_RECT_FROM_ID = "select x, y, w, h from FaceRect where face_id='%s'"

YAW_MIN, YAW_MAX = -pi/4, pi/4
SELECT_YAW_GIVEN_ID = "select yaw from FacePose where face_id='%s'"
SELECT_ROLL_GIVEN_ID = "select roll from FacePose where face_id='%s'"

count = 0

@need_tmp_dir(tmp_dir)
def main():
    tic = time.time()
    global count
    count = 0
    reporter = 0

    for _tar in tar_in_files:
        print _tar
        tar_idx = int(re.match('.*-(\d*)\..*', _tar).group(1))

        with tarfile.open(os.path.join(input_dir, _tar)) as tar_in:
            while True:
                # if count > MAX_COUNT:
                #     break
                if reporter > REPORT_INTERVAL:
                    toc = time.time()
                    print 'tar_idx, count, time = %d, %d, %d' % (tar_idx, count, toc-tic)
                    tic = time.time()
                    reporter -= REPORT_INTERVAL

                ### sanity check of filename
                ti = tar_in.next()
                if not ti:
                    break

                # get file_id
                file_id = None
                try:
                    m = re.match('.*/(.*\.jpg)', ti.name)
                    file_id = m.group(1)
                except IndexError, e:
                    continue
                except AttributeError, e:
                    continue
                if file_id is None:
                    continue
                ###

                # extract to /tmp and read image
                tar_in.extract(ti, tmp_dir)
                img_file = os.path.join(tmp_dir, res_prefix, str(tar_idx), file_id)
                img = cv2.imread(img_file)

                ### sanity check of the extracted image
                try:
                    # get face_id
                    cursor.execute(SELECT_FACE_ID_FROM_FILE_ID % file_id)
                    face_id = cursor.fetchone()
                    if face_id is None:
                        continue
                    face_id = face_id[0]  # it's a tuple, so get the first element

                    # filter face with yaw
                    cursor.execute(SELECT_YAW_GIVEN_ID % face_id)
                    yaw = cursor.fetchone()
                    if yaw is None:
                        continue
                    yaw = yaw[0]
                    if yaw < YAW_MIN or yaw > YAW_MAX:
                        continue

                    # get face rect
                    cursor.execute(SELECT_FACE_RECT_FROM_ID % face_id)
                    rect = cursor.fetchone()
                    if rect is None:
                        continue
                    x, y, w, h = rect
                    box = (y, x, h, w)

                    cursor.execute(SELECT_ROLL_GIVEN_ID % face_id)
                    roll = cursor.fetchone()
                    if not roll:
                        continue
                    roll = roll[0]
                    roll = int(roll / pi * 180)
                    angles = xrange(ANGLE_MIN-roll, ANGLE_MAX-roll, ANGLE_DELTA)

                finally:
                    os.remove(img_file)
                ###

                # patch faces with rotations
                img_out_file = os.path.join(tmp_dir, 'tmp.jpg')
                imgs = oversample_img(img, box, angles, H, W)
                num_imgs = len(imgs)
                for img_it, idx in zip(imgs, xrange(num_imgs)):
                    cv2.imwrite(img_out_file, img_it)
                    tar_out.add(img_out_file, '%06d_%d.jpg' % (face_id, idx))

                # update counter
                count += num_imgs
                reporter += num_imgs

                # clean up
                os.remove(img_out_file)



def show():
    display_imgs_in_tar(os.path.join(output_dir, tar_out_file))


if __name__ == '__main__':
    tar_out = tarfile.open(os.path.join(output_dir, tar_out_file), 'w')
    sqlfile = os.path.join(input_dir, 'aflw.sqlite')
    conn = sqlite3.connect(sqlfile)
    cursor = conn.cursor()

    main()

    conn.close()
    tar_out.close()

    renamed = '%s_%s.%s' % (tar_out_name, count, tar_out_suffix)
    os.rename(os.path.join(output_dir, tar_out_file),
              os.path.join(output_dir, renamed))