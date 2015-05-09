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
CASCADE = os.path.join(CURRENT_DIR, '../../data/cascade/haarcascade_frontalface_alt2.xml')

tar_in_files = ['aflw-images-3.tar.gz', 'aflw-images-2.tar.gz', 'aflw-images-0.tar.gz']
tar_out_name = 'face_aflw'
tar_out_suffix = 'tar.gz'
TAR_OUT_UP = '0-%s-up.%s' % (tar_out_name, tar_out_suffix)
TAR_OUT_DOWN = '1-%s-down.%s' % (tar_out_name, tar_out_suffix)

W, H = 30, 30
ANGLE_MIN = -20  # inclusive
ANGLE_MAX = 30  # exclusive
ANGLE_DELTA = 20

MAX_COUNT = 25000
REPORT_INTERVAL = 5000

SELECT_FACE_ID_FROM_FILE_ID = "select face_id from Faces where file_id='%s'"
SELECT_FACE_RECT_FROM_ID = "select x, y, w, h from FaceRect where face_id='%s'"

YAW_MIN, YAW_MAX = -pi/4, pi/4
SELECT_YAW_GIVEN_ID = "select yaw from FacePose where face_id='%s'"
SELECT_ROLL_GIVEN_ID = "select roll from FacePose where face_id='%s'"
# heads up or down depends on pitch
PITCH_LOW, PITCH_HIGH = -pi/6, pi/18
SELECT_PITCH_GIVEN_ID = "select pitch from FacePose where face_id=%s"

count_up = 0
count_down = 0

@need_tmp_dir(tmp_dir)
def main():
    cascade = cv2.CascadeClassifier(CASCADE)
    tic = time.time()
    global count_up
    global count_down
    count_up = 0
    count_down = 0
    reporter = 0

    for _tar in tar_in_files:
        print _tar
        tar_idx = int(re.match('.*-(\d*)\..*', _tar).group(1))

        with tarfile.open(os.path.join(input_dir, _tar)) as tar_in:
            while True:
                if count_up > MAX_COUNT and count_down > MAX_COUNT:
                    break
                if reporter > REPORT_INTERVAL:
                    toc = time.time()
                    print 'tar_idx, count, time = %d, %d, %d' % (tar_idx, count_up, toc-tic)
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

                    # get angles of the faces so that they look straight
                    cursor.execute(SELECT_ROLL_GIVEN_ID % face_id)
                    roll = cursor.fetchone()
                    if not roll:
                        continue
                    roll = roll[0]
                    roll = int(roll / pi * 180)
                    # angles = xrange(ANGLE_MIN-roll, ANGLE_MAX-roll, ANGLE_DELTA)
                    angles = [-roll]  # rotate it straight

                    # get pitch so as to judge up or down
                    cursor.execute(SELECT_PITCH_GIVEN_ID % face_id)
                    pitch = cursor.fetchone()
                    if not pitch:
                        continue
                    pitch = pitch[0]
                    # not obvious up or down
                    if PITCH_LOW < pitch < PITCH_HIGH:
                        continue

                    up = pitch > 0
                    if up and count_up > MAX_COUNT:
                        continue
                    if not up and count_down > MAX_COUNT:
                        continue

                finally:
                    os.remove(img_file)
                ###

                # patch faces with rotations
                img_out_file = os.path.join(tmp_dir, 'tmp.jpg')
                imgs = oversample_img(img, box, angles, H, W)
                num_imgs = len(imgs)

                # use opencv to capture faces again
                # because later we use opencv for detection
                for _img, idx in zip(imgs, xrange(num_imgs)):
                    ## won't work, cannot find face in face
                    # rects = cascade.detectMultiScale(_img)
                    # if len(rects) <= 0:
                    #     continue

                    # x, y, w, h = rects[0]
                    # face = _img[y:y+h, x:x+w]
                    # face = cv2.resize(face, (W, H))
                    # cv2.imwrite(img_out_file, face)

                    cv2.imwrite(img_out_file, _img)
                    # tar_out_up.add(img_out_file, '%06d_%d.jpg' % (face_id, idx))
                    if up:
                        tar_out_up.add(img_out_file, '%06d_%d.jpg' % (face_id, idx))
                        count_up += 1
                    else:
                        tar_out_down.add(img_out_file, '%06d_%d.jpg' % (face_id, idx))
                        count_down += 1
                    reporter += 1

                    # clean up
                    os.remove(img_out_file)


def show():
    display_imgs_in_tar(os.path.join(output_dir, TAR_OUT_UP))


if __name__ == '__main__':
    tar_out_up = tarfile.open(os.path.join(output_dir, TAR_OUT_UP), 'w')
    tar_out_down = tarfile.open(os.path.join(output_dir, TAR_OUT_DOWN), 'w')
    sqlfile = os.path.join(input_dir, 'aflw.sqlite')
    conn = sqlite3.connect(sqlfile)
    cursor = conn.cursor()

    main()

    conn.close()
    tar_out_up.close()
    tar_out_down.close()

    renamed = '0-%s-up_%s.%s' % (tar_out_name, count_up, tar_out_suffix)
    os.rename(os.path.join(output_dir, TAR_OUT_UP),
              os.path.join(output_dir, renamed))
    renamed = '1-%s-down_%s.%s' % (tar_out_name, count_down, tar_out_suffix)
    os.rename(os.path.join(output_dir, TAR_OUT_DOWN),
              os.path.join(output_dir, renamed))