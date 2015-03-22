#!/usr/bin/python
import os
import time
import tarfile
import lmdb
import cv2

from utils import (need_tmp_dir, cv2_img_to_datum)

TMP_DIR = '/tmp/lmdb'
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
POS_TAR = os.path.join(CURRENT_DIR, '../../data/xiyuan/pos.tar.gz')
NEG_TAR = os.path.join(CURRENT_DIR, '../../data/xiyuan/neg.tar.gz')
NUM_POS = 189720
NUM_NEG = 1120000
REPORT_INTERVAL = 20000
LABEL_POS = 0  # starts from 0
LABEL_NEG = 1
DB_TRAIN_NAME = 'db_train'
DB_TEST_NAME = 'db_test'
RATIO_TRAIN = 0.8
NUM_NEG_AFTER_POS = 5
DB_KEY_FORMAT = '%07d'

# pos_files = []
# for _file in os.listdir(os.path.join(POS_DIR)):
#     if _file.lower().endswith('.jpg'):
#         pos_files.append(_file)
# n_pos_train = int((len(pos_files) * RATIO_TRAIN))
# pos_files_train = pos_files[:n_pos_train]
# pos_files_test = pos_files[n_pos_train:]

# neg_files = []
# for _file in os.listdir(os.path.join(NEG_DIR)):
#     if _file.lower().endswith('.jpg'):
#         neg_files.append(_file)


def read_next_img_in_tar(tar):
    ti = tar.next()
    if ti is None:
        return None
    tmp_file = os.path.join(TMP_DIR, ti.name)
    tar.extract(ti, TMP_DIR)
    img = cv2.imread(tmp_file)
    os.remove(tmp_file)
    return img


@need_tmp_dir(TMP_DIR)
def main():
    tar_pos = tarfile.open(POS_TAR)
    tar_neg = tarfile.open(NEG_TAR)
    num_pos = NUM_POS or len(tar_pos.getmembers())
    num_neg = NUM_NEG or len(tar_neg.getmembers())
    assert num_neg >= num_pos * NUM_NEG_AFTER_POS
    num_pos_train = int(num_pos * RATIO_TRAIN)

    count = 0
    steps_to_report = 0
    tic_report = time.time()
    tic = time.time()
    print 'Creating db_train...'
    with lmdb.open(os.path.join(CURRENT_DIR, DB_TRAIN_NAME), map_size=3900000000) as env:  # ~3900M
        with env.begin(write=True) as txn:
            for pos_idx in xrange(num_pos_train):
                count += 1
                img = read_next_img_in_tar(tar_pos)
                datum = cv2_img_to_datum(img, LABEL_POS)  # 0 is pos
                txn.put(DB_KEY_FORMAT % count, datum.SerializeToString())

                for i in xrange(NUM_NEG_AFTER_POS):
                    count += 1
                    img = read_next_img_in_tar(tar_neg)
                    datum = cv2_img_to_datum(img, LABEL_NEG)  # 1 is neg
                    txn.put(DB_KEY_FORMAT % count, datum.SerializeToString())

                steps_to_report += 1 + NUM_NEG_AFTER_POS
                if steps_to_report >= REPORT_INTERVAL:
                    steps_to_report -= REPORT_INTERVAL
                    toc_report = time.time()
                    print 'Processed %d; time spent: %ds' % (REPORT_INTERVAL * int(count / REPORT_INTERVAL),
                                                             toc_report - tic_report)
                    tic_report = time.time()

    print '*' * 50
    toc = time.time()
    print 'Creating db_train takes %ds' % (toc - tic)
    print '*' * 50
    print ''

    count = 0
    tic = time.time()
    print 'Creating db_test...'
    with lmdb.open(os.path.join(CURRENT_DIR, DB_TEST_NAME), map_size=975000000) as env:  # ~975M
        with env.begin(write=True) as txn:
            while True:  # loop till end
                count += 1
                img = read_next_img_in_tar(tar_pos)
                if img is None:
                    break
                datum = cv2_img_to_datum(img, LABEL_POS)
                txn.put(DB_KEY_FORMAT % count, datum.SerializeToString())

                for i in xrange(NUM_NEG_AFTER_POS):
                    count += 1
                    img = read_next_img_in_tar(tar_neg)
                    datum = cv2_img_to_datum(img, LABEL_NEG)
                    txn.put(DB_KEY_FORMAT % count, datum.SerializeToString())

                steps_to_report += 1 + NUM_NEG_AFTER_POS
                if steps_to_report >= REPORT_INTERVAL:
                    steps_to_report -= REPORT_INTERVAL
                    toc_report = time.time()
                    print 'Processed %d; time spent: %ds' % (REPORT_INTERVAL * int(count / REPORT_INTERVAL),
                                                             toc_report - tic_report)
                    tic_report = time.time()

    print '*' * 50
    toc = time.time()
    print 'Creating db_test takes %ds' % (toc - tic)
    print '*' * 50
    print ''

    tar_pos.close()
    tar_neg.close()


if __name__ == '__main__':
    main()