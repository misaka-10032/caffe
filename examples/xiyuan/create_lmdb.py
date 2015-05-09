#!/usr/bin/python
import os
import time
import re
import tarfile
import lmdb
import cv2

from utils import (need_tmp_dir, cv2_img_to_datum)

TMP_DIR = '/tmp/lmdb'
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(CURRENT_DIR, '../../data/xiyuan/arranged')
INPUT_SUFFIX = 'tar.gz'

RATIO_TRAIN = 0.8  # TODO: not used
ITER_TOTAL = 1492  # TODO: adjust this BY HAND
ITER_TRAIN = int(ITER_TOTAL * RATIO_TRAIN)
REPORT_INTERVAL = 500  # this is # samples, not iterations

DB_TRAIN_NAME = 'db_train'
DB_TEST_NAME = 'db_test'

DB_KEY_FORMAT = '%07d'

INPUT_TARS, LABELS = [], []
for _file in os.listdir(INPUT_DIR):
    m = re.match('^(\d*)-.*\.%s$' % INPUT_SUFFIX, _file)
    if not m:
        continue
    label = int(m.group(1))
    INPUT_TARS.append(os.path.join(INPUT_DIR, _file))
    LABELS.append(label)
NUM_LABELS = len(INPUT_TARS)


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
    class open_tars:
        def __enter__(self):
            self.tars = []
            for INPUT_TAR in INPUT_TARS:
                self.tars.append(tarfile.open(INPUT_TAR))
            return self.tars

        def __exit__(self, exc_type, exc_val, exc_tb):
            for tar in self.tars:
                tar.close()
            return False

    class open_db:
        def __init__(self, db_loc, map_size):
            self.db_loc = db_loc
            self.map_size = map_size

        def __enter__(self):
            self.env = lmdb.open(self.db_loc, map_size=self.map_size)
            self.txn = self.env.begin(write=True)
            return self.txn

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.txn.commit()
            self.env.close()
            return False

    with open_tars() as tars:
        # construct db_train
        count = 0  # it counts iterations, rather than samples
        steps_to_report = 0
        tic_report = time.time()
        tic = time.time()
        print 'Creating db_train...'
        with open_db(os.path.join(CURRENT_DIR, DB_TRAIN_NAME), 40000000000) as txn:  # !!40G
            for iter in xrange(ITER_TRAIN):
                if steps_to_report >= REPORT_INTERVAL:
                    steps_to_report -= REPORT_INTERVAL
                    toc_report = time.time()
                    print 'Processed %d; time spent: %ds' % (REPORT_INTERVAL * int(count / REPORT_INTERVAL),
                                                             toc_report - tic_report)
                    tic_report = time.time()

                for tar, label in zip(tars, LABELS):
                    img = read_next_img_in_tar(tar)
                    count += 1
                    datum = cv2_img_to_datum(img, label)
                    txn.put(DB_KEY_FORMAT % count, datum.SerializeToString())

                steps_to_report += NUM_LABELS


        print '*' * 50
        toc = time.time()
        print 'Creating db_train takes %ds' % (toc - tic)
        print '*' * 50
        print ''

        # construct db_test
        count = 0
        steps_to_report = 0
        tic_report = time.time()
        tic = time.time()
        print 'Creating db_test...'
        with open_db(os.path.join(CURRENT_DIR, DB_TEST_NAME), 10000000000) as txn:  #!!10G
            while True:
                should_terminate = False

                for tar, label in zip(tars, LABELS):
                    img = read_next_img_in_tar(tar)
                    if img is None:  # terminal when any one of training set reaches end
                        should_terminate = True
                        break
                    count += 1
                    datum = cv2_img_to_datum(img, label)
                    txn.put(DB_KEY_FORMAT % count, datum.SerializeToString())

                if should_terminate:
                    break

                steps_to_report += NUM_LABELS
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


if __name__ == '__main__':
    main()