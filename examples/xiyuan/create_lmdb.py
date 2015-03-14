#!/usr/bin/python
import os
import lmdb
import cv2
import caffe.io

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
POS_DIR = '../../data/xiyuan/positive_images.extended'
NEG_DIR = '../../data/xiyuan/negative_images.extended'
DB_TRAIN_NAME = 'db_train'
DB_TEST_NAME = 'db_test'
P_TRAIN = 0.8
NUM_NEG_AFTER_POS = 2
# #pos, #neg ~= 30000, 80000
DB_KEY_FORMAT = '%06d'

pos_files = []
for _file in os.listdir(os.path.join(CURRENT_DIR, POS_DIR)):
    if _file.lower().endswith('.jpg'):
        pos_files.append(_file)
n_pos_train = int((len(pos_files) * P_TRAIN))
pos_files_train = pos_files[:n_pos_train]
pos_files_test = pos_files[n_pos_train:]

neg_files = []
for _file in os.listdir(os.path.join(CURRENT_DIR, NEG_DIR)):
    if _file.lower().endswith('.jpg'):
        neg_files.append(_file)

count = 0
env = lmdb.open(os.path.join(CURRENT_DIR, DB_TRAIN_NAME), map_size=800000000)  # ~800M
with env.begin(write=True) as txn:
    for pos in pos_files_train:
        count += 1
        img = cv2.imread(os.path.join(CURRENT_DIR, POS_DIR, pos))
        img = img.transpose(2, 1, 0)
        datum = caffe.io.array_to_datum(img, 0)  # 0 is pos
        # 5 bits should be enough, at most 4000 pos's and 40000 neg's
        txn.put(DB_KEY_FORMAT % count, datum.SerializeToString())
        for i in xrange(NUM_NEG_AFTER_POS):
            count += 1
            neg = neg_files.pop()
            img = cv2.imread(os.path.join(CURRENT_DIR, NEG_DIR, neg))
            img = img.transpose(2, 1, 0)
            datum = caffe.io.array_to_datum(img, 1)  # 1 is neg
            txn.put(DB_KEY_FORMAT % count, datum.SerializeToString())
env.close()

count = 0
env = lmdb.open(os.path.join(CURRENT_DIR, DB_TEST_NAME), map_size=200000000)  # ~200M
with env.begin(write=True) as txn:
    for pos in pos_files_test:
        count += 1
        img = cv2.imread(os.path.join(CURRENT_DIR, POS_DIR, pos))
        img = img.transpose(2, 1, 0)
        datum = caffe.io.array_to_datum(img, 0)
        txn.put(DB_KEY_FORMAT % count, datum.SerializeToString())
        for i in xrange(NUM_NEG_AFTER_POS):
            count += 1
            neg = neg_files.pop()
            img = cv2.imread(os.path.join(CURRENT_DIR, NEG_DIR, neg))
            img = img.transpose(2, 1, 0)
            datum = caffe.io.array_to_datum(img, 1)
            txn.put(DB_KEY_FORMAT % count, datum.SerializeToString())
env.close()
