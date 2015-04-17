import caffe
import cv2
import numpy as np
import os
import time

from xiyuan.detector import (SlidingWindowDetector, MultiDetector)

CURRENT_DIR = CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

model_file_single = str(os.path.join(os.path.dirname(CURRENT_DIR), 'xiyuan_quick_deploy.proto'))
# model_file_multi = str(os.path.join(os.path.dirname(CURRENT_DIR), 'xiyuan_full_conv-610x343.proto'))
pretrained_file_single = str(os.path.join(os.path.dirname(CURRENT_DIR), 'xiyuan_quick_iter_120000.caffemodel'))
pretrained_file_multi = str(os.path.join(os.path.dirname(CURRENT_DIR), 'xiyuan_full_conv-768x432.caffemodel'))
WINDOW_MEAN = str(os.path.join(os.path.dirname(CURRENT_DIR), 'mean_uni-30x30.npy'))
FRAME_MEAN = str(os.path.join(os.path.dirname(CURRENT_DIR), 'mean_uni-512x288.npy'))
frame_file = str(os.path.join(CURRENT_DIR, 'frame.jpg'))


net_single = caffe.Net(model_file_single, pretrained_file_single, caffe.TEST)
# net_surgery = caffe.Net(model_file_multi, pretrained_file_single, caffe.TEST)
# net_multi = caffe.Net(model_file_multi, pretrained_file_multi, caffe.TEST)

window_mean = np.load(WINDOW_MEAN).astype(np.float32, copy=False)
frame_mean = np.load(FRAME_MEAN).astype(np.float32, copy=False)

detector_sw = SlidingWindowDetector(model_file_single, pretrained_file_single, WINDOW_MEAN)
detector_multi = MultiDetector()
frame = cv2.imread(frame_file)

# t1=time.time(); d,s=detector.detect_windows(frame, [0], thresh=.8); t2=time.time(); t2-t1
# for y, x, h, w in d[0]:
#     cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

img=cv2.imread('/Users/rocky/Research/caffe/data/xiyuan/hand/hand_dataset/test_dataset/test_data/images/VOC2007_3.jpg')
# t1=time.time(); d_test,s_test=detector.detect_windows(img, None, windows=[(0,0,30,30)], thresh=.5); t2=time.time(); t2-t1
