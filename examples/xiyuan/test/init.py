import caffe
import cv2
import numpy as np
import os
import time

from xiyuan.detector import Detector

CURRENT_DIR = CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

model_file_single = str(os.path.join(os.path.dirname(CURRENT_DIR), 'xiyuan_quick_single_deploy.proto'))
model_file_multi = str(os.path.join(os.path.dirname(CURRENT_DIR), 'xiyuan_quick_multi_deploy.proto'))
pretrained_file_single = str(os.path.join(os.path.dirname(CURRENT_DIR), 'xiyuan_quick_lr1_iter_7000.caffemodel'))
pretrained_file_multi = str(os.path.join(os.path.dirname(CURRENT_DIR), 'xiyuan_quick_lr1_iter_7000_multi.caffemodel'))
mean_file = str(os.path.join(os.path.dirname(CURRENT_DIR), 'mean.npy'))
frame_file = str(os.path.join(CURRENT_DIR, 'frame_half.jpg'))


net_single = caffe.Net(model_file_single, pretrained_file_single, caffe.TEST)
net_surgery = caffe.Net(model_file_multi, pretrained_file_single, caffe.TEST)
net_multi = caffe.Net(model_file_multi, pretrained_file_multi, caffe.TEST)

img_mean = np.load(mean_file).astype(np.float32, copy=False)


detector = Detector(model_file_multi, pretrained_file_multi, mean_file)
frame = cv2.imread(frame_file)
