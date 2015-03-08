import caffe
import cv2
import numpy as np

from xiyuan.detector import Detector

model_file = '/Users/rocky/Research/caffe/examples/xiyuan/cifar10_quick.prototxt'
pretrained_file = '/Users/rocky/Research/caffe/examples/xiyuan/cifar10_quick_iter_2000.caffemodel'
mean_file = '/Users/rocky/Research/caffe/examples/xiyuan/mean.npy'
img_file = '/Users/rocky/Research/opencv-haar-classifier-training/positive_images/Subject01_A_01_-10.jpg'
frame_file = '/Users/rocky/Research/caffe/examples/xiyuan/test/frame.jpg'

net = caffe.Net(model_file, pretrained_file, caffe.TEST)
img_mean = np.load(mean_file).astype(np.float32, copy=False)
img = cv2.imread(img_file)
caffe_img = img.transpose(2, 1, 0).astype(np.float32, copy=False)
caffe_img -= img_mean

detector = Detector(model_file, pretrained_file, mean_file)
frame = cv2.imread(frame_file)
