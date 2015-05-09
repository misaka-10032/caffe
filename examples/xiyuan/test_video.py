#!/usr/bin/python
import os
import cv2
import caffe
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CASCADE = os.path.join(CURRENT_DIR, '../../data/xiyuan/cascade/haarcascade_frontalface_alt.xml')
NET_MODEL = os.path.join(CURRENT_DIR, 'xiyuan_quick_iter_20000.caffemodel')
NET_DEPLOY = os.path.join(CURRENT_DIR, 'xiyuan_quick_deploy.proto')
MEAN = os.path.join(CURRENT_DIR, 'mean_uni-30x30.npy')

W, H = 30, 30

def main():
    cascade = cv2.CascadeClassifier(CASCADE)
    net = caffe.Net(NET_DEPLOY, NET_MODEL, caffe.TEST)
    mean_img = np.load(MEAN)
    cap = cv2.VideoCapture(0)
    while (True):
        ret, frame = cap.read()
        rects = cascade.detectMultiScale(frame)
        for x, y, w, h in rects:
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (W, H))
            caffe_img = preprocess(face, mean_img)
            input = caffe_img.reshape((1, ) + caffe_img.shape)
            probs = net.forward_all(data=input)['prob'][0]
            predicts = np.argmax(probs.reshape((probs.shape[0], -1)), axis=0).reshape(probs.shape[-2:])
            up = predicts[1] > predicts[0]
            color = (255, 0, 0) if up else (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

def preprocess(cv2_img, mean_caffe_img):
        img = cv2_img.transpose(2, 0, 1)  # h, w, c -> c, h, w
        img = img.astype(np.float32, copy=False)
        img -= mean_caffe_img
        img = img[(2, 1, 0), :, :]
        return img

if __name__ == '__main__':
    main()