#!/usr/bin/python
import os
import cv2
import caffe
import numpy as np
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CASCADE = os.path.join(CURRENT_DIR, '../../data/xiyuan/cascade/haarcascade_frontalface_alt.xml')
NET_MODEL = os.path.join(CURRENT_DIR, 'xiyuan_quick_iter_20000.caffemodel')
NET_DEPLOY = os.path.join(CURRENT_DIR, 'xiyuan_quick_deploy.proto')
MEAN = os.path.join(CURRENT_DIR, 'mean_uni-30x30.npy')
VIDEO = os.path.join(CURRENT_DIR, '../../data/xiyuan/class/8.mp4')

W_WIN, H_WIN = 30, 30
W_FRAME, H_FRAME = 512, 288

def main():
    cascade = cv2.CascadeClassifier(CASCADE)
    net = caffe.Net(NET_DEPLOY, NET_MODEL, caffe.TEST)
    mean_img = np.load(MEAN)
    cap = cv2.VideoCapture(VIDEO)
    while (True):
        ret, frame = cap.read()
        # frame = cv2.resize(frame, (W_FRAME, H_FRAME))

        tic = time.time()
        rects = cascade.detectMultiScale(frame)
        toc = time.time()
        # print 'cascade spends %ds' % (toc - tic)

        for x, y, w, h in rects:
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (W_WIN, H_WIN))

            tic = time.time()
            caffe_img = preprocess(face, mean_img)
            input = caffe_img.reshape((1, ) + caffe_img.shape)
            probs = net.forward_all(data=input)['prob'][0]
            toc = time.time()
            # print 'caffe spends %ds' % (toc - tic)

            up = probs[1, 0, 0] > 0.9
            down = probs[0, 0, 0] > 0.99
            if up:
                color = (255, 0, 0)
            elif down:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        cv2.imshow("test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def preprocess(cv2_img, mean_caffe_img):
        img = cv2_img.transpose(2, 0, 1)  # h, w, c -> c, h, w
        img = img.astype(np.float32, copy=False)
        img -= mean_caffe_img
        img = img[(2, 1, 0), :, :]
        return img

if __name__ == '__main__':
    main()