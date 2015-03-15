import caffe
import cv2
import numpy as np
import time

WINDOW_SIZES = [30, 40, 50, 60, 70, 80]
STRIDE = 4
BATCH_SIZE = 100

class Detector:
    def __init__(self, model_def_file, model_file,
                 mean_file=None, need_equalize=True):
        self.net = caffe.Net(model_def_file, model_file, caffe.TEST)
        self.mean_caffe_img = np.load(mean_file).astype(np.float32, copy=False) if mean_file else 0
        self.need_equalize = need_equalize

        self.t_resize = 0
        self.t_cvtColor = 0
        self.t_equalizeHist = 0
        self.t_transpose = 0
        self.t_forward = 0
        self.t_nms = 0

    def detect(self, cv2_img, labels_of_interest=None, thresh=.9, windows=None, batch_size=BATCH_SIZE):
        '''
        :param cv2_img: shape as (H, W, C)
        :param windows: [(h_start, w_start, h, w), ...]
        :return: {[(), (), ...], ...}, where r[i] is the detection for label i, r[i][j] is the jth window
        '''
        self.detections, self.scores = {}, {}
        labels_of_interest = range(self.net.blobs[self.net.outputs[0]])\
            if labels_of_interest is None else labels_of_interest
        for label in labels_of_interest:
            self.detections[label], self.scores[label] = [], []

        windows = self.gen_windows(cv2_img) if not windows else windows

        for idx_offset in xrange(0, len(windows), batch_size):
            self._detect_batch(cv2_img, labels_of_interest, thresh, windows, idx_offset, batch_size)
        idx_offset += batch_size
        self._detect_batch(cv2_img, labels_of_interest, thresh, windows,
                           idx_offset, len(windows) - idx_offset)

        # non-max-suppression
        # now detections[pred] and scores[pred] changes from list to array
        for label_pred in self.detections:
            self.detections[label_pred], self.scores[label_pred] = \
                np.array(self.detections[label_pred]), np.array(self.scores[label_pred])
            t1=time.time(); self.detections[label_pred], self.scores[label_pred] = \
                self.nms_detections(self.detections[label_pred], self.scores[label_pred]); t2=time.time(); self.t_nms+=t2-t1

        return self.detections, self.scores

    def _gen_patches(self, cv2_img, windows):
        patches = []
        for window in windows:
            h_start, w_start, h, w = window
            roi = cv2_img[h_start:h_start+h, w_start:w_start+w]
            patches.append(self.preprocess(roi))
        return np.array(patches)

    def _detect_batch(self, cv2_img, labels_of_interest, thresh, windows, idx_offset, batch_size):
        if batch_size <= 0:
            return
        patches = self._gen_patches(cv2_img, windows[idx_offset:idx_offset+batch_size])
        t1=time.time(); probs_batch = self.net.forward_all(**{self.net.inputs[0]: patches})['prob'].squeeze((2, 3)); t2=time.time(); self.t_forward+=t2-t1
        idx = idx_offset
        for probs in probs_batch:
            window = windows[idx]; idx += 1
            label_pred = np.argmax(probs)
            score = probs[label_pred]
            if label_pred in labels_of_interest and score > thresh:
                self.detections[label_pred].append(list(window))
                self.scores[label_pred].append(probs[label_pred])

    def nms_detections(self, dets, scores, overlap=0.3):
        """
        Non-maximum suppression: Greedily select high-scoring detections and
        skip detections that are significantly covered by a previously
        selected detection.

        This version is translated from Matlab code by Tomasz Malisiewicz,
        who sped up Pedro Felzenszwalb's code.

        Parameters
        ----------
        dets: array
            //each row is ['xmin', 'ymin', 'xmax', 'ymax', 'score']
            each row is ('ymin', 'xmin', 'height', 'width', 'score')
        scores: array of scores,
            must be in the same shape as dets
        overlap: float
            minimum overlap ratio (0.3 default)

        Output
        ------
        dets: ndarray
            remaining after suppression.
        """
        if len(dets) <= 0:
            return []

        y1, x1, h, w = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
        y2, x2 = y1 + h, x1 + w
        ind = np.argsort(scores)
        area = (w * h).astype(float)

        pick = []
        while len(ind) > 0:
            i = ind[-1]
            pick.append(i)
            ind = ind[:-1]

            xx1 = np.maximum(x1[i], x1[ind])
            yy1 = np.maximum(y1[i], y1[ind])
            xx2 = np.minimum(x2[i], x2[ind])
            yy2 = np.minimum(y2[i], y2[ind])

            w = np.maximum(0., xx2 - xx1)
            h = np.maximum(0., yy2 - yy1)

            wh = w * h
            o = wh / (area[i] + area[ind] - wh)

            ind = ind[np.nonzero(o <= overlap)[0]]

        return dets[pick, :], scores[pick]

    def preprocess(self, cv2_img):
        input_blob = self.net.blobs[self.net.inputs[0]]
        t1=time.time(); img = cv2.resize(cv2_img, (input_blob.width, input_blob.height)); t2=time.time(); self.t_resize+=t2-t1
        if self.need_equalize:
            t1=time.time(); img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); t2=time.time(); self.t_cvtColor+=t2-t1
            t1=time.time(); img = cv2.equalizeHist(img); t2=time.time(); self.t_equalizeHist+=t2-t1
            t1=time.time(); img = np.tile(img, (3, 1, 1)).transpose(0, 2, 1); t2=time.time(); self.t_transpose+=t2-t1
        else:
            img = img.transpose(2, 1, 0)
        img = img.astype(np.float32, copy=False)
        img -= self.mean_caffe_img  # always ok, because if you don't want to subtract, it's 0.
        return img


    def gen_windows(self, cv2_img, window_sizes=WINDOW_SIZES, stride=STRIDE):
        windows = []  # each element looks like: h_start, w_start, h, w
        H, W, C = cv2_img.shape
        for size in window_sizes:
            for h_start in xrange(0, H-size+1, stride):
                for w_start in xrange(0, W-size+1, stride):
                    windows.append((h_start, w_start, size, size))
        return windows