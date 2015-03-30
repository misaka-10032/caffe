import caffe
import cv2
import numpy as np

# WINDOW_SIZES = [30, 40, 50, 60, 70, 80]
WINDOW_SIZES = [22, 24, 28, 32, 36, 40]
STRIDE = 4
BATCH_SIZE = 100

SCALES = [.6, .8, 1., 1.2, 1.6, 2.]

class Detector:
    def __init__(self, model_def_file, model_file,
                 mean_file=None, need_equalize=False, swap_channel=(2, 1, 0)):  # b, g, r -> r, g, b
        self.net = caffe.Net(model_def_file, model_file, caffe.TEST)
        self.mean_caffe_img = np.load(mean_file).astype(np.float32, copy=False) if mean_file else 0
        self.need_equalize = need_equalize
        self.swap_channel = swap_channel

    def detect_whole(self, cv2_img, labels_of_interest=[0]):
        #h_origin, w_origin = cv2_img.shape[:2]  # original
        #cv2_imgs = [cv2.resize(cv2_img, (int(w_origin*scale), int(h_origin*scale))) for scale in SCALES]

        input_blob = self.net.blobs[self.net.inputs[0]]
        w_win, h_win = input_blob.width, input_blob.height

        # now each value is list, after running nms, it would be np.array
        rects = {label: [] for label in labels_of_interest}
        scores = {label: [] for label in labels_of_interest}

        img = cv2_img
        h, w = img.shape[:2]  # rescaled frame size
        caffe_img = self.preprocess_whole(img)
        input = caffe_img.reshape((1, ) + caffe_img.shape)
        # probs = self.net.forward_all(**{self.net.inputs[0]: input})['prob'].squeeze(0)
        probs = self.net.forward_all(data=input)['prob'][0]
        predicts = np.argmax(probs.reshape(probs.shape[0]), axis=1).reshape(probs.shape[:-2])
        ws, hs = probs.shape[:-2]  # how may steps in each axis
        stride_w = int((w - w_win) / (ws - 1))
        stride_h = int((h - h_win) / (hs - 1))
        rects_scaled = {label: [] for label in labels_of_interest}
        scores_scaled = {label: [] for label in labels_of_interest}
        for w_start, w_idx in zip(xrange(0, w-w_win, stride_w), xrange(0, ws)):
            for h_start, h_idx in zip(xrange(0, h-h_win, stride_h), xrange(0, hs)):
                pred = predicts[w_idx, h_idx]  # by caffe it's w by h
                if (pred in labels_of_interest):
                    rects_scaled[pred].append((h_start, w_start, h, w))  # by cv2 it's h by w
                    scores_scaled[pred].append(probs[pred][w_idx][h_idx])  # by caffe

        for label in labels_of_interest:
            rects[label].extend(rects_scaled[label])
            scores[label].extend(scores_scaled[label])

        for label in labels_of_interest:
            rects[label], scores[label] = self.nms_detections(rects[label], scores[label])

        return rects, scores

    def detect_windows(self, cv2_img, labels_of_interest=None, thresh=.9, windows=None, batch_size=BATCH_SIZE):
        '''
        :param cv2_img: shape as (H, W, C)
        :param windows: [(h_start, w_start, h, w), ...]
        :return: {[(), (), ...], ...}, where r[i] is the detection for label i, r[i][j] is the jth window
        '''
        self.detections, self.scores = {}, {}
        labels_of_interest = range(self.net.blobs[self.net.outputs[0]].count)\
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
            self.detections[label_pred], self.scores[label_pred] = \
                self.nms_detections(self.detections[label_pred], self.scores[label_pred])

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
        probs_batch = self.net.forward_all(**{self.net.inputs[0]: patches})['prob'].squeeze((2, 3))
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
            return np.array([]), np.array([])

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

    def preprocess_whole(self, cv2_img):
        img = cv2_img.transpose(2, 1, 0)
        img = img.astype(np.float32, copy=False)
        img -= self.mean_caffe_img.mean()  # TODO: can I do in this way??
        if self.swap_channel:
            img = img[self.swap_channel, :, :]
        return img

    def preprocess(self, cv2_img):
        input_blob = self.net.blobs[self.net.inputs[0]]
        img = cv2.resize(cv2_img, (input_blob.width, input_blob.height))
        if self.need_equalize:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.equalizeHist(img)
            img = np.tile(img, (3, 1, 1)).transpose(1, 2, 0)  # c, h, w -> h, w, c

        # img = img.transpose(2, 0, 1)  # h, w, c -> c, h, w
        img = img.transpose(2, 1, 0)  # h, w, c -> c, h, w
        img = img.astype(np.float32, copy=False)

        if self.swap_channel:
            img = img[self.swap_channel, :, :]

        img -= self.mean_caffe_img  # is zero if not specified

        return img

    def gen_windows(self, cv2_img, window_sizes=WINDOW_SIZES, stride=STRIDE):
        windows = []  # each element looks like: h_start, w_start, h, w
        H, W, C = cv2_img.shape
        for size in window_sizes:
            for h_start in xrange(0, H-size+1, stride):
                for w_start in xrange(0, W-size+1, stride):
                    windows.append((h_start, w_start, size, size))
        return windows