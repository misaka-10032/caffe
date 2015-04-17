import os
import shutil
import tarfile
import math
import cv2
import caffe

import functools

tmp_dir = os.path.join('/tmp', __name__)

'''
Convention: inner representation uses (h, w);
    user interfaces represents as (w, h).
    Take that in func signature design and order them as (h, w), because that's inner call.
'''


def need_tmp_dir(tmp_dir):
    def decorator(func):
        @functools.wraps(func)
        def f(*args, **kwargs):
            _create_tmp_dir(tmp_dir)
            func(*args, **kwargs)
            _remove_tmp_dir(tmp_dir)
        return f
    return decorator


def _create_tmp_dir(tmp_dir):
    try:
        os.makedirs(tmp_dir)
    except OSError:
        if not os.path.isdir(tmp_dir):
            raise


def _remove_tmp_dir(tmp_dir):
    shutil.rmtree(tmp_dir)


@need_tmp_dir(tmp_dir)
def display_imgs_in_tar(tar_file, size=5):
    import matplotlib.pyplot as plt
    plt.axis('off')
    with tarfile.open(tar_file) as tar:
        for y in xrange(size):
            for x in xrange(size):
                # get next image from tar
                while True:
                    ti = tar.next()
                    if ti.name.endswith('.jpg'):
                        break
                tar.extract(ti, tmp_dir)
                img = cv2.imread(os.path.join(tmp_dir, ti.name))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                idx = y * size + x
                plt.subplot(size, size, idx)
                plt.imshow(img)
    plt.show()


def cv2_img_to_datum(cv2_img, label):
    img = cv2_img.transpose(2, 0, 1)  # h, w, c -> c, h, w
    img = img[(2, 1, 0), :, :]  # b, g, r -> r, g, b
    return caffe.io.array_to_datum(img, label)


def dist(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def oversample_img(img, box, angles, H, W):
    '''
    :param img:
    :param box: (y, x, h, w)
    :param angles: list of angles
    :param H: uni-height of the cropped image
    :param W: uni-width of the cropped image
    :return: list of rotated & flipped & cropped (to square) & resized (to uni-size) images
    '''
    y, x, h, w = box
    if x < 0 or y < 0 or\
            x+w > img.shape[1] or\
            y+h > img.shape[0]:
        img = cv2.copyMakeBorder(img,
                                 -min(0, y),
                                 max(0, y+h-img.shape[0]),
                                 -min(0, x),
                                 max(0, x+w-img.shape[1]),
                                 cv2.BORDER_CONSTANT)
        x = max(0, x)
        y = max(0, y)
    box = (y, x, h, w)

    if h * w <= 0:
        return []

    imgs = []
    for angle in angles:
        r = cv2.getRotationMatrix2D((x+w/2, y+h/2), angle, 1.0)
        img_rot = cv2.warpAffine(img, r, (img.shape[1], img.shape[0]))
        target = img_rot[y:y+h, x:x+w]
        try:
            imgs.extend(crop_resize_image(target, H, W))
            imgs.extend(crop_resize_image(cv2.flip(target, 1), H, W))
        except:
            print '**oversample_img'
            print box
            print img.shape
            print img_rot.shape
            print target.shape
            raise
    return imgs


def crop_resize_image(img, H, W):
    h_whole, w_whole = img.shape[:2]
    if abs(w_whole - h_whole) < 3:
        return [cv2.resize(img, (W, H))]

    imgs = []
    if w_whole > h_whole:
        h_start, h, w = 0, h_whole, h_whole
        # left
        w_start = 0

        try:
            imgs.append(cv2.resize(
                img[h_start:h_start+h, w_start:w_start+w],
                (W, H)))
        except:
            print '***crop_resize_image'
            print img.shape
            print h_start, w_start, h, w
            raise

        # middle
        w_start = (w_whole - w) / 2
        imgs.append(cv2.resize(
            img[h_start:h_start+h, w_start:w_start+w],
            (W, H)))
        # right
        w_start = w_whole - w
        imgs.append(cv2.resize(
            img[h_start:h_start+h, w_start:w_start+w],
            (W, H)))
        # whole
        imgs.append(cv2.resize(img, (W, H)))
    else:
        w_start, h, w = 0, w_whole, w_whole
        # top
        h_start = 0
        imgs.append(cv2.resize(
            img[h_start:h_start+h, w_start:w_start+w],
            (W, H)))
        # middle
        h_start = (h_whole - h) / 2
        imgs.append(cv2.resize(
            img[h_start:h_start+h, w_start:w_start+w],
            (W, H)))
        # bottom
        h_start = h_whole - h
        imgs.append(cv2.resize(
            img[h_start:h_start+h, w_start:w_start+w],
            (W, H)))
        # whole
        imgs.append(cv2.resize(img, (W, H)))

    return imgs


def put_imgs_to_tar(imgs, tar, name_prefix, TMP_DIR):
    for img, idx in zip(imgs, xrange(len(imgs))):
        tmp_file = os.path.join(TMP_DIR, 'tmp.jpg')
        out_name = '%s_%s.jpg' % (name_prefix, idx)
        cv2.imwrite(tmp_file, img)
        tar.add(tmp_file, out_name)
        os.remove(tmp_file)