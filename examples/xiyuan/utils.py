import os
import shutil
import tarfile
import cv2
import caffe

import functools

tmp_dir = os.path.join('/tmp', __name__)


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