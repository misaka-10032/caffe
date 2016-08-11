#!/usr/bin/env python
# encoding: utf-8

import os
import cv2
import numpy as np
from subprocess import call
from scipy.misc import imread, imsave
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from skimage.measure import block_reduce

# make sure it has bg/ ln/ post/ bg-pool ln-pool/ post-pool/ diff-pool rank/
root = '../eval'
cnt = 64
fmt_bg = fmt_ln = 'final_{}.png'
fmt_hmap = 'hmap_{}.png'
fmt_lmap = 'lmap_{}.npz'
fmt_dmap = 'dmap_{}.npz'
fmt_diff = 'diff_{}.png'
fmt_ln_over = 'ln+bg_{}.png'
fmt_post_over = 'hmap+bg_{}.png'
fmt_diff_over = 'diff+bg_{}.png'
f_rank = 'rankings.txt'

alpha_over = .75
pool_sz = 10

cdict = {'blue': ((0, -1, 1), (.48, 0, 0), (1, 0, -1)),
         'green': ((0, -1, 0), (1, 0, -1)),
         'red': ((0, -1, 0), (.52, 0, 0), (1, 1, -1))}
cmap = LinearSegmentedColormap('kayvon', cdict)

def main():
    # dir check
    assert os.path.isdir(os.path.join(root, 'bg'))
    assert os.path.isdir(os.path.join(root, 'ln'))
    #assert os.path.isdir(os.path.join(root, 'pred'))
    assert os.path.isdir(os.path.join(root, 'post'))
    assert os.path.isdir(os.path.join(root, 'bg-pool'))
    assert os.path.isdir(os.path.join(root, 'ln-pool'))
    #assert os.path.isdir(os.path.join(root, 'pred-pool'))
    assert os.path.isdir(os.path.join(root, 'post-pool'))
    assert os.path.isdir(os.path.join(root, 'diff-pool'))
    assert os.path.isdir(os.path.join(root, 'rank'))

    """ Maps idx to square score """
    score = {}

    """ Pool """
    for i in xrange(cnt):
        print 'Processing', i, '...'
        f_bg = fmt_bg.format(i)
        bg = imread(os.path.join(root, 'bg', f_bg), mode='RGB')
        f_ln = fmt_ln.format(i)
        ln = (imread(os.path.join(root, 'ln', f_ln), mode='RGBA')[:, :, 3])
        ln = (ln.astype(float) / 255.) > .5
        f_hmap = fmt_hmap.format(i)
        f_lmap = fmt_lmap.format(i)
        f_dmap = fmt_dmap.format(i)
        f_diff = fmt_diff.format(i)
        post = np.load(os.path.join(root, 'post', f_lmap))['labels']

        bg_pool = block_reduce(bg, (pool_sz, pool_sz, 1), np.average)
        ln_pool = block_reduce(ln, (pool_sz, pool_sz), np.max)
        post_pool = block_reduce(post, (pool_sz, pool_sz), np.max)
        diff_pool = post_pool.astype(float) - ln_pool.astype(float)

        # trick to make bg green
        if diff_pool.min() == 0:
            diff_pool[0, 0] = -1
        if diff_pool.max() == 0:
            diff_pool[0, 1] = 1

        pred, gt = post_pool, ln_pool
        pred_not = np.logical_not(pred)
        gt_not = np.logical_not(gt)
        tp = np.sum(np.logical_and(pred, gt))
        fp = np.sum(np.logical_and(pred, gt_not))
        tn = np.sum(np.logical_and(pred_not, gt_not))
        fn = np.sum(np.logical_and(pred_not, gt))
        if tp+fp != 0:
            precision = float(tp) / (tp+fp)
        else:
            precision = -1
        if tp+fn != 0:
            recall = float(tp) / (tp+fn)
        else:
            recall = -1
        if precision == -1 or recall == -1:
            f1 = -1
        else:
            f1 = 2 * precision * recall / (precision + recall)
        score[i] = f1

        imsave(os.path.join(root, 'bg-pool', f_bg), bg_pool)

        plt.imsave(os.path.join(root, 'ln-pool', f_ln), ln_pool, cmap=cmap)
        np.savez_compressed(os.path.join(root, 'ln-pool', f_lmap), labels=ln_pool)

        plt.imsave(os.path.join(root, 'post-pool', f_hmap), post_pool, cmap=cmap)
        np.savez_compressed(os.path.join(root, 'post-pool', f_lmap), labels=post_pool)

        plt.imsave(os.path.join(root, 'diff-pool', f_diff), diff_pool, cmap=cmap)
        np.savez_compressed(os.path.join(root, 'diff-pool', f_dmap), diff=diff_pool)

        ln_pool = imread(os.path.join(root, 'ln-pool', f_ln), mode='RGB')
        post_pool = imread(os.path.join(root, 'post-pool', f_hmap), mode='RGB')
        diff_pool = imread(os.path.join(root, 'diff-pool', f_diff), mode='RGB')
        f_ln_over = fmt_ln_over.format(i)
        f_post_over = fmt_post_over.format(i)
        f_diff_over = fmt_diff_over.format(i)
        ln_over = cv2.addWeighted(bg_pool, 1., ln_pool, alpha_over, 0., dtype=cv2.CV_32F)
        post_over = cv2.addWeighted(bg_pool, 1., post_pool, alpha_over, 0., dtype=cv2.CV_32F)
        diff_over = cv2.addWeighted(bg_pool, 1., diff_pool, alpha_over, 0., dtype=cv2.CV_32F)
        imsave(os.path.join(root, 'ln-pool', f_ln_over), ln_over)
        imsave(os.path.join(root, 'post-pool', f_post_over), post_over)
        imsave(os.path.join(root, 'diff-pool', f_diff_over), diff_over)

    """ Rank """
    ranked = sorted(score.keys(), key=lambda i: score[i], reverse=True)
    with open(os.path.join(root, 'rank', f_rank), 'w') as f:
        for i in xrange(cnt):
            f.write('{} {} {}\n'.format(i, ranked[i], score[ranked[i]]))


if __name__ == '__main__':
    main()

