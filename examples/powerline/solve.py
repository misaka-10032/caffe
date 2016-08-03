from __future__ import division
import numpy as np
import sys
import caffe
from skimage.measure import block_reduce
from surgery import transplant, interp

# a fully convolutional VGG16 net.
# http://nbviewer.ipython.org/github/BVLC/caffe/blob/master/examples/net_surgery.ipynb
base_def = 'models/hed.deploy'
base_weights = 'models/5stage-vgg.caffemodel'
#base_weights = 'models/hed_pretrained_bsds.caffemodel'
#base_weights = 'models/hed+fp.caffemodel'
#base_def = 'models/hough_v2.deploy'
#base_weights = 'models/hough_v2.caffemodel'
base_net = caffe.Net(base_def, base_weights, caffe.TRAIN)

test_lst = 'data/test/test_pair.lst'
h_coarse = w_coarse = 20  # evaluate in coarse granularity

# init
caffe.set_mode_gpu()
caffe.set_device(0)

solver = caffe.SGDSolver('solver.prototxt')

# surgery
transplant(solver.net, base_net)
# upsample layers
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
interp(solver.net, interp_layers)

# tricky transplant conv5_x into conv6_x
#p_bases = ['score-dsn5']
#p_news = ['score-dsn6']
#for k, p_base in enumerate(p_bases):
#    p_new = p_news[k]
#    for i in xrange(len(base_net.params[p_base])):
#        solver.net.params[p_new][i].data.flat = base_net.params[p_base][i].data.flat

# release
base_net = None

# restore
#solver.restore('dsn-full-res-3-scales_iter_29000.solverstate')


def seg_tests(solver):
    print 'testing...'
    net = solver.test_nets[0]
    net.share_with(solver.net)
    with open(test_lst) as f:
        cnt = len(f.readlines())
    tp = fp = tn = fn = 0
    for _ in xrange(cnt):
        net.forward()
        pred = np.squeeze(net.blobs['sigmoid_fuse'].data[0]) > 0.5
        pred = block_reduce(pred, (h_coarse, w_coarse), np.max)
        not_pred = np.logical_not(pred)
        gt = np.squeeze(net.blobs['label'].data[0]) > 0.5
        gt = block_reduce(gt, (h_coarse, w_coarse), np.max)
        not_gt = np.logical_not(gt)
        tp += np.sum(np.logical_and(pred, gt))
        fp += np.sum(np.logical_and(pred, not_gt))
        tn += np.sum(np.logical_and(not_pred, not_gt))
        fn += np.sum(np.logical_and(not_pred, gt))

    if tp+fp > 0:
        precision = float(tp) / (tp+fp)
    else:
        precision = -1
    if tp+fn > 0:
        recall = float(tp) / (tp+fn)
    else:
        recall = -1
    accuracy = float(tp+tn) / (tp+fp+tn+fn)
    print 'tp, fp, tn, fn = {}, {}, {}, {}'.format(tp, fp, tn, fn)
    print 'precision:', precision
    print 'recall:', recall
    print 'accuracy:', accuracy


# solve straight through -- a better approach is to define a solving loop to
# 1. take SGD steps
# 2. score the model by the test net `solver.test_nets[0]`
# 3. repeat until satisfied
for _ in xrange(3):
    solver.step(2000)
    seg_tests(solver)
solver.step(100)

