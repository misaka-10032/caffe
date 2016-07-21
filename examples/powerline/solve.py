from __future__ import division
import numpy as np
import sys
import caffe
from surgery import transplant, interp

# a fully convolutional VGG16 net.
# http://nbviewer.ipython.org/github/BVLC/caffe/blob/master/examples/net_surgery.ipynb
base_def = 'models/hed.deploy'
base_weights = 'models/5stage-vgg.caffemodel'
#base_weights = 'models/hed_pretrained_bsds.caffemodel'
#base_weights = 'models/hed+fp.caffemodel'
base_net = caffe.Net(base_def, base_weights, caffe.TRAIN)

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
p_bases = ['score-dsn5']
p_news = ['score-dsn6']
for k, p_base in enumerate(p_bases):
    p_new = p_news[k]
    for i in xrange(len(base_net.params[p_base])):
        solver.net.params[p_new][i].data.flat = base_net.params[p_base][i].data.flat

# release
base_net = None

# restore
#solver.restore('dsn-full-res-3-scales_iter_29000.solverstate')

# solve straight through -- a better approach is to define a solving loop to
# 1. take SGD steps
# 2. score the model by the test net `solver.test_nets[0]`
# 3. repeat until satisfied
solver.step(6100)

