#!/usr/bin/python
import os
import caffe
from google.protobuf import text_format

from gen_mean import gen_mean
from gen_mean import MEAN_NPY as MEAN_NPY_ORIGINAL

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

SCALE_MIN = 3./10.
SCALE_MAX = 3./4.
NUM_SCALES = 10
order = (SCALE_MAX / SCALE_MIN) ** (1. / (NUM_SCALES-1))
SCALES = [SCALE_MIN * order**i for i in xrange(NUM_SCALES)]
W_BASE, H_BASE = 1024, 576
BASE_SHAPE = (H_BASE, W_BASE)
SHAPES = [(int(H_BASE*s), int(W_BASE*s)) for s in SCALES]

_MODEL_FILE = 'xiyuan_quick_iter_120000.caffemodel'
_DEPLOY_FILE = 'xiyuan_quick_deploy.proto'
DEPLOY = os.path.join(CURRENT_DIR, _DEPLOY_FILE)
MODEL = os.path.join(CURRENT_DIR, _MODEL_FILE)
params = ['ip1', 'ip2', 'ip3']

# TUNE THE BASE PROTO BY HAND
DEPLOY_PREFIX = 'xiyuan_full_conv'
BASE_DEPLOY = os.path.join(CURRENT_DIR, '%s-%dx%d.proto' % (DEPLOY_PREFIX, W_BASE, H_BASE))
DEPLOYS_SG = [os.path.join(CURRENT_DIR, '%s-%dx%d.proto' % (DEPLOY_PREFIX, w, h)) for h, w in SHAPES]

MODEL_PREFIX = DEPLOY_PREFIX
MODELS_SG = [os.path.join(CURRENT_DIR, '%s-%dx%d.caffemodel' % (MODEL_PREFIX, w, h)) for h, w in SHAPES]
params_full_conv = ['ip1-conv', 'ip2-conv', 'ip3-conv']

MEAN_PREFIX = DEPLOY_PREFIX
MEANS_NPY = [os.path.join(CURRENT_DIR, '%s-%dx%d.mean.npy' % (MEAN_PREFIX, w, h)) for h, w in SHAPES]
MEANS_BIN = [os.path.join(CURRENT_DIR, '%s-%dx%d.mean.binaryproto' % (MEAN_PREFIX, w, h)) for h, w in SHAPES]


def main():
    # original net
    net = caffe.Net(DEPLOY, MODEL, caffe.TEST)
    fc_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params}

    for fc in params:
        print '{} weights are {} dimensional and biases are {} dimensional'.format(fc, fc_params[fc][0].shape, fc_params[fc][1].shape)
    print

    for MODEL_OUT, DEPLOY_OUT, MEAN_NPY, MEAN_BIN, SHAPE in \
            zip(MODELS_SG, DEPLOYS_SG, MEANS_NPY, MEANS_BIN, SHAPES):
        print '*' * 50
        print MODEL_OUT
        print '*' * 50

        h, w = SHAPE
        # generate mean.npy & mean.binaryproto
        gen_mean(h, w, MEAN_NPY_ORIGINAL, MEAN_NPY, MEAN_BIN)

        deploy = caffe.proto.caffe_pb2.NetParameter()
        # generate deploy.proto
        with open(str(BASE_DEPLOY)) as f:
            text_format.Merge(f.read(), deploy)
            deploy.input_dim[2] = h  # H
            deploy.input_dim[3] = w  # W
        with open(str(DEPLOY_OUT), 'w') as f:
            str_out = text_format.MessageToString(deploy)
            f.write(str_out)

        # new net
        net_full_conv = caffe.Net(DEPLOY_OUT, MODEL, caffe.TEST)
        conv_params = {pr: (net_full_conv.params[pr][0].data, net_full_conv.params[pr][1].data)
                       for pr in params_full_conv}

        for conv in params_full_conv:
            print '{} weights are {} dimensional and biases are {} dimensional'\
                .format(conv, conv_params[conv][0].shape, conv_params[conv][1].shape)
        print

        # surgery on bias
        for pr, pr_conv in zip(params, params_full_conv):
            conv_params[pr_conv][1][...] = fc_params[pr][1]

        # surgery on weights
        for pr, pr_conv in zip(params, params_full_conv):
            out, in_, h, w = conv_params[pr_conv][0].shape
            W = fc_params[pr][0].reshape((out, in_, h, w))
            conv_params[pr_conv][0][...] = W

        # save net
        net_full_conv.save(MODEL_OUT)
        print


if __name__ == '__main__':
    main()