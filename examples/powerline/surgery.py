from __future__ import division
import caffe
import numpy as np


def transplant(net, base_net, suffix=''):
    """
    Transplant net with base_net
    """
    for p_base in base_net.params:
        p_new = p_base + suffix
        if p_new not in net.params:
            print 'dropping', p_base
            continue
        for i in range(len(base_net.params[p_base])):
            if i > (len(net.params[p_new]) - 1):
                print 'dropping', p_base, i
                break
            if base_net.params[p_base][i].data.shape != net.params[p_new][i].data.shape:
                if np.product(base_net.params[p_base][i].data.shape ==
                   np.product(net.params[p_new][i].data.shape)):
                    print 'coercing', p_base, i, \
                        'from', base_net.params[p_base][i].data.shape, \
                        'to', net.params[p_new][i].data.shape
                else:
                    print 'dropping', p_base, i
                    break
            else:
                print 'copying', p_base, ' -> ', p_new, i
            net.params[p_new][i].data.flat = base_net.params[p_base][i].data.flat


def expand_score(new_net, new_layer, net, layer):
    old_cl = net.params[layer][0].num
    new_net.params[new_layer][0].data[:old_cl][...] = net.params[layer][0].data
    new_net.params[new_layer][1].data[0,0,0,:old_cl][...] = net.params[layer][1].data


def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)


def interp(net, layers):
    for l in layers:
        m, k, h, w = net.params[l][0].data.shape
        if m != k and k != 1:
            print 'input + output channels need to be the same or |output| == 1'
            raise
        if h != w:
            print 'filters need to be square'
            raise
        filt = upsample_filt(h)
        net.params[l][0].data[range(m), range(k), :, :] = filt
