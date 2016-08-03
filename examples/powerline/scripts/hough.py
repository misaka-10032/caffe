#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse.linalg as spla

sin = lambda x: np.sin(x*np.pi/180)
cos = lambda x: np.cos(x*np.pi/180)
sigmoid = lambda x: 1./(1+np.exp(-x))


class Hough:
    def __init__(self, H, W, THETA=180, RHO=240):
        self.H = H
        self.W = W
        self.THETA = THETA
        self.RHO = RHO

        theta_min = -90.
        theta_max = 90.
        theta_step = float(theta_max-theta_min) / THETA
        rho_min = np.floor(-np.sqrt(H*H+W*W))
        rho_max = np.ceil(np.sqrt(H*H+W*W))
        rho_step = float(rho_max-rho_min) / RHO

        sin_ = [sin(x) for x in np.arange(theta_min, theta_max, theta_step)]
        cos_ = [cos(x) for x in np.arange(theta_min, theta_max, theta_step)]
        val_ = np.zeros(H*W*THETA, dtype=float)
        ci_ = np.zeros(H*W*THETA, dtype=int)
        ro_ = np.zeros(H*W+1, dtype=int)
        for idx in xrange(H*W*THETA):
            hw = idx / THETA
            theta_i = idx % THETA
            h, w = hw/W, hw%W
            ro = hw * THETA

            rho = h*sin_[theta_i] + w*cos_[theta_i]
            rho_i = int( (rho-rho_min)/rho_step )
            ci = theta_i * RHO + rho_i
            val_[ro+theta_i] = 1.
            ci_[ro+theta_i] = ci

            if theta_i == 0:
                ro_[hw] = ro
                if idx == H*W*THETA-1:
                    ro[hw+1] = ro + THETA

        self.hb = csr_matrix((val_, ci_, ro_), shape=(H*W, THETA*RHO), dtype=float)

    def forward(self, bottom, normalize=True):
        h, w = bottom.shape
        hft = self.hb.transpose().dot(bottom.reshape(-1)).reshape((self.THETA, self.RHO))
        if normalize and hft.max() > 0:
            hft /= self.H * self.W
        return hft

    def backward(self, top_diff, normalize=True):
        h, w = top_diff.shape
        pre = self.hb.dot(top_diff.reshape(-1)).reshape((self.H, self.W))
        if normalize and pre.max() > 0:
            pre /= pre.max()
        return pre

