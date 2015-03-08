#!/usr/bin/env sh
set -e

TOOLS=./build/tools

$TOOLS/caffe train \
    --solver=examples/xiyuan/cifar10_quick_solver_lr1.prototxt \
    --snapshot=examples/xiyuan/cifar10_quick_iter_2000.solverstate