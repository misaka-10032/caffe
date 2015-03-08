#!/usr/bin/env sh
set -e

TOOLS=./build/tools

$TOOLS/caffe train \
  --solver=examples/xiyuan/cifar10_quick_solver.prototxt
