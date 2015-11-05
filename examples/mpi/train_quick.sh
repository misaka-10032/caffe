#!/usr/bin/env sh

TOOLS=./build/tools
EXAMPLE=examples/mpi

$TOOLS/caffe train \
  --solver=$EXAMPLE/cifar10_quick_solver.prototxt
