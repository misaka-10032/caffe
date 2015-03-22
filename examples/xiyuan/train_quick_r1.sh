#!/usr/bin/env sh
set -e

TOOLS=./build/tools

$TOOLS/caffe train \
    --solver=examples/xiyuan/xiyuan_quick_solver_lr1.prototxt \
    --snapshot=examples/xiyuan/xiyuan_quick_iter_2000.solverstate