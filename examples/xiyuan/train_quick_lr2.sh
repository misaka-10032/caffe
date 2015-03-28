#!/usr/bin/env sh
set -e

TOOLS=./build/tools

$TOOLS/caffe train \
    --solver=examples/xiyuan/xiyuan_quick_solver_lr2.proto \
    --snapshot=examples/xiyuan/xiyuan_quick_lr1_iter_10000.solverstate
