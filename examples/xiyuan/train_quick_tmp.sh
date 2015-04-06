#!/usr/bin/env sh
set -e

TOOLS=./build/tools

$TOOLS/caffe train \
    --solver=examples/xiyuan/xiyuan_quick_solver.proto \
    --snapshot=examples/xiyuan/xiyuan_quick_iter_15000.solverstate
