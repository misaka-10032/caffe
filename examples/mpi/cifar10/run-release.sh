#!/bin/bash

mpirun -np 3 -wdir $CAFFE_ROOT build-release/tools/caffe train --solver=examples/mpi/cifar10_quick_solver.prototxt
