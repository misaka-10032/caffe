#!/bin/bash

mpirun -np 5 -wdir $CAFFE_ROOT build-release/tools/caffe train --solver=examples/mpi/cifar10/cifar10_quick_solver.prototxt
