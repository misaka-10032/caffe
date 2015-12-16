#!/bin/bash

mpirun -np 6 -wdir $CAFFE_ROOT build-release/tools/caffe train --solver=examples/mpi/mnist/lenet_solver.prototxt
