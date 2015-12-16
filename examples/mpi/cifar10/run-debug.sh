#!/bin/bash

mpirun -np 3 -wdir $CAFFE_ROOT build/tools/caffe-d train --solver=examples/mpi/cifar10_quick_solver.prototxt
