#!/bin/bash
# run under build

mpirun -np 3 gdb tools/caffe-d train --solver=../examples/mpi/cifar10_quick_solver.prototxt
