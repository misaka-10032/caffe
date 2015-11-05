#!/bin/bash

mpirun -np 3 build/tools/caffe-d train --solver=examples/mpi/cifar10_quick_solver.prototxt
