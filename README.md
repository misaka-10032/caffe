# Model parallelism with Caffe

It's just a class project. We played around with it and did some experiments.

### What's up

* Forked from [Caffe](https://github.com/BVLC/caffe)
* [10701](http://www.cs.cmu.edu/~epxing/Class/10701-15F/) class project
* Implemented model-parallel version of inner product layer using MPI

### Patched classes
* `Scheduler`: schedules setups, forwards and backwards.
* `Operator`: wrapper for layers; used to operate on both original layers and mpi layers.
* `MpiOperable`: marked on mpi layers in order to be operable by `Operator`.


### Bugs

* Cannot save snapshots/models
* Cannot load snapshots/models into distributed environment
* Removed shared weight feature