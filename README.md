# Model parallelism with Caffe

It's just a class project. __Please don't expect too much of it.__ We played around with it and did some experiments.

### What's up

* Forked from [Caffe](https://github.com/BVLC/caffe)
* [10701](http://www.cs.cmu.edu/~epxing/Class/10701-15F/) class project
* Implemented model-parallel version of inner product layer using [boost-mpi](http://www.boost.org/doc/libs/1_59_0/doc/html/mpi/tutorial.html)
* Extensible for other parallelism schemes.
* [This report](report.pdf) briefly introduces what we have done. Again, please don't expect too much of it.

### Patched classes
* `Scheduler`: schedules setups, forwards and backwards.
* `Operator`: wrapper for layers; used to operate on both original layers and mpi layers.
* `MpiOperable`: marked on mpi layers in order to be operable by `Operator`.


### Bugs

* Cannot save snapshots/models
* Cannot load snapshots/models into distributed environment
* Removed shared weight feature
* `mpirun -np 5 ...` renders NaN.
