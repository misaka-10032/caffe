#!/bin/bash
./build/tools/compute_image_mean -backend=lmdb \
	examples/xiyuan/db_train examples/xiyuan/mean.binaryproto
