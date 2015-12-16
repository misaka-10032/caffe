#include <vector>
#include <glog/logging.h>
#include <iostream>
#include <boost/smart_ptr/shared_ptr.hpp>
#include "caffe/caffe.hpp"

using std::cout;
using std::endl;
using caffe::Blob;
using caffe::BlobProto;
using caffe::Caffe;
using caffe::vector;
using caffe::shared_ptr;

int main(int argc, char **argv) {
  vector<int> shape;
  int pieces = 3;
  shape.resize(2);
  shape[0] = 2;
  shape[1] = 3;

  vector<int> lastShape;
  lastShape.resize(2);
  lastShape[0] = 2;
  lastShape[1] = 4;

  vector<shared_ptr<Blob<float> > > blobs(pieces);
  blobs[0].reset(new Blob<float>(shape));
  blobs[1].reset(new Blob<float>(shape));
  blobs[2].reset(new Blob<float>(lastShape));

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
      int idx = i * 3 + j;
      *(blobs[0]->mutable_cpu_data() + idx) = (float) (idx);
      cout << *(blobs[0]->cpu_data() + idx) << " ";
    }
    cout << endl;
  }
  cout << endl;

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
      int idx = i * 3 + j;
      *(blobs[1]->mutable_cpu_data() + idx) = (float) (idx);
      cout << *(blobs[1]->cpu_data() + idx) << " ";
    }
    cout << endl;
  }
  cout << endl;

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 4; j++) {
      int idx = i * 4 + j;
      *(blobs[2]->mutable_cpu_data() + idx) = (float) (idx);
      cout << *(blobs[2]->cpu_data() + idx) << " ";
    }
    cout << endl;
  }
  cout << endl;

  cout << endl;

  vector<int> shapeMerge(2);
  shapeMerge[0] = 2;
  shapeMerge[1] = 10;

  shared_ptr<Blob<float> > blob(new Blob<float>(shapeMerge));
  Blob<float>::Merge1(blobs, blob.get());

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 10; j++) {
      int idx = i * 10 + j;
      cout << *(blob->cpu_data() + idx) << " ";
    }
    cout << endl;
  }

  return 0;
}