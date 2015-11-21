/* @name split_blob
 * @brief Test split blob
 *
 * @author misaka-10032 (longqic@andrew.cmu.edu)
 */

#include <vector>
#include <glog/logging.h>
#include <iostream>
#include <boost/smart_ptr/shared_ptr.hpp>
#include "caffe/caffe.hpp"

using std::cout;
using std::endl;
using caffe::Blob;
using caffe::Caffe;
using caffe::vector;
using caffe::shared_ptr;

int main(int argc, char** argv) {
  vector<int> shape;
  shape.resize(2);
  shape[0] = 2;
  shape[1] = 10;

  Blob<float>* blob = new Blob<float>(shape);
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 10; j++) {
      int idx = i * 10 + j;
      *(blob->mutable_cpu_diff() + idx) = (float) idx;
      cout << *(blob->cpu_diff() + idx) << " ";
    }
    cout << endl;
  }
  cout << endl << endl;

  vector<shared_ptr<Blob<float> > > blobs;
  Blob<float>::Split1(blob, 3, blobs);

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
      int idx = i * 3 + j;
      cout << *(blobs[0]->cpu_diff() + idx) << " ";
    }
    cout << endl;
  }
  cout << endl;

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
      int idx = i * 3 + j;
      cout << *(blobs[1]->cpu_diff() + idx) << " ";
    }
    cout << endl;
  }
  cout << endl;

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 4; j++) {
      int idx = i * 4 + j;
      cout << *(blobs[2]->cpu_diff() + idx) << " ";
    }
    cout << endl;
  }

  return 0;
}
