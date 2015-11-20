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
  shape[1] = 5;

  Blob<float>* blob = new Blob<float>(shape);
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 5; j++) {
      int idx = i * 5 + j;
      *(blob->mutable_cpu_diff() + idx) = (float) idx;
      cout << *(blob->cpu_diff() + idx) << " ";
    }
    cout << endl;
  }
  cout << endl << endl;

  Blob<float>* blob_acc = new Blob<float>(shape);
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 5; j++) {
      int idx = i * 5 + j;
      *(blob_acc->mutable_cpu_diff() + idx) = (float) 0;
    }
  }
  for (int i = 0; i < 5; i++) {
    Blob<float>::AccumulateDiff(blob_acc, blob);
  }

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 5; j++) {
      int idx = i * 5 + j;
      cout << *(blob_acc->cpu_diff() + idx) << " ";
    }
    cout << endl;
  }

  return 0;
}
