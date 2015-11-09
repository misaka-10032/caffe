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

  Blob<float> blob(shape);
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 5; j++) {
      int idx = i * 5 + j;
      *(blob.mutable_cpu_data() + idx) = (float) idx;
      cout << *(blob.cpu_data() + idx) << " ";
    }
    cout << endl;
  }
  cout << endl << endl;
	
  vector<shared_ptr<Blob<float> > > blobs = Blob<float>::Split1(blob, 2);

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      int idx = i * 2 + j;
      cout << *(blobs[0]->cpu_data() + idx) << " ";
    }
    cout << endl;
  }
  cout << endl;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
      int idx = i * 3 + j;
      cout << *(blobs[1]->cpu_data() + idx) << " ";
    }
    cout << endl;
  }

  return 0;
}
