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
using caffe::BlobProto;
using caffe::Caffe;
using caffe::vector;
using caffe::shared_ptr;
using caffe::string;

int main(int argc, char** argv) {
  vector<int> shape;
  shape.resize(2);
  shape[0] = 4;
  shape[1] = 10;

  Blob<float>* blob = new Blob<float>(shape);
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 10; j++) {
      int idx = i * 10 + j;
      *(blob->mutable_cpu_data() + idx) = (float) -idx * 0.001;
      cout << *(blob->cpu_data() + idx) << " ";
//      *(blob->mutable_cpu_diff() + idx) = (float) idx;
//      cout << *(blob->cpu_diff() + idx) << " ";
    }
    cout << endl;
  }
  cout << endl << endl;

  vector<shared_ptr<Blob<float> > > blobs_remote;
  Blob<float>::Split1(blob, 3, blobs_remote);

  vector<shared_ptr<Blob<float> > > blobs(3);
  for (int i = 0; i < 3; i++) {
    string blob_str;
    shared_ptr<BlobProto> blobRemoteProto(new BlobProto());
    blobs_remote[i]->ToProto(blobRemoteProto.get(), true);
    blob_str = blobRemoteProto->SerializeAsString();

    shared_ptr<BlobProto> blobProto(new BlobProto());
    blobProto->ParseFromString(blob_str);
    blobs[i].reset(new Blob<float>());
    blobs[i]->FromProto(*blobProto, true);
  }

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 3; j++) {
      int idx = i * 3 + j;
      cout << *(blobs[0]->cpu_data() + idx) << " ";
//      cout << *(blobs[0]->cpu_diff() + idx) << " ";
    }
    cout << endl;
  }
  cout << endl;

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 3; j++) {
      int idx = i * 3 + j;
      cout << *(blobs[1]->cpu_data() + idx) << " ";
//      cout << *(blobs[1]->cpu_diff() + idx) << " ";
    }
    cout << endl;
  }
  cout << endl;

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      int idx = i * 4 + j;
      cout << *(blobs[2]->cpu_data() + idx) << " ";
//      cout << *(blobs[2]->cpu_diff() + idx) << " ";
    }
    cout << endl;
  }

  return 0;
}
