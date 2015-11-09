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
    int pieces = 2;
    shape.resize(2);
    shape[0] = 2;
    shape[1] = 5;

    vector<shared_ptr<Blob<float> > > blobs =
            *(new vector<shared_ptr<Blob<float> > >(pieces));
    for (int i = 0; i < pieces; i++)
    {
            blobs[i].reset(new Blob<float>(shape));
    }
    for (int k = 0; k < pieces; k++ )
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 5; j++) {
            int idx = i * 5 + j;
            *(blobs[k]->mutable_cpu_data() + idx) = (float) (k * 2 * 5 + idx);
            cout << *(blobs[k]->cpu_data() + idx) << " ";
        }
        cout << endl;
    }
    cout << endl << endl;

    Blob<float>& blob = Blob<float>::Merge1(blobs);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 10; j++) {
            int idx = i * 10 + j;
            cout << *(blob.cpu_data() + idx) << " ";
        }
        cout << endl;
    }

    return 0;
}