/* @name mpi_split_layer
 * @brief TODO
 *
 * @author misaka-10032 (longqic@andrew.cmu.edu)
 */

#include "caffe/mpi/mpi_layers.h"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"

namespace caffe {

  template <typename Dtype>
  void MpiDistrLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
    // make sure it's in parallel
    CHECK_GE(this->parallelism_, 1);
    // bottom and top are set up previously
    // make sure bottom and top point to the same blob
    CHECK_GE(bottom[0], top[0]);
  }

  template <typename Dtype>
  void MpiDistrLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                     const vector<Blob<Dtype>*>& top) {
    // Already reshaped in previous layer
  }

  template <typename Dtype>
  void MpiDistrLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
    // TODO
  }

  template <typename Dtype>
  void MpiDistrLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                          const vector<bool>& propagate_down,
                                          const vector<Blob<Dtype>*>& bottom) {
    // TODO
  }


//#ifdef CPU_ONLY
//  STUB_GPU(MpiSyncLayer);
//#endif

  INSTANTIATE_CLASS(MpiDistrLayer);
  REGISTER_LAYER_CLASS(MpiDistr);
}