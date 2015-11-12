/* @name mpi_sync_layer
 * @brief TODO
 *
 * @author misaka-10032 (longqic@andrew.cmu.edu)
 */

#include "caffe/mpi/mpi_layers.h"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"

namespace caffe {

  template <typename Dtype>
  void MpiSyncLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
    // do nothing
  }

  template <typename Dtype>
  void MpiSyncLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top) {
    // do nothing
  }

  template <typename Dtype>
  void MpiSyncLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
    // do nothing
  }

  template <typename Dtype>
  void MpiSyncLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                         const vector<bool>& propagate_down,
                                         const vector<Blob<Dtype>*>& bottom) {
    // do nothing
  }


//#ifdef CPU_ONLY
//  STUB_GPU(MpiSyncLayer);
//#endif

  INSTANTIATE_CLASS(MpiSyncLayer);
  REGISTER_LAYER_CLASS(MpiSync);
}