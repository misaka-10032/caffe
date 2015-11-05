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
  void MpiSplitLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
    // TODO
  }

  template <typename Dtype>
  void MpiSplitLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                     const vector<Blob<Dtype>*>& top) {
    // TODO
  }

  template <typename Dtype>
  void MpiSplitLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
    // TODO
  }

  template <typename Dtype>
  void MpiSplitLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                          const vector<bool>& propagate_down,
                                          const vector<Blob<Dtype>*>& bottom) {
    // TODO
  }


//#ifdef CPU_ONLY
//  STUB_GPU(MpiSyncLayer);
//#endif

  INSTANTIATE_CLASS(MpiSplitLayer);
  REGISTER_LAYER_CLASS(MpiSplit);
}