/* @name forwarder
 * @brief Impl for normal forwarder
 *
 * @author misaka-10032 (longqic@andrew.cmu.edu)
 */

#include "caffe/mpi/operators.h"

namespace caffe {
  template<typename Dtype>
  Operator<Dtype>::Operator(Net<Dtype>* net) : net_(net) {
      scheduler_ = Scheduler<Dtype>::Get();
      slave_cnt_ = scheduler_->getSlaveCnt();
      rank_ = scheduler_->getRank();
  }

  template<typename Dtype>
  Dtype Operator<Dtype>::Forward(int layer_id) {
    vector<Blob<Dtype>*>& top_vecs = this->net_->top_vecs_[layer_id];
    vector<Blob<Dtype>*>& bottom_vecs = this->net_->bottom_vecs_[layer_id];
    shared_ptr<Layer<Dtype> >& layer = this->net_->layers_[layer_id];

    Dtype loss = static_cast<Dtype>(0);

    if (this->rank_ == 0) {        /* is master */
      loss = layer->Forward(bottom_vecs, top_vecs);
    }

    return loss;
  }

  template<typename Dtype>
  void Operator<Dtype>::Backward(int layer_id) {
    shared_ptr<Layer<Dtype> >& layer = this->net_->layers_[layer_id];
    vector<Blob<Dtype>*>& top_vecs = net_->top_vecs_[layer_id];
    vector<Blob<Dtype>*>& bottom_vecs = net_->bottom_vecs_[layer_id];
    vector<bool>& bottom_need_backward = net_->bottom_need_backward_[layer_id];

    if (this->rank_ == 0) {        /* is master */
      layer->Backward(top_vecs, bottom_need_backward, bottom_vecs);
    }
  }

  INSTANTIATE_CLASS(Operator);
}