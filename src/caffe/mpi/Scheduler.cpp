/* @name Scheduler
 * @brief Impl of Scheduler.h
 *
 * @author misaka-10032 (longqic@andrew.cmu.edu)
 * @bug
 */

#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/mpi/scheduler.h"

namespace caffe {

  template<typename Dtype>
  Scheduler<Dtype> *Scheduler<Dtype>::instance_ = nullptr;


  template <typename Dtype>
  void Scheduler<Dtype>::InputDebugInfo(const int input_id) {
    vector<Blob<Dtype>*>& net_input_blobs_ = net_->net_input_blobs_;
    vector<string>& blob_names_ = net_->blob_names_;
    vector<int>& net_input_blob_indices_ = net_->net_input_blob_indices_;

    const Blob<Dtype>& blob = *net_input_blobs_[input_id];
    const string& blob_name = blob_names_[net_input_blob_indices_[input_id]];
    const Dtype data_abs_val_mean = blob.asum_data() / blob.count();
    LOG_IF(INFO, Caffe::root_solver())
    << "    [Forward] "
    << "Input " << blob_name << " data: " << data_abs_val_mean;
  }

  template <typename Dtype>
  void Scheduler<Dtype>::ForwardDebugInfo(const int layer_id) {
    vector<vector<Blob<Dtype>*> >& bottom_vecs_ = net_->bottom_vecs_;
    vector<vector<Blob<Dtype>*> >& top_vecs_ = net_->top_vecs_;
    vector<string>& blob_names_ = net_->blob_names_;
    vector<vector<int> >& top_id_vecs_ = net_->top_id_vecs_;
    vector<string>& layer_names_ = net_->layer_names_;
    vector<shared_ptr<Layer<Dtype> > >& layers_ = net_->layers_;
    vector<vector<int> >& param_id_vecs_ = net_->param_id_vecs_;
    vector<string>& param_display_names_ = net_->param_display_names_;

    for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) {
      const Blob<Dtype>& blob = *top_vecs_[layer_id][top_id];
      const string& blob_name = blob_names_[top_id_vecs_[layer_id][top_id]];
      const Dtype data_abs_val_mean = blob.asum_data() / blob.count();
      LOG_IF(INFO, Caffe::root_solver())
      << "    [Forward] "
      << "Layer " << layer_names_[layer_id]
      << ", top blob " << blob_name
      << " data: " << data_abs_val_mean;
    }
    for (int param_id = 0; param_id < layers_[layer_id]->blobs().size();
         ++param_id) {
      const Blob<Dtype>& blob = *layers_[layer_id]->blobs()[param_id];
      const int net_param_id = param_id_vecs_[layer_id][param_id];
      const string& blob_name = param_display_names_[net_param_id];
      const Dtype data_abs_val_mean = blob.asum_data() / blob.count();
      LOG_IF(INFO, Caffe::root_solver())
      << "    [Forward] "
      << "Layer " << layer_names_[layer_id]
      << ", param blob " << blob_name
      << " data: " << data_abs_val_mean;
    }
  }

  template <typename Dtype>
  void Scheduler<Dtype>::BackwardDebugInfo(const int layer_id) {
    vector<vector<Blob<Dtype>*> >& bottom_vecs_ = net_->bottom_vecs_;
    vector<string>& blob_names_ = net_->blob_names_;
    vector<string>& layer_names_ = net_->layer_names_;
    vector<shared_ptr<Layer<Dtype> > >& layers_ = net_->layers_;
    vector<vector<bool> >& bottom_need_backward_ = net_->bottom_need_backward_;
    vector<vector<int> >& bottom_id_vecs_ = net_->bottom_id_vecs_;

    const vector<Blob<Dtype>*>& bottom_vec = bottom_vecs_[layer_id];
    for (int bottom_id = 0; bottom_id < bottom_vec.size(); ++bottom_id) {
      if (!bottom_need_backward_[layer_id][bottom_id]) { continue; }
      const Blob<Dtype>& blob = *bottom_vec[bottom_id];
      const string& blob_name = blob_names_[bottom_id_vecs_[layer_id][bottom_id]];
      const Dtype diff_abs_val_mean = blob.asum_diff() / blob.count();
      LOG_IF(INFO, Caffe::root_solver())
      << "    [Backward] "
      << "Layer " << layer_names_[layer_id]
      << ", bottom blob " << blob_name
      << " diff: " << diff_abs_val_mean;
    }
    for (int param_id = 0; param_id < layers_[layer_id]->blobs().size();
         ++param_id) {
      if (!layers_[layer_id]->param_propagate_down(param_id)) { continue; }
      const Blob<Dtype>& blob = *layers_[layer_id]->blobs()[param_id];
      const Dtype diff_abs_val_mean = blob.asum_diff() / blob.count();
      LOG_IF(INFO, Caffe::root_solver())
      << "    [Backward] "
      << "Layer " << layer_names_[layer_id]
      << ", param blob " << param_id
      << " diff: " << diff_abs_val_mean;
    }
  }

  template <typename Dtype>
  Dtype Scheduler<Dtype>::ForwardFromTo(int start, int end) {
    bool debug_info_ = net_->debug_info_;
    vector<shared_ptr<Layer<Dtype> > >& layers_ = net_->layers_;
    vector<Blob<Dtype>*>& net_input_blobs_ = net_->net_input_blobs_;
    vector<vector<Blob<Dtype>*> > bottom_vecs_ = net_->bottom_vecs_;
    vector<vector<Blob<Dtype>*> > top_vecs_ = net_->top_vecs_;

    CHECK_GE(start, 0);
    CHECK_LT(end, layers_.size());
    Dtype loss = 0;
    if (debug_info_) {
      for (int i = 0; i < net_input_blobs_.size(); ++i) {
        InputDebugInfo(i);
      }
    }
    for (int i = start; i <= end; ++i) {
      // LOG(ERROR) << "Forwarding " << layer_names_[i];
      Dtype layer_loss = layers_[i]->Forward(bottom_vecs_[i], top_vecs_[i]);
      loss += layer_loss;
      if (debug_info_) { ForwardDebugInfo(i); }
    }
    return loss;
  }

  template <typename Dtype>
  void Scheduler<Dtype>::BackwardFromTo(int start, int end) {
    bool debug_info_ = net_->debug_info_;
    vector<shared_ptr<Layer<Dtype> > >& layers_ = net_->layers_;
    vector<vector<Blob<Dtype>*> > bottom_vecs_ = net_->bottom_vecs_;
    vector<vector<Blob<Dtype>*> > top_vecs_ = net_->top_vecs_;
    vector<bool>& layer_need_backward_ = net_->layer_need_backward_;
    vector<vector<bool> >& bottom_need_backward_ = net_->bottom_need_backward_;

    CHECK_GE(end, 0);
    CHECK_LT(start, layers_.size());
    for (int i = start; i >= end; --i) {
      if (layer_need_backward_[i]) {
        layers_[i]->Backward(
            top_vecs_[i], bottom_need_backward_[i], bottom_vecs_[i]);
        if (debug_info_) { BackwardDebugInfo(i); }
      }
    }
  }

  INSTANTIATE_CLASS(Scheduler);
}