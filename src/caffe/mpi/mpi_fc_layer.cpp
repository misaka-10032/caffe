/* @name mpi_fc_layer
 * @brief Impl of MpiFcLayer
 *
 * @author misaka-10032 (longqic@andrew.cmu.edu)
 */

#include "caffe/mpi/mpi_layers.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void MpiFcLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top) {
  CHECK_GE(this->slave_cnt_, 1)
    << "parallelism must >= 1";

  const int num_output = this->layer_param_.inner_product_param().num_output();
  this->bias_term_ = this->layer_param_.inner_product_param().bias_term();
  this->N_ = num_output;

  if (this->rank_ != 0) {  /* Slave */
    if (this->rank_ == this->slave_cnt_) {
      /* Last worker */
      this->N_ = this->N_ / this->slave_cnt_ + this->N_ % this->slave_cnt_;
    } else {
      this->N_ /= this->slave_cnt_;
    }
  }

  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  this->K_ = bottom[0]->count(axis);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (this->bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Intialize the weight
    vector<int> weight_shape(2);
    weight_shape[0] = this->N_;
    weight_shape[1] = this->K_;
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.inner_product_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, intiialize and fill the bias term
    if (this->bias_term_) {
      vector<int> bias_shape(1, this->N_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.inner_product_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void MpiFcLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top) {
  InnerProductLayer<Dtype>::Reshape(bottom, top);
}

template <typename Dtype>
void MpiFcLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top) {
  InnerProductLayer<Dtype>::Forward_cpu(bottom, top);
}

template <typename Dtype>
void MpiFcLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                     const vector<bool>& propagate_down,
                                     const vector<Blob<Dtype>*>& bottom) {
  InnerProductLayer<Dtype>::Backward_cpu(top, propagate_down, bottom);
}

#ifdef CPU_ONLY
  STUB_GPU(MpiFcLayer);
#endif

  INSTANTIATE_CLASS(MpiFcLayer);
  REGISTER_LAYER_CLASS(MpiFc);
}