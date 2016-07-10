#include "caffe/layers/hough_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
map<vector<int>, shared_ptr<HoughBasis<Dtype> > >
HoughBasisFactory<Dtype>::factory_;

template <typename Dtype>
boost::mutex
HoughBasisFactory<Dtype>::mutex_;

template <typename Dtype>
void HoughLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), 1) << "Only 1 bottom is allowed.";
  CHECK_EQ(top.size(), 1) << "Only 1 top is allowed.";
  CHECK_EQ(bottom[0]->shape(0), 1) << "Bach size should be 1";
  CHECK_EQ(bottom[0]->shape(1), 1) << "Channel size should be 1";
  H_ = bottom[0]->shape(2);
  W_ = bottom[0]->shape(3);
  // -1 of THETA/RHO means to be inferred
  THETA_ = this->layer_param_.hough_param().theta_range();
  RHO_ = this->layer_param_.hough_param().rho_range();
  vector<int> hough_shape(4);
  hough_shape[0] = H_;
  hough_shape[1] = W_;
  hough_shape[2] = THETA_;
  hough_shape[3] = RHO_;
  hb_ptr_ = HoughBasisFactory<Dtype>::Get(hough_shape);
  // update with the inferred ranges
  THETA_ = hb_ptr_->THETA();
  RHO_ = hb_ptr_->RHO();
}

template <typename Dtype>
void HoughLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape(4);
  top_shape[0] = 1;
  top_shape[1] = 1;
  top_shape[2] = hb_ptr_->THETA();
  top_shape[3] = hb_ptr_->RHO();
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void HoughLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top) {
  caffe_csrmv(CblasNoTrans, hb_ptr_->RHO()*hb_ptr_->THETA(), hb_ptr_->H()*hb_ptr_->W(),
              Dtype(1), hb_ptr_->csc_val_cpu_data(), hb_ptr_->csc_co_cpu_data(),
              hb_ptr_->csc_ri_cpu_data(), bottom[0]->cpu_data(), Dtype(0),
              top[0]->mutable_cpu_data());
}

template <typename Dtype>
void HoughLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                     const vector<bool>& propagate_down,
                                     const vector<Blob<Dtype>*>& bottom) {
  caffe_csrmv(CblasNoTrans, hb_ptr_->H()*hb_ptr_->W(), hb_ptr_->RHO()*hb_ptr_->THETA(),
              Dtype(1), hb_ptr_->csr_val_cpu_data(), hb_ptr_->csr_ro_cpu_data(),
              hb_ptr_->csr_ci_cpu_data(), top[0]->cpu_diff(), Dtype(0),
              bottom[0]->mutable_cpu_diff());
}

INSTANTIATE_CLASS(HoughLayer);
REGISTER_LAYER_CLASS(Hough);

}  // namespace caffe
