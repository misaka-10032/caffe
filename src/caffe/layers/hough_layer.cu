#include <vector>
#include "caffe/common.hpp"
#include "caffe/blob.hpp"
#include "caffe/layers/hough_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void InitHoughBasis(const int H_, const int W_,
                               const int THETA_, const int RHO_,
                               const Dtype* sin_, const Dtype* cos_,
                               const int rho_min_, const Dtype rho_step_,
                               Dtype* val_, int* ro_, int* ci_) {
  CUDA_KERNEL_LOOP(idx, H_*W_*THETA_) {
    const int hw = idx / THETA_;
    const int theta_i = idx % THETA_;
    const int h = hw / W_;
    const int w = hw % W_;
    const int ro = hw * THETA_;

    Dtype rho = h*sin_[theta_i] + w*cos_[theta_i];
    int rho_i = int( (rho-rho_min_)/rho_step_ );
    int ci = theta_i * RHO_ + rho_i;  // col idx
    val_[ro+theta_i] = Dtype(1);
    ci_[ro+theta_i] = ci;

    if (theta_i == 0) {
      ro_[hw] = ro;
      if (idx == H_*W_*THETA_-1) {
        ro_[hw+1] = ro + THETA_;
      }
    }
  }
}

template <typename Dtype>
void HoughBasis<Dtype>::Init_gpu() {
  const Dtype pi = std::acos(-1);
  SyncedMemory theta_(sizeof(Dtype) * THETA_);
  for (int theta_i = 0; theta_i < THETA_; theta_i++) {
    Dtype theta = theta_min_ + theta_i * theta_step_;
    ((Dtype*) theta_.mutable_cpu_data())[theta_i] = theta * pi / 180;
  }
  SyncedMemory sin_(sizeof(Dtype) * THETA_);
  SyncedMemory cos_(sizeof(Dtype) * THETA_);
  caffe_gpu_sincos(THETA_, (const Dtype*) theta_.gpu_data(),
                   (Dtype*) sin_.mutable_gpu_data(),
                   (Dtype*) cos_.mutable_gpu_data());
  InitHoughBasis<<<CAFFE_GET_BLOCKS(H_*W_*THETA_), CAFFE_CUDA_NUM_THREADS>>>(
      H_, W_, THETA_, RHO_, (const Dtype*) sin_.gpu_data(), (const Dtype*) cos_.gpu_data(),
      rho_min_, rho_step_, csr_val_mutable_gpu_data(), csr_ro_mutable_gpu_data(),
      csr_ci_mutable_gpu_data());
  CUDA_POST_KERNEL_CHECK;
  caffe_gpu_csr2csc(H_*W_, THETA_*RHO_, H_*W_*THETA_,
                    csr_val_gpu_data(), csr_ro_gpu_data(), csr_ci_gpu_data(),
                    csc_val_mutable_gpu_data(), csc_ri_mutable_gpu_data(),
                    csc_co_mutable_gpu_data());
}

template void HoughBasis<float>::Init_gpu();
template void HoughBasis<double>::Init_gpu();

/**
 * HoughLayer
 */

template <typename Dtype>
void HoughLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top) {
  caffe_gpu_csrmv(CblasNoTrans, hb_ptr_->RHO()*hb_ptr_->THETA(),
                  hb_ptr_->H()*hb_ptr_->W(), hb_ptr_->nnz(),
                  Dtype(1), hb_ptr_->csc_val_gpu_data(),
                  hb_ptr_->csc_co_gpu_data(), hb_ptr_->csc_ri_gpu_data(),
                  bottom[0]->gpu_data(), Dtype(0), top[0]->mutable_gpu_data());
}

template <typename Dtype>
void HoughLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                     const vector<bool>& propagate_down,
                                     const vector<Blob<Dtype>*>& bottom) {
  caffe_gpu_csrmv(CblasNoTrans, hb_ptr_->H()*hb_ptr_->W(),
                  hb_ptr_->RHO()*hb_ptr_->THETA(), hb_ptr_->nnz(),
                  Dtype(1), hb_ptr_->csr_val_gpu_data(),
                  hb_ptr_->csr_ro_gpu_data(), hb_ptr_->csr_ci_gpu_data(),
                  top[0]->gpu_diff(), Dtype(0), bottom[0]->mutable_gpu_diff());
}

INSTANTIATE_LAYER_GPU_FUNCS(HoughLayer);

/**
 * HoughTransposeLayer
 */

template <typename Dtype>
void HoughTransposeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
  caffe_gpu_csrmv(CblasNoTrans, hb_ptr_->H()*hb_ptr_->W(),
                  hb_ptr_->RHO()*hb_ptr_->THETA(), hb_ptr_->nnz(),
                  Dtype(1), hb_ptr_->csr_val_gpu_data(),
                  hb_ptr_->csr_ro_gpu_data(), hb_ptr_->csr_ci_gpu_data(),
                  bottom[0]->gpu_data(), Dtype(0), top[0]->mutable_gpu_data());
}

template <typename Dtype>
void HoughTransposeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                              const vector<bool>& propagate_down,
                                              const vector<Blob<Dtype>*>& bottom) {
  caffe_gpu_csrmv(CblasNoTrans, hb_ptr_->RHO()*hb_ptr_->THETA(),
                  hb_ptr_->H()*hb_ptr_->W(), hb_ptr_->nnz(),
                  Dtype(1), hb_ptr_->csc_val_gpu_data(),
                  hb_ptr_->csc_co_gpu_data(), hb_ptr_->csc_ri_gpu_data(),
                  top[0]->gpu_diff(), Dtype(0), bottom[0]->mutable_gpu_diff());
}

INSTANTIATE_LAYER_GPU_FUNCS(HoughTransposeLayer);

}  // namespace caffe
