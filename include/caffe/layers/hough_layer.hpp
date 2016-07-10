#ifndef CAFFE_HOUGH_LAYER_HPP_
#define CAFFE_HOUGH_LAYER_HPP_

#include <vector>
#include <cmath>
#include <boost/thread.hpp>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

/**
 * @brief Sparse transform matrix for Hough transform
 */
template <typename Dtype>
class HoughBasis {
public:
  HoughBasis(int H, int W, int THETA=-1, int RHO=-1) :
      H_(H), W_(W) {
    theta_min_ = InferThetaMin();
    theta_max_ = InferThetaMax();
    rho_min_ = InferRhoMin(H, W);
    rho_max_ = InferRhoMax(H, W);

    THETA = THETA > 0 ? THETA : (theta_max_-theta_min_);
    RHO = RHO > 0 ? RHO : (rho_max_-rho_min_);
    this->THETA_ = THETA;
    this->RHO_ = RHO;

    theta_step_ = Dtype(theta_max_-theta_min_) / THETA;
    rho_step_ = Dtype(rho_max_-rho_min_) / RHO;

    csr_val_.reset(new SyncedMemory(sizeof(Dtype) * H*W*THETA));
    csr_ci_.reset(new SyncedMemory(sizeof(int) * H*W*THETA));
    csr_ro_.reset(new SyncedMemory(sizeof(int) * (1+H*W)));

    csc_val_.reset(new SyncedMemory(sizeof(Dtype) * H*W*THETA));
    csc_ri_.reset(new SyncedMemory(sizeof(int) * H*W*THETA));
    csc_co_.reset(new SyncedMemory(sizeof(int) * (1+THETA*RHO)));

    if (Caffe::mode() == Caffe::CPU)
      Init_cpu();
    else
      Init_gpu();
  }

  void Init_gpu();

  void Init_cpu() {
    const Dtype pi = std::acos(-1);
    SyncedMemory theta_(sizeof(Dtype) * THETA_);
    for (int theta_i = 0; theta_i < THETA_; theta_i++) {
      Dtype theta = theta_min_ + theta_i * theta_step_;
      ((Dtype*) theta_.mutable_cpu_data())[theta_i] = theta * pi / 180;
    }
    SyncedMemory sin_(sizeof(Dtype) * THETA_);
    SyncedMemory cos_(sizeof(Dtype) * THETA_);
    caffe_sincos(THETA_, (const Dtype*) theta_.cpu_data(),
                 (Dtype*) sin_.mutable_cpu_data(),
                 (Dtype*) cos_.mutable_cpu_data());

    // TODO: parallel for
    for (int idx = 0; idx < H_*W_*THETA_; idx++) {
      const int hw = idx / THETA_;
      const int theta_i = idx % THETA_;
      const int h = hw / W_;
      const int w = hw % W_;
      const int ro = hw * THETA_;

      Dtype rho = h * ((Dtype*) sin_.cpu_data())[theta_i] +
                  w * ((Dtype*) cos_.cpu_data())[theta_i];
      int rho_i = int( (rho-rho_min_)/rho_step_ );
      int ci = theta_i * RHO_ + rho_i;  // col idx
      csr_val_mutable_cpu_data()[ro+theta_i] = Dtype(1);
      csr_ci_mutable_cpu_data()[ro+theta_i] = ci;

      if (theta_i == 0) {
        csr_ro_mutable_cpu_data()[hw] = ro;
        if (idx == H_*W_*THETA_-1) {
          csr_ro_mutable_cpu_data()[hw+1] = ro + THETA_;
        }
      }
    }

    caffe_csr2csc(H_*W_, THETA_*RHO_, H_*W_*THETA_,
                  csr_val_cpu_data(), csr_ro_cpu_data(), csr_ci_cpu_data(),
                  csc_val_mutable_cpu_data(), csc_ri_mutable_cpu_data(),
                  csc_co_mutable_cpu_data());
}

  inline int H() { return H_; }
  inline int W() { return W_; }
  inline int RHO() { return RHO_; }
  inline int THETA() { return THETA_; }
  inline int theta_min() { return theta_min_; }
  inline int theta_max() { return theta_max_; }
  inline int rho_min() { return rho_min_; }
  inline int rho_max() { return rho_max_; }
  inline int nnz() { return H_*W_*THETA_; }

  inline const Dtype* csr_val_cpu_data() { return (const Dtype*) csr_val_->cpu_data(); }
  inline const int* csr_ro_cpu_data() { return (const int*) csr_ro_->cpu_data(); }
  inline const int* csr_ci_cpu_data() { return (const int*) csr_ci_->cpu_data(); }
  inline const Dtype* csr_val_gpu_data() { return (const Dtype*) csr_val_->gpu_data(); }
  inline const int* csr_ro_gpu_data() { return (const int*) csr_ro_->gpu_data(); }
  inline const int* csr_ci_gpu_data() { return (const int*) csr_ci_->gpu_data(); }

  inline const Dtype* csc_val_cpu_data() { return (const Dtype*) csc_val_->cpu_data(); }
  inline const int* csc_ri_cpu_data() { return (const int*) csc_ri_->cpu_data(); }
  inline const int* csc_co_cpu_data() { return (const int*) csc_co_->cpu_data(); }
  inline const Dtype* csc_val_gpu_data() { return (const Dtype*) csc_val_->gpu_data(); }
  inline const int* csc_ri_gpu_data() { return (const int*) csc_ri_->gpu_data(); }
  inline const int* csc_co_gpu_data() { return (const int*) csc_co_->gpu_data(); }

  // inclusive min of theta
  static inline int InferThetaMin() { return -90; }
  // exclusive max of theta
  static inline int InferThetaMax() { return 90; }
  // inclusive min of rho
  static inline int InferRhoMin(int H, int W) {
    return int(std::floor(-std::sqrt(H*H+W*W)));
  }
  // exclusive max of rho
  static int InferRhoMax(int H, int W) {
    return int(std::ceil(std::sqrt(H*H+W*W)));
  }
  // infer shape of basis
  static vector<int> InferShape(int H, int W) {
    vector<int> shape(4);
    shape[0] = H;
    shape[1] = W;
    shape[2] = InferThetaMax() - InferThetaMin();
    shape[3] = InferRhoMax(H, W) - InferRhoMin(H, W);
    return shape;
  }

protected:
  inline Dtype* csr_val_mutable_cpu_data() { return (Dtype*) csr_val_->mutable_cpu_data(); }
  inline int* csr_ro_mutable_cpu_data() { return (int*) csr_ro_->mutable_cpu_data(); }
  inline int* csr_ci_mutable_cpu_data() { return (int*) csr_ci_->mutable_cpu_data(); }
  inline Dtype* csr_val_mutable_gpu_data() { return (Dtype*) csr_val_->mutable_gpu_data(); }
  inline int* csr_ro_mutable_gpu_data() { return (int*) csr_ro_->mutable_gpu_data(); }
  inline int* csr_ci_mutable_gpu_data() { return (int*) csr_ci_->mutable_gpu_data(); }

  inline Dtype* csc_val_mutable_cpu_data() { return (Dtype*) csc_val_->mutable_cpu_data(); }
  inline int* csc_ri_mutable_cpu_data() { return (int*) csc_ri_->mutable_cpu_data(); }
  inline int* csc_co_mutable_cpu_data() { return (int*) csc_co_->mutable_cpu_data(); }
  inline Dtype* csc_val_mutable_gpu_data() { return (Dtype*) csc_val_->mutable_gpu_data(); }
  inline int* csc_ri_mutable_gpu_data() { return (int*) csc_ri_->mutable_gpu_data(); }
  inline int* csc_co_mutable_gpu_data() { return (int*) csc_co_->mutable_gpu_data(); }

private:
  int H_;            // range of height in spatial domain
  int W_;            // rnage of width in spatial domain
  int RHO_;          // rnage of rho in Hough domain
  int THETA_;        // range of theta in Hough domain
  int theta_min_;    // min of theta
  int theta_max_;    // max of theta
  int theta_step_;   // step of theta
  int rho_min_;      // min of rho
  int rho_max_;      // max of rho
  int rho_step_;     // step of rho

  shared_ptr<SyncedMemory> csr_val_;   // csr values, array of Dtype
  shared_ptr<SyncedMemory> csr_ro_;    // csr row offsets, array of int
  shared_ptr<SyncedMemory> csr_ci_;    // csr col indices, array of int

  shared_ptr<SyncedMemory> csc_val_;   // csc values, array of Dtype
  shared_ptr<SyncedMemory> csc_ri_;    // csc row indices, equiv to ci of csr.T
  shared_ptr<SyncedMemory> csc_co_;    // csc col offsets, equiv to ro of csr.T
};

/**
 * @brief Global factory for HoughBasis. As basis takes space, there's
 *        only one allocated each kind of shapes.
 */
template <typename Dtype>
class HoughBasisFactory {
public:
  static shared_ptr<HoughBasis<Dtype> > Get(vector<int> shape) {
    boost::mutex::scoped_lock lock(mutex_);
    shape = HoughBasis<Dtype>::InferShape(shape[0], shape[1]);
    typename map<vector<int>, shared_ptr<HoughBasis<Dtype> > >::iterator
        it = factory_.find(shape);
    if (it != factory_.end()) {
      LOG(INFO) << "Found HoughBasis with shape {" << shape[0] << ", "
                << shape[1] << ", " << shape[2] << ", " << shape[3] << "}";
      return it->second;
    } else {
      LOG(INFO) << "Didn't find HoughBasis with shape {" << shape[0] << ", "
                << shape[1] << ", " << shape[2] << ", " << shape[3] << "}";
      shared_ptr<HoughBasis<Dtype> > ptr(new HoughBasis<Dtype>(
            shape[0], shape[1], shape[2], shape[3]));
      LOG(INFO) << "Finished building HoughBasis {" << shape[0] << ", "
                << shape[1] << ", " << shape[2] << ", " << shape[3] << "}";
      factory_[shape] = ptr;
      return ptr;
    }
  }

private:
  static map<vector<int>, shared_ptr<HoughBasis<Dtype> > > factory_;
  static boost::mutex mutex_;
};


/**
 * @brief Hough transform layer.
 *        It takes only one bottom and outputs one top.
 *        It only takes batch_size as 1.
 */
template <typename Dtype>
class HoughLayer : public Layer<Dtype> {
public:
  explicit HoughLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Hough"; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);

  shared_ptr<HoughBasis<Dtype> > hb_ptr_;
  int H_, W_, THETA_, RHO_;
};

}  // namespace caffe

#endif  // CAFFE_HOUGH_LAYER_HPP_
