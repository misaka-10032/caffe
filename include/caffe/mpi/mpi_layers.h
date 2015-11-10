/* @name mpi_layers
 * @brief Define basic layers that supports parallelism using mpi.
 *
 * @author misaka-10032 (longic@andrew.cmu.edu)
 * @bug Only support CPU
 */

#ifndef CAFFE_MPI_LAYERS_H
#define CAFFE_MPI_LAYERS_H

#include "caffe/layer.hpp"
#include "caffe/mpi/scheduler.h"

namespace caffe {
  /* Base class */
  template<typename Dtype>
  class MpiLayer : public Layer<Dtype> {
  public:
    explicit MpiLayer(const LayerParameter& param)
        : Layer<Dtype>(param) {
      Scheduler<Dtype> *scheduler = Scheduler<Dtype>::Get();
      parallelism_ = scheduler->getParallelism();
      rank_ = scheduler->getRank();
    }

  protected:
    int parallelism_;
    int rank_;
  };

  /* MpiFcLayer */
  template<typename Dtype>
  class MpiFcLayer : public MpiLayer<Dtype> {
  public:
    explicit MpiFcLayer(const LayerParameter &param)
        : MpiLayer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top);

  protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
//    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
//                             const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down,
                              const vector<Blob<Dtype>*>& bottom);
//    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
//                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  private:
    int M_;
    int K_;
    int N_;
    bool bias_term_;
    Blob<Dtype> bias_multiplier_;
  };

  /* MpiDistrLayer */
  template<typename Dtype>
  class MpiDistrLayer : public MpiLayer<Dtype> {
  public:
    MpiDistrLayer(const LayerParameter &param)
        : MpiLayer<Dtype>(param) { }

    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top);

    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
//    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
//    const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down,
                              const vector<Blob<Dtype>*>& bottom);
//    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
//    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  };

  /* MpiSyncLayer */
  template<typename Dtype>
  class MpiSyncLayer : public MpiLayer<Dtype> {
  public:
    MpiSyncLayer(const LayerParameter &param)
        : MpiLayer<Dtype>(param) { }

    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top);

    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
//    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
//    const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down,
                              const vector<Blob<Dtype>*>& bottom);
//    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
//    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  };

}
#endif //CAFFE_MPI_LAYERS_H
