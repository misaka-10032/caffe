/* @name mpi_layers
 * @brief Define basic layers that supports parallelism using mpi.
 *
 * @author misaka-10032 (longic@andrew.cmu.edu)
 * @bug Only support CPU
 */

#ifndef CAFFE_MPI_LAYERS_H
#define CAFFE_MPI_LAYERS_H

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/mpi/scheduler.h"

namespace caffe {
  /* Base class */
  template<typename Dtype>
  class MpiOperable {
  public:
    explicit MpiOperable() {
      Scheduler<Dtype> *scheduler = Scheduler<Dtype>::Get();
      slave_cnt_ = scheduler->getSlaveCnt();
      rank_ = scheduler->getRank();
    }

  protected:
    int slave_cnt_;
    int rank_;
  };

  /* MpiFcLayer */
  template<typename Dtype>
  class MpiFcLayer : public MpiOperable<Dtype>,
                     public InnerProductLayer<Dtype> {
  public:
    explicit MpiFcLayer(const LayerParameter &param) :
        MpiOperable<Dtype>(),
        InnerProductLayer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top);

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
  };

  /* MpiSyncLayer */
  template<typename Dtype>
  class MpiSyncLayer : public MpiOperable<Dtype>,
                       public Layer<Dtype> {
  public:
    MpiSyncLayer(const LayerParameter &param)
        : MpiOperable<Dtype>(), Layer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top);

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
  };

}
#endif //CAFFE_MPI_LAYERS_H
