#ifndef CAFFE_IMAGE_LABELMAP_DATA_LAYER_HPP_
#define CAFFE_IMAGE_LABELMAP_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class LabelmapBatch {
 public:
  Blob<Dtype> data_, labelmap_;
};


template <typename Dtype>
class BasePrefetchingLabelmapDataLayer :
    public BaseDataLayer<Dtype>, public InternalThread {
 public:
  explicit BasePrefetchingLabelmapDataLayer(const LayerParameter& param);
  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden.
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  // Prefetches batches (asynchronously if to GPU memory)
  static const int PREFETCH_COUNT = 3;

 protected:
  virtual void InternalThreadEntry();
  virtual void load_batch(LabelmapBatch<Dtype>* labelmapbatch) = 0;

  LabelmapBatch<Dtype> prefetch_[PREFETCH_COUNT];
  BlockingQueue<LabelmapBatch<Dtype>*> prefetch_free_;
  BlockingQueue<LabelmapBatch<Dtype>*> prefetch_full_;

  Blob<Dtype> transformed_data_;
  Blob<Dtype> transformed_labelmap_;
};


/**
 * @brief Provides data to the Net from image groundtruth pairs.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class ImageLabelmapDataLayer : public BasePrefetchingLabelmapDataLayer<Dtype> {
 public:
  explicit ImageLabelmapDataLayer(const LayerParameter& param)
      : BasePrefetchingLabelmapDataLayer<Dtype>(param) {}
  virtual ~ImageLabelmapDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ImageLabelmapData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; } //could be three if considering label

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleImages();
  virtual void load_batch(LabelmapBatch<Dtype>* batch);

  vector<std::pair<std::string, std::string> > lines_;
  int lines_id_;
};

}  // namespace caffe

#endif  // CAFFE_IMAGE_LABELMAP_DATA_LAYER_HPP_
