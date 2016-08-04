#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"

#define protected public
#include "caffe/layers/balanced_sigmoid_cross_entropy_loss_layer.hpp"
#include "caffe/layers/balanced_sigmoid_cross_entropy_loss_ref_layer.hpp"
#undef protected

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class BalancedSigmoidCrossEntropyLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  BalancedSigmoidCrossEntropyLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(1, 10, 5, 5)),
        blob_bottom_targets_(new Blob<Dtype>(1, 10, 5, 5)),
        blob_top_loss_(new Blob<Dtype>(1, 1, 1, 1)) {
    // Fill the data vector
    FillerParameter data_filler_param;
    data_filler_param.set_std(1);
    GaussianFiller<Dtype> data_filler(data_filler_param);
    data_filler.Fill(blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    // Fill the targets vector
    FillerParameter targets_filler_param;
    targets_filler_param.set_min(0);
    targets_filler_param.set_max(1);
    UniformFiller<Dtype> targets_filler(targets_filler_param);
    targets_filler.Fill(blob_bottom_targets_);
    for (int i = 0; i < blob_bottom_targets_->count(); i++) {
      blob_bottom_targets_->mutable_cpu_data()[i] =
          Dtype(blob_bottom_targets_->cpu_data()[i] > 0.5);
    }
    blob_bottom_vec_.push_back(blob_bottom_targets_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~BalancedSigmoidCrossEntropyLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_targets_;
    delete blob_top_loss_;
  }

  void TestForward() {
    LayerParameter layer_param;
    const Dtype kLossWeight = 3.7;
    layer_param.add_loss_weight(kLossWeight);
    FillerParameter data_filler_param;
    data_filler_param.set_std(1);
    GaussianFiller<Dtype> data_filler(data_filler_param);
    FillerParameter targets_filler_param;
    targets_filler_param.set_min(0.0);
    targets_filler_param.set_max(1.0);
    UniformFiller<Dtype> targets_filler(targets_filler_param);
    Dtype eps = 2e-2;

    BalancedSigmoidCrossEntropyLossLayer<Dtype> layer(layer_param);
    BalancedSigmoidCrossEntropyLossRefLayer<Dtype> layer_ref(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer_ref.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    for (int i = 0; i < 100; ++i) {
      // Fill the data vector
      data_filler.Fill(this->blob_bottom_data_);
      // Fill the targets vector
      targets_filler.Fill(this->blob_bottom_targets_);
      for (int k = 0; k < this->blob_bottom_targets_->count(); k++) {
        this->blob_bottom_targets_->mutable_cpu_data()[k] =
            Dtype(this->blob_bottom_targets_->cpu_data()[k] > 0.5);
      }

      //Caffe::set_mode(Caffe::CPU);  // will mess up syncedmem, don't know why
      Dtype reference_loss = layer_ref.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
      this->blob_top_vec_[0]->mutable_cpu_data()[0] = -1;
      //Caffe::set_mode(Caffe::GPU);
      Dtype layer_loss = layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

      EXPECT_NEAR(reference_loss, layer_loss, eps) << "debug: trial #" << i;
    }
  }

  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_targets_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(BalancedSigmoidCrossEntropyLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(BalancedSigmoidCrossEntropyLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  const Dtype kLossWeight = 3.7;
  layer_param.add_loss_weight(kLossWeight);
  BalancedSigmoidCrossEntropyLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}


}  // namespace caffe
