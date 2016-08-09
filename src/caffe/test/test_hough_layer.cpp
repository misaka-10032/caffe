#include "gtest/gtest.h"
#include "caffe/common.hpp"
#include "caffe/blob.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/hough_layer.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

/**
 * HoughLayer
 */

template <typename TypeParam>
class HoughLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
protected:
  HoughLayerTest() :
      blob_bottom_(new Blob<Dtype>(1, 1, 5, 5)),
      blob_top_(new Blob<Dtype>(1, 1, 5, 5)) {
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~HoughLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(HoughLayerTest, TestDtypesAndDevices);

TYPED_TEST(HoughLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  HoughParameter hough_param = layer_param.hough_param();
  hough_param.set_theta_range(5);
  hough_param.set_rho_range(5);
  HoughLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer,
                                  this->blob_bottom_vec_,
                                  this->blob_top_vec_);
}

/**
 * HoughTransposeLayer
 */

template <typename TypeParam>
class HoughTransposeLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
protected:
  HoughTransposeLayerTest() :
      blob_bottom_(new Blob<Dtype>(1, 1, 5, 5)),
      blob_top_(new Blob<Dtype>(1, 1, 5, 5)) {
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~HoughTransposeLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(HoughTransposeLayerTest, TestDtypesAndDevices);

TYPED_TEST(HoughTransposeLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  HoughParameter hough_param = layer_param.hough_param();
  hough_param.set_h_range(5);
  hough_param.set_w_range(5);
  HoughTransposeLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer,
                                  this->blob_bottom_vec_,
                                  this->blob_top_vec_);
}

}  // namespace caffe
