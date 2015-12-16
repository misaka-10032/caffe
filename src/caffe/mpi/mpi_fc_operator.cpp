/* @name mpi_fc_forwarder
 * @brief worker for MpiFcLayer
 *
 * @author misaka-10032 (longqic@andrew.cmu.edu)
 */

#include "caffe/mpi/operators.h"

namespace caffe {

  template<typename Dtype>
  Dtype MpiFcOperator<Dtype>::Forward(int layer_id) {
    Dtype loss = static_cast<Dtype>(0);
    vector<Blob<Dtype>*>& top_vecs = this->net_->top_vecs_[layer_id];
    vector<Blob<Dtype>*>& sliced_top_vecs =
        this->net_->sliced_top_vecs_[layer_id];
    vector<Blob<Dtype>*>& bottom_vecs =
        this->net_->bottom_vecs_[layer_id];
    shared_ptr<Layer<Dtype> >& layer = this->net_->layers_[layer_id];

    if (this->rank_ == 0) {           /* master */
      /* 1. broadcast */
      this->scheduler_->BroadcastBlobs(bottom_vecs, 0);

      /* 2. receive & merge & store */
      int top_cnt = top_vecs.size();
      for (int top_id = 0; top_id < top_cnt; top_id++) {
        vector<shared_ptr<Blob<Dtype> > > blobs(this->slave_cnt_);
        for (int slave_id = 1; slave_id <= this->slave_cnt_; slave_id++) {
          blobs[slave_id - 1].reset(new Blob<Dtype>());
          this->scheduler_->RecvBlob(slave_id, top_id, *blobs[slave_id - 1]);
        }
        // TODO: transpose, append, transpose?
        Blob<Dtype>::Merge1(blobs, top_vecs[top_id]);
      }

      /* 3. receive & accumulate loss */
      // TODO: add back if necessary
//      for (int slave_id = 1; slave_id <= this->slave_cnt_; slave_id++) {
//        Dtype local_loss;
//        this->scheduler_->RecvLoss(slave_id, local_loss);
//        loss += local_loss;
//      }

    } else {                         /* slave */
      /* 1. broadcast recv bottoms from master */
      this->scheduler_->BroadcastBlobs(bottom_vecs, 0);

      /* 2. do local job */
      Dtype local_loss = layer->Forward(bottom_vecs, sliced_top_vecs);

      /* 3. send blob */
      this->scheduler_->SendBlobs(0, sliced_top_vecs);

      /* 4. send loss */
      // TODO: add back if necessary
//      this->scheduler_->SendLoss(0, local_loss);

    }

    return loss;
  }

  template<typename Dtype>
  void MpiFcOperator<Dtype>::Backward(int layer_id) {
    vector<Blob<Dtype>*>& top_vecs = this->net_->top_vecs_[layer_id];
    vector<Blob<Dtype>*>& sliced_top_vecs = this->net_->sliced_top_vecs_[layer_id];
    vector<Blob<Dtype>*>& bottom_vecs = this->net_->bottom_vecs_[layer_id];
    shared_ptr<Layer<Dtype> >& layer = this->net_->layers_[layer_id];
    vector<bool>& bottom_need_backward = this->net_->bottom_need_backward_[layer_id];

    if (this->rank_ == 0) {          /* master */
      /* 1. send sliced tops */
      int top_cnt = top_vecs.size();
      for (int top_id = 0; top_id < top_cnt; top_id++) {
        vector<shared_ptr<Blob<Dtype> > > blobs;
        Blob<Dtype>::Split1(top_vecs[top_id], this->slave_cnt_, blobs);
        for (int slave_id = 1; slave_id <= this->slave_cnt_; slave_id++) {
          this->scheduler_->SendBlob(slave_id, top_id, *blobs[slave_id - 1]);
        }
      }

      /* 2. recv and combine bottoms from slaves */
      int bottom_cnt = bottom_vecs.size();
      for (int bottom_id = 0; bottom_id < bottom_cnt; bottom_id++) {
        shared_ptr<Blob<Dtype> > blob(new Blob<Dtype>());
        Blob<Dtype>* bottom_vec = bottom_vecs[bottom_id];
        for (int slave_id = 1; slave_id <= this->slave_cnt_; slave_id++) {
          if (slave_id == 1) {      /* Need reset diff before accum */
            this->scheduler_->RecvBlob(slave_id, bottom_id, *bottom_vec);
          } else {
            this->scheduler_->RecvBlob(slave_id, bottom_id, *blob);
            Blob<Dtype>::AccumulateDiff(bottom_vec, blob.get());
          }
        }
      }
    } else {                          /* slave */
      /* 1. recv tops from master */
      this->scheduler_->RecvBlobs(0, sliced_top_vecs);

      /* 2. back propagate */
      layer->Backward(sliced_top_vecs, bottom_need_backward, bottom_vecs);

      /* 3. send bottoms */
      this->scheduler_->SendBlobs(0, bottom_vecs);
    }
  }

  INSTANTIATE_CLASS(MpiFcOperator);
}
