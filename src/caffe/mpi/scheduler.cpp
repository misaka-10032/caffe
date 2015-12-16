/* @name Scheduler
 * @brief Impl of Scheduler.h
 *
 * @author misaka-10032 (longqic@andrew.cmu.edu)
 * @bug
 */

#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/serialization/string.hpp>

#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/mpi/scheduler.h"
#include "caffe/mpi/operators.h"
#include "caffe/mpi/mpi_layers.h"

namespace mpi = boost::mpi;


namespace caffe {
  template<typename Dtype>
  Scheduler<Dtype>::Scheduler() { }

  template<typename Dtype>
  shared_ptr<Scheduler<Dtype> > Scheduler<Dtype>::instance_;

  template<typename Dtype>
  std::mutex Scheduler<Dtype>::mutex_;

  template<typename Dtype>
  Scheduler<Dtype>* Scheduler<Dtype>::Get() {
    if (!instance_) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!instance_) {
        instance_.reset(new Scheduler<Dtype>());
      }
    }
    return instance_.get();
  }

  template<typename Dtype>
  Scheduler<Dtype>* Scheduler<Dtype>::Init(mpi::environment* env,
                                           mpi::communicator* world) {
    Get();
    instance_->env.reset(env);
    instance_->world.reset(world);
    return instance_.get();
  }

  template <typename Dtype>
  void Scheduler<Dtype>::SetUpLayer(Net<Dtype>* net_, int layer_id) {
    shared_ptr<Layer<Dtype> >& layer = net_->layers_[layer_id];
    vector<Blob<Dtype>*>& bottom_vecs = net_->bottom_vecs_[layer_id];
    vector<Blob<Dtype>*>& top_vecs = net_->top_vecs_[layer_id];
    vector<Blob<Dtype>*>& sliced_top_vecs = net_->sliced_top_vecs_[layer_id];
    MpiOperable<Dtype>* __as_mpi__ =
        dynamic_cast<MpiOperable<Dtype>*>(layer.get());

    if (getRank() == 0) {  /* is master */
      layer->SetUp(bottom_vecs, top_vecs);
      if (__as_mpi__) {
        BroadcastBlobs(top_vecs, 0);
      }
    } else {               /* is slave */
      if (__as_mpi__) {
        BroadcastBlobs(top_vecs, 0);

        MpiFcLayer<Dtype>* __as_mpi_fc__ =
            dynamic_cast<MpiFcLayer<Dtype>*>(layer.get());
        if (__as_mpi_fc__) {
          layer->SetUp(bottom_vecs, sliced_top_vecs);
        }
      }
    }
  }

  template <typename Dtype>
  void Scheduler<Dtype>::InputDebugInfo(Net<Dtype>* net_, const int input_id) {
    vector<Blob<Dtype>*>& net_input_blobs_ = net_->net_input_blobs_;
    vector<string>& blob_names_ = net_->blob_names_;
    vector<int>& net_input_blob_indices_ = net_->net_input_blob_indices_;

    const Blob<Dtype>& blob = *net_input_blobs_[input_id];
    const string& blob_name = blob_names_[net_input_blob_indices_[input_id]];
    const Dtype data_abs_val_mean = blob.asum_data() / blob.count();
    LOG_IF(INFO, Caffe::root_solver())
    << "    [Forward] "
    << "Input " << blob_name << " data: " << data_abs_val_mean;
  }

  template <typename Dtype>
  void Scheduler<Dtype>::ForwardDebugInfo(Net<Dtype>* net_, const int layer_id) {
    vector<vector<Blob<Dtype>*> >& bottom_vecs_ = net_->bottom_vecs_;
    vector<vector<Blob<Dtype>*> >& top_vecs_ = net_->top_vecs_;
    vector<string>& blob_names_ = net_->blob_names_;
    vector<vector<int> >& top_id_vecs_ = net_->top_id_vecs_;
    vector<string>& layer_names_ = net_->layer_names_;
    vector<shared_ptr<Layer<Dtype> > >& layers_ = net_->layers_;
    vector<vector<int> >& param_id_vecs_ = net_->param_id_vecs_;
    vector<string>& param_display_names_ = net_->param_display_names_;

    for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) {
      const Blob<Dtype>& blob = *top_vecs_[layer_id][top_id];
      const string& blob_name = blob_names_[top_id_vecs_[layer_id][top_id]];
      const Dtype data_abs_val_mean = blob.asum_data() / blob.count();
      LOG_IF(INFO, Caffe::root_solver())
      << "    [Forward] "
      << "Layer " << layer_names_[layer_id]
      << ", top blob " << blob_name
      << " data: " << data_abs_val_mean;
    }
    for (int param_id = 0; param_id < layers_[layer_id]->blobs().size();
         ++param_id) {
      const Blob<Dtype>& blob = *layers_[layer_id]->blobs()[param_id];
      const int net_param_id = param_id_vecs_[layer_id][param_id];
      const string& blob_name = param_display_names_[net_param_id];
      const Dtype data_abs_val_mean = blob.asum_data() / blob.count();
      LOG_IF(INFO, Caffe::root_solver())
      << "    [Forward] "
      << "Layer " << layer_names_[layer_id]
      << ", param blob " << blob_name
      << " data: " << data_abs_val_mean;
    }
  }

  template <typename Dtype>
  void Scheduler<Dtype>::BackwardDebugInfo(Net<Dtype>* net_, const int layer_id) {
    vector<vector<Blob<Dtype>*> >& bottom_vecs_ = net_->bottom_vecs_;
    vector<string>& blob_names_ = net_->blob_names_;
    vector<string>& layer_names_ = net_->layer_names_;
    vector<shared_ptr<Layer<Dtype> > >& layers_ = net_->layers_;
    vector<vector<bool> >& bottom_need_backward_ = net_->bottom_need_backward_;
    vector<vector<int> >& bottom_id_vecs_ = net_->bottom_id_vecs_;

    const vector<Blob<Dtype>*>& bottom_vec = bottom_vecs_[layer_id];
    for (int bottom_id = 0; bottom_id < bottom_vec.size(); ++bottom_id) {
      if (!bottom_need_backward_[layer_id][bottom_id]) { continue; }
      const Blob<Dtype>& blob = *bottom_vec[bottom_id];
      const string& blob_name = blob_names_[bottom_id_vecs_[layer_id][bottom_id]];
      const Dtype diff_abs_val_mean = blob.asum_diff() / blob.count();
      LOG_IF(INFO, Caffe::root_solver())
      << "    [Backward] "
      << "Layer " << layer_names_[layer_id]
      << ", bottom blob " << blob_name
      << " diff: " << diff_abs_val_mean;
    }
    for (int param_id = 0; param_id < layers_[layer_id]->blobs().size();
         ++param_id) {
      if (!layers_[layer_id]->param_propagate_down(param_id)) { continue; }
      const Blob<Dtype>& blob = *layers_[layer_id]->blobs()[param_id];
      const Dtype diff_abs_val_mean = blob.asum_diff() / blob.count();
      LOG_IF(INFO, Caffe::root_solver())
      << "    [Backward] "
      << "Layer " << layer_names_[layer_id]
      << ", param blob " << param_id
      << " diff: " << diff_abs_val_mean;
    }
  }

  template <typename Dtype>
  Dtype Scheduler<Dtype>::ForwardFromTo(Net<Dtype>* net_, int start, int end) {
    bool debug_info_ = net_->debug_info_;
    vector<shared_ptr<Layer<Dtype> > >& layers_ = net_->layers_;
    vector<Blob<Dtype>*>& net_input_blobs_ = net_->net_input_blobs_;

    CHECK_GE(start, 0);
    CHECK_LT(end, layers_.size());
    Dtype loss = 0;
    if (debug_info_) {
      for (int i = 0; i < net_input_blobs_.size(); ++i) {
        InputDebugInfo(net_, i);
      }
    }

    for (int i = start; i <= end; ++i) {
      shared_ptr<Layer<Dtype> >& layer = layers_[i];
      MpiOperable<Dtype>* __as_mpi__ =
          dynamic_cast<MpiOperable<Dtype>*>(layer.get());

      if (__as_mpi__) {
        MpiFcLayer<Dtype>* __as_mpi_fc__ =
            dynamic_cast<MpiFcLayer<Dtype>*>(layer.get());
        if (__as_mpi_fc__) {            /* MpiFcLayer */
          MpiFcOperator<Dtype>(net_).Forward(i);
        }
      } else {                          /* Normal layer */
        Operator<Dtype>(net_).Forward(i);
      }

      if (debug_info_ && getRank() == 0) {
        ForwardDebugInfo(net_, i);
      }
    }
    return loss;
  }

  template <typename Dtype>
  void Scheduler<Dtype>::BroadcastLoss(Dtype& loss, int root) {
    broadcast(*world, loss, root);
  }

  template <typename Dtype>
  void Scheduler<Dtype>::BroadcastBlobs(vector<Blob<Dtype>*>& blobs, int root) {
    int blob_cnt = blobs.size();
    for (int i = 0; i < blob_cnt; i++) {
      BroadcastBlob(*blobs[i], root);
    }
  }

  template <typename Dtype>
  void Scheduler<Dtype>::BroadcastBlob(Blob<Dtype>& blob, int root) {
    shared_ptr<BlobProto> blobProto(new BlobProto());
    string blob_str;
    if (world->rank() == root) {
      // TODO: better serialization
      blob.ToProto(blobProto.get(), true);
      blob_str = blobProto->SerializeAsString();
    }

    broadcast(*world, blob_str, root);

    if (world->rank() != root) {
      blobProto->ParseFromString(blob_str);
      blob.FromProto(*blobProto, true);
    }
  }

  template <typename Dtype>
  void Scheduler<Dtype>::SendBlob(int dst, int tag, Blob<Dtype>& blob) {
    shared_ptr<BlobProto> blobProto(new BlobProto());
    blob.ToProto(blobProto.get(), true);
    string blob_str = blobProto->SerializeAsString();
    world->send(dst, tag, blob_str);
  }

  template <typename Dtype>
  void Scheduler<Dtype>::SendBlobs(int dst,
                                   vector<Blob<Dtype>*>& blobs) {
    int blob_cnt = blobs.size();
    for (int i = 0; i < blob_cnt; i++) {
      SendBlob(dst, i, *blobs[i]);
    }
  }

  template <typename Dtype>
  void Scheduler<Dtype>::RecvBlob(int src, int tag, Blob<Dtype>& blob) {
    string blob_str;
    world->recv(src, tag, blob_str);
    shared_ptr<BlobProto> blobProto(new BlobProto());
    blobProto->ParseFromString(blob_str);
    blob.FromProto(*blobProto, true);
  }

  template <typename Dtype>
  void Scheduler<Dtype>::RecvBlobs(int src,
                                   vector<Blob<Dtype>*>& blobs) {
    int blob_cnt = blobs.size();
    for (int i = 0; i < blob_cnt; i++) {
      RecvBlob(src, i, *blobs[i]);
    }
  }

  template <typename Dtype>
  void Scheduler<Dtype>::SendLoss(int dst, Dtype& loss) {
    world->send(dst, TAG_LOSS, loss);
  }

  template <typename Dtype>
  void Scheduler<Dtype>::RecvLoss(int src, Dtype& loss) {
    world->recv(src, TAG_LOSS, loss);
  }

  template <typename Dtype>
  void Scheduler<Dtype>::BackwardFromTo(Net<Dtype>* net_, int start, int end) {
    bool debug_info_ = net_->debug_info_;
    vector<shared_ptr<Layer<Dtype> > >& layers_ = net_->layers_;
    vector<bool>& layer_need_backward_ = net_->layer_need_backward_;

    CHECK_GE(end, 0);
    CHECK_LT(start, layers_.size());

    for (int i = start; i >= end; --i) {
      if (!layer_need_backward_[i]) {
        continue;
      }

      shared_ptr<Layer<Dtype> >& layer = layers_[i];
      MpiOperable<Dtype>* __as_mpi__ =
          dynamic_cast<MpiOperable<Dtype>*>(layer.get());

      if (__as_mpi__) {
        MpiFcLayer<Dtype>* __as_mpi_fc__ =
            dynamic_cast<MpiFcLayer<Dtype>*>(layer.get());
        if (__as_mpi_fc__) {          /* MpiFcLayer */
          MpiFcOperator<Dtype>(net_).Backward(i);
        }
      } else {                        /* Normal layer */
        Operator<Dtype>(net_).Backward(i);
      }

      // only master prints debug_info
      if (debug_info_ && getRank() == 0) {
        BackwardDebugInfo(net_, i);
      }
    }
  }

  INSTANTIATE_CLASS(Scheduler);
}
