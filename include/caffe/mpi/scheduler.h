/* @name Scheduler
 * @brief Singleton scheduler.
 *
 * @author misaka-10032 (longqic@andrew.cmu.edu)
 * @bug Doesn't support share_from_root
 */

#ifndef CAFFE_SCHEDULER_H
#define CAFFE_SCHEDULER_H

#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include "caffe/net.hpp"

namespace mpi = boost::mpi;

namespace caffe {
  template<typename Dtype> class Net;

  /**
   * Uses all-reduce scheme. Require at least a master and a slave.
   * Master is responsible for collect and distribute data.
   * Slave is responsible for calculation.
   *
   * For model parallelism:
   * Master holds
   *   1) all the data
   *   2) non-mpi layer parameters
   * Slave (mainly for mpi layers) holds
   *   1) Its sliced data
   *   2) Its layer params (master doesn't hold!)
   *
   */
  template<typename Dtype>
  class Scheduler {
  public:
    static Scheduler<Dtype> *Init(mpi::environment* env,
                                  mpi::communicator* world);
    static Scheduler<Dtype> *Get();

    inline int getSlaveCnt() {
      return world->size() - 1;  // only slaves work in parallel
    }

    inline int getRank() {
      return world->rank();
    }

    void InputDebugInfo(Net<Dtype>* net, const int input_id);
    void ForwardDebugInfo(Net<Dtype>* net, const int layer_id);
    void BackwardDebugInfo(Net<Dtype>* net, const int layer_id);

    void SetUpLayer(Net<Dtype>* net, int layer_id);

    Dtype ForwardFromTo(Net<Dtype>* net, int start, int end);
    void BackwardFromTo(Net<Dtype>* net, int start, int end);

    void BroadcastLoss(Dtype& loss, int root);
    void BroadcastBlob(Blob<Dtype>& blob, int root);
    void BroadcastBlobs(vector<Blob<Dtype>*>& blobs, int root);
    void SendBlob(int dst, int tag, Blob<Dtype>& blob);
    void SendBlobs(int dst, vector<Blob<Dtype>*>& blobs);
    void RecvBlob(int src, int tag, Blob<Dtype>& blob);
    void RecvBlobs(int src, vector<Blob<Dtype>*>& blobs);
    void SendLoss(int dst, Dtype& loss);
    void RecvLoss(int src, Dtype& loss);

  protected:
    explicit Scheduler<Dtype>();
    shared_ptr<mpi::environment> env;
    shared_ptr<mpi::communicator> world;

  private:
    static const int TAG_BLOB_PIECE = 1;
    static const int TAG_LOSS = 200;
    static shared_ptr<Scheduler<Dtype> > instance_;
    static std::mutex mutex_;
  };
}
#endif //CAFFE_SCHEDULER_H
