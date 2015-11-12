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

  template<typename Dtype>
  class Scheduler {
  public:
    static Scheduler<Dtype> *Init(mpi::environment* env,
                                  mpi::communicator* world);
    static Scheduler<Dtype> *Get();

    inline int getParallelism() {
      return world->size() - 1;  // only slaves work in parallel
    }

    inline int getRank() {
      return world->rank();
    }

    inline void setNet(Net<Dtype> *net_) {
      this->net_ = net_;
    }

    void InputDebugInfo(const int input_id);
    void ForwardDebugInfo(const int layer_id);
    void BackwardDebugInfo(const int layer_id);

    void SetUpLayer(int layer_id);

    Dtype ForwardFromTo(int start, int end);
    void BackwardFromTo(int start, int end);

    void BroadcastBlob(Blob<Dtype>& blob, int root);
    void BroadcastBlobs(vector<Blob<Dtype>*>& blobs, int root);
    void SendBlob(int dst, int tag, Blob<Dtype>& blob);
    void RecvBlob(int src, int tag, Blob<Dtype>& blob);

    inline void sync() {
      int s;
      const int ACK = 1;
      if (world->rank() == 0) {
        s = 1;
        LOG(INFO) << "master set sync as 1";
      }

      broadcast(*world, s, 0);

      if (world->rank() == 0) {
        int size = world->size();
        for (int i = 1; i < size; i++) {
          world->recv(i, ACK);
        }
      } else {
        LOG(INFO) << "slave recv sync as " << s;
        world->send(0, ACK);
      }
    }

  protected:
    explicit Scheduler<Dtype>();

    shared_ptr<mpi::environment> env;
    shared_ptr<mpi::communicator> world;

  private:
    static const int TAG_BLOB_PIECE = 1;
    static const int TAG_LOSS = 200;
    static shared_ptr<Scheduler<Dtype> > instance_;
    static std::mutex mutex_;
    Net<Dtype> *net_;
  };
}
#endif //CAFFE_SCHEDULER_H
