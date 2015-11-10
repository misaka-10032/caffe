/* @name Scheduler
 * @brief Singleton scheduler.
 *
 * @author misaka-10032 (longqic@andrew.cmu.edu)
 * @bug Doesn't support share_from_root
 */

#ifndef CAFFE_SCHEDULER_H
#define CAFFE_SCHEDULER_H

#include <boost/mpi.hpp>
#include "caffe/net.hpp"

namespace mpi = boost::mpi;

namespace caffe {
  template<typename Dtype> class Net;

  template<typename Dtype>
  class Scheduler {
  public:
    static Scheduler<Dtype> *Get() {
      if (!instance_) {
        instance_ = new Scheduler<Dtype>();
      }
      return instance_;
    }

    inline int getParallelism() {
      return world.size() - 1;  // only slaves work in parallel
    }

    inline int getRank() {
      return world.rank();
    }

    inline void setNet(Net<Dtype> *net_) {
      this->net_ = net_;
    }

    void InputDebugInfo(const int input_id);
    void ForwardDebugInfo(const int layer_id);
    void BackwardDebugInfo(const int layer_id);

    Dtype ForwardFromTo(int start, int end);
    void BackwardFromTo(int start, int end);

    vector<int>& Scheduler<Dtype>::ShapeForSlave(const vector<int>& completeShape);

  protected:
    Scheduler<Dtype>() { }

    mpi::environment env;
    mpi::communicator world;

  private:
    static const int TAG_BLOB_PIECE = 1;
    static Scheduler<Dtype> *instance_;
    Net<Dtype> *net_;
  };
}
#endif //CAFFE_SCHEDULER_H
