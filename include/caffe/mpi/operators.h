/* @name forwarders
 * @brief do forward/backward for scheduler
 *
 * @author misaka-10032 (longqic@andrew.cmu.edu)
 */

#ifndef CAFFE_FORWARDERS_H
#define CAFFE_FORWARDERS_H

#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include "caffe/net.hpp"
#include "caffe/mpi/scheduler.h"

namespace mpi = boost::mpi;

namespace caffe {
  template<typename Dtype> class Net;
  template<typename Dtype> class Scheduler;

  /**
   * Normal forwarder
   */
  template<typename Dtype>
  class Operator {
  public:
    explicit Operator(Net<Dtype>* net);
    virtual Dtype Forward(int layer_id);
    virtual void Backward(int layer_id);

  protected:
    Net<Dtype>* net_;
    Scheduler<Dtype>* scheduler_;
    int slave_cnt_;
    int rank_;
  };

  /**
   * Forwarder for MpiFcLayer
   */
  template<typename Dtype>
  class MpiFcOperator : public Operator<Dtype> {
  public:
    explicit MpiFcOperator(Net<Dtype>* net) : Operator<Dtype>(net) {}
    virtual Dtype Forward(int layer_id);
    virtual void Backward(int layer_id);
  };
}

#endif //CAFFE_FORWARDERS_H
