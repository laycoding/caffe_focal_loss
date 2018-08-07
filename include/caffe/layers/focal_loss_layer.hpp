#ifndef CAFFE_FOCAL_LOSS_LAYER_HPP_
#define CAFFE_FOCAL_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe
{
template <typename Dtype>
class FocalLossLayer : public LossLayer<Dtype>
{
  public:
    explicit FocalLossLayer(const LayerParameter &param)
        : LossLayer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                            const vector<Blob<Dtype> *> &top);
    virtual void Reshape(const vector<Blob<Dtype> *> &bottom,
                         const vector<Blob<Dtype> *> &top);

    virtual inline const char *type() const { return "FocalLoss"; }
    virtual inline int ExactNumBottomBlobs() const { return -1; }
    virtual inline int MinBottomBlobs() const { return 2; }
    virtual inline int MaxBottomBlobs() const { return 3; }
    virtual inline int ExactNumTopBlobs() const { return -1; }
    virtual inline int MinTopBlobs() const { return 1; }
    virtual inline int MaxTopBlobs() const { return 2; }

  protected:
    virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                             const vector<Blob<Dtype> *> &top);
    // virtual void Forward_gpu(const vector<Blob<Dtype> *> &bottom,
    //                          const vector<Blob<Dtype> *> &top);

    virtual void Backward_cpu(const vector<Blob<Dtype> *> &top,
                              const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom);
    // virtual void Backward_gpu(const vector<Blob<Dtype> *> &top,
    //                           const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom);



    int batch_size, cls_cnt;

    Blob<Dtype> scaler_;
    vector<Dtype> alphas_;
    Dtype gamma_;
    Dtype normalizer_;
};

} // namespace caffe

#endif