#include "caffe/layers/focal_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe
{

template <typename Dtype>
void FocalLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top)
{
    LossLayer<Dtype>::LayerSetUp(bottom, top);
    // alphas_ = this->layer_param().focal_loss_param().alpha();
    const FocalLossParameter& focal_param = this->layer_param_.focal_loss_param();
    alphas_.clear();
    std::copy(focal_param.alpha().begin(),
        focal_param.alpha().end(),
        std::back_inserter(alphas_));

    gamma_ = focal_param.gamma();
    batch_size = bottom[0]->count(0);
    cls_cnt = bottom[0]->count(1);
}

template <typename Dtype>
void FocalLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top)
{
    vector<int> loss_shape(1, 1);
    top[0]->Reshape(loss_shape);
}

template <typename Dtype>
void FocalLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top)
{
    const Dtype *prob_data = bottom[0]->cpu_data();
    const Dtype *label = bottom[1]->cpu_data();
    Dtype loss = 0;

    for (int i = 0; i < batch_size; ++i)
    {
        const int label_value = static_cast<int>(label[i]);
        const Dtype prob = prob_data[i*cls_cnt + label_value];
        loss += -alphas_[label_value] * powf((1 - prob), gamma_) * log(prob) / batch_size;
    }
    top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void FocalLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top,
                                                    const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom)
{
    if (propagate_down[1])
    {
        LOG(FATAL) << this->type()
                   << " Layer cannot backpropagate to label inputs.";
    }
    if (propagate_down[0])
    {
        Dtype *bottom_diff = bottom[0]->mutable_cpu_diff();
        const Dtype *prob_data = bottom[0]->cpu_data();
        const Dtype *label = bottom[1]->cpu_data();

        caffe_set(batch_size * cls_cnt, Dtype(0), bottom_diff);

        for (int i = 0; i < batch_size; ++i)
        {
            const int label_value = static_cast<int>(label[i]);
            const Dtype prob = prob_data[i*cls_cnt + label_value];
            bottom_diff[i * cls_cnt + label_value] = -alphas_[label_value] * (-gamma_ * powf((1-prob), gamma_ - 1) * log(prob) 
                                                    + powf((1-prob), gamma_) / prob);
        }
    }
}

#ifdef CPU_ONLY
STUB_GPU(FocalLossLayer);
#endif

INSTANTIATE_CLASS(FocalLossLayer);
REGISTER_LAYER_CLASS(FocalLoss);

} // namespace caffe