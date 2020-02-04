#ifdef USE_CUDNN
#include <vector>

#include "caffe/layers/cudnn_deconv_layer.hpp"

namespace caffe {

__global__ void sync_deconv_groups() {}

template <typename Dtype>
void CuDNNDeconvolutionLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();

#if CUDNN_VERSION_MIN(7,0,0)
    if (multiple_handles_)
      {
#endif

        // Forward through cuDNN in parallel over groups.
        for (int g = 0; g < this->group_; g++) {
          // Filters.
          CUDNN_CHECK(cudnnConvolutionBackwardData(
                                                   handle_[g],
                                                   cudnn::dataType<Dtype>::one,
                                                   filter_desc_,
                                                   weight + this->weight_offset_ * g,
                                                   bottom_descs_[i],
                                                   bottom_data + bottom_offset_ * g,
                                                   conv_descs_[i],
                                                   bwd_data_algo_[i],
                                                   workspace[g],
                                                   workspace_bwd_data_sizes_[i],
                                                   cudnn::dataType<Dtype>::zero,
                                                   top_descs_[i],
                                                   top_data + top_offset_ * g));

          // Bias.
          if (this->bias_term_) {
            const Dtype* bias_data = this->blobs_[1]->gpu_data();
            CUDNN_CHECK(cudnnAddTensor(handle_[g],
                                       cudnn::dataType<Dtype>::one,
                                       bias_desc_,
                                       bias_data + bias_offset_ * g,
                                       cudnn::dataType<Dtype>::one,
                                       top_descs_[i],
                                       top_data + top_offset_ * g));
          }
        }

        // Synchronize the work across groups, each of which went into its own
        // stream, by launching an empty kernel into the default (null) stream.
        // NOLINT_NEXT_LINE(whitespace/operators)
        sync_deconv_groups<<<1, 1>>>();
#if CUDNN_VERSION_MIN(7,0,0)
      }
    else
      {
        CUDNN_CHECK(cudnnConvolutionBackwardData(
                                                 handle_[0],
                                                 cudnn::dataType<Dtype>::one,
                                                 filter_desc_,
                                                 weight,
                                                 bottom_descs_[i],
                                                 bottom_data,
                                                 conv_descs_[i],
                                                 bwdDataPerf_[i].algo,
                                                 workspaceData,
                                                 bwdDataPerf_[i].memory,
                                                 cudnn::dataType<Dtype>::zero,
                                                 top_descs_[i],
                                                 top_data));

        // Bias.
        if (this->bias_term_) {
          const Dtype* bias_data = this->blobs_[1]->gpu_data();
          CUDNN_CHECK(cudnnAddTensor(handle_[0],
                                     cudnn::dataType<Dtype>::one,
                                     bias_desc_,
                                     bias_data,
                                     cudnn::dataType<Dtype>::one,
                                     top_descs_[i],
                                     top_data));
        }
      }
#endif
  }
}

template <typename Dtype>
void CuDNNDeconvolutionLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = NULL;
  Dtype* weight_diff = NULL;
  if (this->param_propagate_down_[0]) {
    weight = this->blobs_[0]->gpu_data();
    weight_diff = this->blobs_[0]->mutable_gpu_diff();
  }
  Dtype* bias_diff = NULL;
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    bias_diff = this->blobs_[1]->mutable_gpu_diff();
  }
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Backward through cuDNN in parallel over groups and gradients.


#if CUDNN_VERSION_MIN(7,0,0)
    if (multiple_handles_)
      {
#endif

        for (int g = 0; g < this->group_; g++) {
          // Gradient w.r.t. bias.
          if (this->bias_term_ && this->param_propagate_down_[1]) {
            CUDNN_CHECK(cudnnConvolutionBackwardBias(handle_[0 * this->group_ + g],
                                                     cudnn::dataType<Dtype>::one,
                                                     top_descs_[i],
                                                     top_diff + top_offset_ * g,
                                                     cudnn::dataType<Dtype>::one,
                                                     bias_desc_,
                                                     bias_diff + bias_offset_ * g));
          }

          // Gradient w.r.t. weights.
          if (this->param_propagate_down_[0]) {
            const Dtype* bottom_data = bottom[i]->gpu_data();
            CUDNN_CHECK(cudnnConvolutionBackwardFilter(
                                                       handle_[1 * this->group_ + g],
                                                       cudnn::dataType<Dtype>::one,
                                                       top_descs_[i],
                                                       top_diff + top_offset_ * g,
                                                       bottom_descs_[i],
                                                       bottom_data + bottom_offset_ * g,
                                                       conv_descs_[i],
                                                       bwd_filter_algo_[i],
                                                       workspace[1 * this->group_ + g],
                                                       workspace_bwd_filter_sizes_[i],
                                                       cudnn::dataType<Dtype>::one,
                                                       filter_desc_,
                                                       weight_diff + this->weight_offset_ * g));
          }

          // Gradient w.r.t. bottom data.
          if (propagate_down[i]) {
            if (weight == NULL) {
              weight = this->blobs_[0]->gpu_data();
            }
            Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
            CUDNN_CHECK(
                        cudnnConvolutionForward(handle_[2 * this->group_ + g],
                                                cudnn::dataType<Dtype>::one,
                                                top_descs_[i],
                                                top_diff + top_offset_ * g,
                                                filter_desc_,
                                                weight + this->weight_offset_ * g,
                                                conv_descs_[i],
                                                fwd_algo_[i],
                                                workspace[2 * this->group_ + g],
                                                workspace_fwd_sizes_[i],
                                                cudnn::dataType<Dtype>::zero,
                                                bottom_descs_[i],
                                                bottom_diff + bottom_offset_ * g));
          }
        }

        // Synchronize the work across groups, each of which went into its own
        // stream, by launching an empty kernel into the default (null) stream.
        // NOLINT_NEXT_LINE(whitespace/operators)
        sync_deconv_groups<<<1, 1>>>();
#if CUDNN_VERSION_MIN(7,0,0)
      }
    else
      {
        if (this->bias_term_ && this->param_propagate_down_[1]) {
          CUDNN_CHECK(cudnnConvolutionBackwardBias(handle_[0],
                                                   cudnn::dataType<Dtype>::one,
                                                   top_descs_[i],
                                                   top_diff,
                                                   cudnn::dataType<Dtype>::one,
                                                   bias_desc_,
                                                   bias_diff));
      }

        // Gradient w.r.t. weights.
        if (this->param_propagate_down_[0]) {
          const Dtype* bottom_data = bottom[i]->gpu_data();
          CUDNN_CHECK(cudnnConvolutionBackwardFilter(
                                                     handle_[0],
                                                     cudnn::dataType<Dtype>::one,
                                                     top_descs_[i],
                                                     top_diff,
                                                     bottom_descs_[i],
                                                     bottom_data,
                                                     conv_descs_[i],
                                                     bwdFilterPerf_[i].algo,
                                                     workspaceData,
                                                     bwdFilterPerf_[i].memory,
                                                     cudnn::dataType<Dtype>::one,
                                                     filter_desc_,
                                                     weight_diff));
          }

          // Gradient w.r.t. bottom data.
        if (propagate_down[i]) {
          if (weight == NULL) {
            weight = this->blobs_[0]->gpu_data();
          }
          Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
          CUDNN_CHECK(cudnnConvolutionForward(handle_[0],
                                              cudnn::dataType<Dtype>::one,
                                              top_descs_[i],
                                              top_diff,
                                              filter_desc_,
                                              weight,
                                              conv_descs_[i],
                                              fwdPerf_[i].algo,
                                              workspaceData,
                                              fwdPerf_[i].memory,
                                              cudnn::dataType<Dtype>::zero,
                                              bottom_descs_[i],
                                              bottom_diff));
        }
      }
#endif
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CuDNNDeconvolutionLayer);

}  // namespace caffe
#endif
