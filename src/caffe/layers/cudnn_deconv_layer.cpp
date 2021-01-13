#ifdef USE_CUDNN
#include <algorithm>
#include <vector>

#include "caffe/layers/cudnn_deconv_layer.hpp"

namespace caffe {

// Set to three for the benefit of the backward pass, which
// can use separate streams for calculating the gradient w.r.t.
// bias, filter weights, and bottom data for each group independently
#define CUDNN_STREAMS_PER_GROUP 3

/**
 * TODO(dox) explain cuDNN interface
 */
template <typename Dtype>
void CuDNNDeconvolutionLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  DeconvolutionLayer<Dtype>::LayerSetUp(bottom, top);

  workspaceSizeInBytes = 0;
  workspaceData = NULL;

#if CUDNN_VERSION_MIN(7,0,0)
  if (multiple_handles_)
    {
#endif
      // Initialize CUDA streams and cuDNN.
      stream_         = new cudaStream_t[this->group_ * CUDNN_STREAMS_PER_GROUP];
      handle_         = new cudnnHandle_t[this->group_ * CUDNN_STREAMS_PER_GROUP];

      // Initialize algorithm arrays
      fwd_algo_       = new cudnnConvolutionFwdAlgo_t[bottom.size()];
      bwd_filter_algo_= new cudnnConvolutionBwdFilterAlgo_t[bottom.size()];
      bwd_data_algo_  = new cudnnConvolutionBwdDataAlgo_t[bottom.size()];

      // initialize size arrays
      workspace_fwd_sizes_ = new size_t[bottom.size()];
      workspace_bwd_filter_sizes_ = new size_t[bottom.size()];
      workspace_bwd_data_sizes_ = new size_t[bottom.size()];

      // workspace data
      workspace = new void*[this->group_ * CUDNN_STREAMS_PER_GROUP];

      for (size_t i = 0; i < bottom.size(); ++i) {
        // initialize all to default algorithms
        fwd_algo_[i] = (cudnnConvolutionFwdAlgo_t)0;
        bwd_filter_algo_[i] = (cudnnConvolutionBwdFilterAlgo_t)0;
        bwd_data_algo_[i] = (cudnnConvolutionBwdDataAlgo_t)0;
        // default algorithms don't require workspace
        workspace_fwd_sizes_[i] = 0;
        workspace_bwd_data_sizes_[i] = 0;
        workspace_bwd_filter_sizes_[i] = 0;
      }

      for (int g = 0; g < this->group_ * CUDNN_STREAMS_PER_GROUP; g++) {
        CAFFE1_CUDA_CHECK(cudaStreamCreate(&stream_[g]));
        CUDNN_CHECK(cudnnCreate(&handle_[g]));
        CUDNN_CHECK(cudnnSetStream(handle_[g], stream_[g]));
        workspace[g] = NULL;
      }

      // Set the indexing parameters.
      bias_offset_ = (this->num_output_ / this->group_);

#if CUDNN_VERSION_MIN(7,0,0)
    }
  else
    {
      handle_         = new cudnnHandle_t[1];
      fwdPerf_.resize(bottom.size());
      bwdFilterPerf_.resize(bottom.size());
      bwdDataPerf_.resize(bottom.size());
      CUDNN_CHECK(cudnnCreate(&handle_[0]));
    }
#endif


  // Create filter descriptor.
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int kernel_h = kernel_shape_data[0];
  const int kernel_w = kernel_shape_data[1];

#if CUDNN_VERSION_MIN(7,0,0)
  if (multiple_handles_)
#endif
    cudnn::createFilterDesc<Dtype>(&filter_desc_,
                                   this->channels_ / this->group_,
                                   this->num_output_ / this->group_,
                                   kernel_h,
                                   kernel_w);
#if CUDNN_VERSION_MIN(7,0,0)
  else
    // GUILLO : if cudnn problems, that may likely be that the third argument should be divided by group, and not the second (hard to tell offline as everything is reversed)
    cudnn::createFilterDesc<Dtype>(&filter_desc_, this->channels_ / this->group_, this->num_output_,
                                   kernel_h, kernel_w);
#endif


  // Create tensor descriptor(s) for data and corresponding convolution(s).
  for (int i = 0; i < bottom.size(); i++) {
    cudnnTensorDescriptor_t bottom_desc;
    cudnn::createTensor4dDesc<Dtype>(&bottom_desc);
    bottom_descs_.push_back(bottom_desc);
    cudnnTensorDescriptor_t top_desc;
    cudnn::createTensor4dDesc<Dtype>(&top_desc);
    top_descs_.push_back(top_desc);
    cudnnConvolutionDescriptor_t conv_desc;
    cudnn::createConvolutionDesc<Dtype>(&conv_desc);
    conv_descs_.push_back(conv_desc);
  }

  // Tensor descriptor for bias.
  if (this->bias_term_) {
    cudnn::createTensor4dDesc<Dtype>(&bias_desc_);
  }

  handles_setup_ = true;
}

template <typename Dtype>
void CuDNNDeconvolutionLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  int old_num = this->num_;
  DeconvolutionLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(2, this->num_spatial_axes_)
      << "CuDNNDeconvolutionLayer input must have 2 spatial axes "
      << "(e.g., height and width). "
      << "Use 'engine: CAFFE' for general ND convolution.";

  // We don't look for best cudnn algorithm if already set
  // and batch size did not change (other dims should never change).
    if (algo_set_ && old_num == this->num_)
      return;

#if CUDNN_VERSION_MIN(7,0,0)
  if (multiple_handles_)
    {
#endif
      bottom_offset_ = this->bottom_dim_ / this->group_;
      top_offset_ = this->top_dim_ / this->group_;
#if CUDNN_VERSION_MIN(7,0,0)
    }
#endif

  const int height = bottom[0]->shape(this->channel_axis_ + 1);
  const int width = bottom[0]->shape(this->channel_axis_ + 2);
  const int height_out = top[0]->shape(this->channel_axis_ + 1);
  const int width_out = top[0]->shape(this->channel_axis_ + 2);
  const int* pad_data = this->pad_.cpu_data();
  const int pad_h = pad_data[0];
  const int pad_w = pad_data[1];
  const int* stride_data = this->stride_.cpu_data();
  const int stride_h = stride_data[0];
  const int stride_w = stride_data[1];

  // Specify workspace limit for kernels directly until we have a
  // planning strategy and a rewrite of Caffe's GPU memory mangagement

  for (int i = 0; i < bottom.size(); i++) {
#if CUDNN_VERSION_MIN(7,0,0)
    if (multiple_handles_)
      {
#endif
        size_t workspace_limit_bytes = 8*1024*1024;
        cudnn::setTensor4dDesc<Dtype>(&bottom_descs_[i],
                                      this->num_,
                                      this->channels_ / this->group_,
                                      height,
                                      width,
                                      this->channels_ * height * width,
                                      height * width,
                                      width,
                                      1);
        cudnn::setTensor4dDesc<Dtype>(&top_descs_[i],
                                      this->num_,
                                      this->num_output_ / this->group_,
                                      height_out,
                                      width_out,
                                      this->num_output_ * height_out * width_out,
                                      height_out * width_out,
                                      width_out,
                                      1);
        cudnn::setConvolutionDesc<Dtype>(&conv_descs_[i],
                                         top_descs_[i],
                                         filter_desc_,
                                         pad_h,
                                         pad_w,
                                         stride_h,
                                         stride_w);

        // choose forward and backward algorithms + workspace(s)
#if CUDNN_VERSION_MIN(8,0,0)
#else
        CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(
                                                        handle_[0],
                                                        top_descs_[i],
                                                        filter_desc_,
                                                        conv_descs_[i],
                                                        bottom_descs_[i],
                                                        CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
                                                        workspace_limit_bytes,
                                                        &fwd_algo_[i]));

        // We have found that CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM is
        // buggy. Thus, if this algo was chosen, choose winograd instead. If
        // winograd is not supported or workspace is larger than threshold, choose
        // implicit_gemm instead.
        if (fwd_algo_[i] == CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM) {
          size_t winograd_workspace_size;
          cudnnStatus_t status = cudnnGetConvolutionForwardWorkspaceSize(
                                                                         handle_[0],
                                                                         top_descs_[i],
                                                                         filter_desc_,
                                                                         conv_descs_[i],
                                                                         bottom_descs_[i],
                                                                         CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
                                                                         &winograd_workspace_size);
          if (status != CUDNN_STATUS_SUCCESS ||
              winograd_workspace_size >= workspace_limit_bytes) {
            fwd_algo_[i] = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
          } else {
            fwd_algo_[i] = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
          }
        }

        CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
                                                            handle_[0],
                                                            top_descs_[i],
                                                            filter_desc_,
                                                            conv_descs_[i],
                                                            bottom_descs_[i],
                                                            fwd_algo_[i],
                                                            &(workspace_fwd_sizes_[i])));
#endif
        // choose backward algorithm for filter
#if CUDNN_VERSION_MIN(8,0,0)
#else	
        CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm(
                                                               handle_[0],
                                                               top_descs_[i],
                                                               bottom_descs_[i],
                                                               conv_descs_[i],
                                                               filter_desc_,
                                                               CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
                                                               workspace_limit_bytes,
                                                               &bwd_filter_algo_[i]));
	
        // get workspace for backwards filter algorithm
        CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(
                                                                   handle_[0],
                                                                   top_descs_[i],
                                                                   bottom_descs_[i],
                                                                   conv_descs_[i],
                                                                   filter_desc_,
                                                                   bwd_filter_algo_[i],
                                                                   &workspace_bwd_filter_sizes_[i]));
#endif

        // choose backward algo for data
#if CUDNN_VERSION_MIN(8,0,0)
#else
        CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(
                                                             handle_[0],
                                                             filter_desc_,
                                                             bottom_descs_[i],
                                                             conv_descs_[i],
                                                             top_descs_[i],
                                                             CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
                                                             workspace_limit_bytes,
                                                             &bwd_data_algo_[i]));
	
        // get workspace size
        CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(
                                                                 handle_[0],
                                                                 filter_desc_,
                                                                 bottom_descs_[i],
                                                                 conv_descs_[i],
                                                                 top_descs_[i],
                                                                 bwd_data_algo_[i],
                                                                 &workspace_bwd_data_sizes_[i]));
#endif

        // reduce over all workspace sizes to get a maximum to allocate / reallocate
        size_t total_workspace_fwd = 0;
        size_t total_workspace_bwd_data = 0;
        size_t total_workspace_bwd_filter = 0;

        for (size_t i = 0; i < bottom.size(); i++) {
          total_workspace_fwd        = std::max(total_workspace_fwd,
                                                workspace_fwd_sizes_[i]);
          total_workspace_bwd_data   = std::max(total_workspace_bwd_data,
                                                workspace_bwd_data_sizes_[i]);
          total_workspace_bwd_filter = std::max(total_workspace_bwd_filter,
                                                workspace_bwd_filter_sizes_[i]);
        }
        // get max over all operations
        size_t max_workspace = std::max(total_workspace_fwd,
                                        total_workspace_bwd_data);
        max_workspace = std::max(max_workspace, total_workspace_bwd_filter);
        // ensure all groups have enough workspace
        size_t total_max_workspace = max_workspace *
          (this->group_ * CUDNN_STREAMS_PER_GROUP);

        // this is the total amount of storage needed over all groups + streams
        if (total_max_workspace > workspaceSizeInBytes) {
          DLOG(INFO) << "Reallocating workspace storage: " << total_max_workspace;
          workspaceSizeInBytes = total_max_workspace;

          // free the existing workspace and allocate a new (larger) one
          cudaFree(this->workspaceData);

          cudaError_t err = cudaMalloc(&(this->workspaceData), workspaceSizeInBytes);
          if (err != cudaSuccess) {
            // force zero memory path
            for (int i = 0; i < bottom.size(); i++) {
              workspace_fwd_sizes_[i] = 0;
              workspace_bwd_filter_sizes_[i] = 0;
              workspace_bwd_data_sizes_[i] = 0;
              fwd_algo_[i] = CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING;
              bwd_filter_algo_[i] = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
              bwd_data_algo_[i] = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
            }

            // NULL out all workspace pointers
            for (int g = 0; g < (this->group_ * CUDNN_STREAMS_PER_GROUP); g++) {
              workspace[g] = NULL;
            }
            // NULL out underlying data
            workspaceData = NULL;
            workspaceSizeInBytes = 0;
          }

          // if we succeed in the allocation, set pointer aliases for workspaces
          for (int g = 0; g < (this->group_ * CUDNN_STREAMS_PER_GROUP); g++) {
            workspace[g] = reinterpret_cast<char *>(workspaceData) + g*max_workspace;
          }
        }
#if CUDNN_VERSION_MIN(7,0,0)
      }
    else
      {
        cudnn::setTensor4dDesc<Dtype>(&bottom_descs_[i],
                                      this->num_,
                                      this->channels_,
                                      height,
                                      width,
                                      this->channels_ * height * width,
                                      height * width,
                                      width,
                                      1);
        cudnn::setTensor4dDesc<Dtype>(&top_descs_[i],
                                      this->num_,
                                      this->num_output_,
                                      height_out,
                                      width_out,
                                      this->num_output_ * height_out * width_out,
                                      height_out * width_out,
                                      width_out,
                                      1);
        cudnn::setConvolutionDesc<Dtype>(&conv_descs_[i],
                                         top_descs_[i],
                                         filter_desc_,
                                         pad_h,
                                         pad_w,
                                         stride_h,
                                         stride_w);
        cudnnSetConvolutionGroupCount(conv_descs_[i], this->group_);


        int num_algs_fwd;
        cudnnGetConvolutionForwardAlgorithm_v7(handle_[0],
                                               top_descs_[i],
                                               filter_desc_,
                                               conv_descs_[i],
                                               bottom_descs_[i],
                                               1,
                                               &num_algs_fwd,
                                               &fwdPerf_[i]);
        if (fwdPerf_[i].memory > workspaceSizeInBytes)
          workspaceSizeInBytes = fwdPerf_[i].memory;

        cudnnSetConvolutionMathType(conv_descs_[i],fwdPerf_[i].mathType);

        if (!min_memory_)
          {
            int num_algs_bwd_filter;
            cudnnGetConvolutionBackwardFilterAlgorithm_v7(handle_[0],
                                                          top_descs_[i],
                                                          bottom_descs_[i],
                                                          conv_descs_[i],
                                                          filter_desc_,
                                                          1,
                                                          &num_algs_bwd_filter,
                                                          &bwdFilterPerf_[i]);

            if (bwdFilterPerf_[i].memory > workspaceSizeInBytes)
              workspaceSizeInBytes = bwdFilterPerf_[i].memory;

            int num_algs_bwd_data;
            cudnnGetConvolutionBackwardDataAlgorithm_v7(handle_[0],
                                                        filter_desc_,
                                                        bottom_descs_[i],
                                                        conv_descs_[i],
                                                        top_descs_[i],
                                                        1,
                                                        &num_algs_bwd_data,
                                                        &bwdDataPerf_[i]);

            if (bwdDataPerf_[i].memory > workspaceSizeInBytes)
              workspaceSizeInBytes = bwdDataPerf_[i].memory;
            cudaFree(workspaceData);
            cudaError_t err = cudaMalloc(&workspaceData, workspaceSizeInBytes);
            if (err != cudaSuccess)
              min_memory_ = true;
          }
        if (min_memory_)
          {
            fwdPerf_[i].algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
            fwdPerf_[i].memory = 0;
            bwdFilterPerf_[i].algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
            bwdFilterPerf_[i].memory = 0;
            bwdDataPerf_[i].algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
            bwdDataPerf_[i].memory = 0;
            workspaceData = NULL;
          }
      }
#endif
  }
  // Tensor descriptor for bias.
  if (this->bias_term_) {
#if CUDNN_VERSION_MIN(7,0,0)
    if (multiple_handles_)
#endif
      cudnn::setTensor4dDesc<Dtype>(
                                    &bias_desc_, 1, this->num_output_ / this->group_, 1, 1);
#if CUDNN_VERSION_MIN(7,0,0)
    else
      cudnn::setTensor4dDesc<Dtype>(&bias_desc_,
                                    1, this->num_output_, 1, 1);
#endif

  }
}

template <typename Dtype>
CuDNNDeconvolutionLayer<Dtype>::~CuDNNDeconvolutionLayer() {
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }

  for (int i = 0; i < bottom_descs_.size(); i++) {
    cudnnDestroyTensorDescriptor(bottom_descs_[i]);
    cudnnDestroyTensorDescriptor(top_descs_[i]);
    cudnnDestroyConvolutionDescriptor(conv_descs_[i]);
  }
  if (this->bias_term_) {
    cudnnDestroyTensorDescriptor(bias_desc_);
  }
  cudnnDestroyFilterDescriptor(filter_desc_);

  cudaFree(workspaceData);


#if CUDNN_VERSION_MIN(7,0,0)
  if (multiple_handles_)
    {
#endif
      for (int g = 0; g < this->group_ * CUDNN_STREAMS_PER_GROUP; g++) {
        cudaStreamDestroy(stream_[g]);
        cudnnDestroy(handle_[g]);
      }
      delete [] stream_;
      delete [] workspace;
      delete [] fwd_algo_;
      delete [] bwd_filter_algo_;
      delete [] bwd_data_algo_;
      delete [] workspace_fwd_sizes_;
      delete [] workspace_bwd_data_sizes_;
      delete [] workspace_bwd_filter_sizes_;
#if CUDNN_VERSION_MIN(7,0,0)
    }
  else
    {
      fwdPerf_.clear();
      bwdDataPerf_.clear();
      bwdFilterPerf_.clear();
      cudnnDestroy(handle_[0]);
    }
#endif

  delete [] handle_;

}

INSTANTIATE_CLASS(CuDNNDeconvolutionLayer);

}   // namespace caffe
#endif
