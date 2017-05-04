#include <vector>

#include "caffe/data_layers.hpp"

namespace caffe {

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, join the thread
  JoinPrefetchThread();
  // Reshape to loaded data.
  top[0]->ReshapeLike(this->prefetch_data_);
  // Copy the data
  caffe_copy(prefetch_data_.count(), prefetch_data_.cpu_data(),
      top[0]->mutable_gpu_data());
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(prefetch_label_);
    // Copy the labels.
    caffe_copy(prefetch_label_.count(), prefetch_label_.cpu_data(),
        top[1]->mutable_gpu_data());
  }
  // Start a new prefetch thread
  CreatePrefetchThread();
}

template <typename Dtype>
void BasePrefetchingSparseDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    SparseBatch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
    if (SparseBlob<Dtype>* sparseBlob = dynamic_cast<SparseBlob<Dtype>*>(top[0]))
    {
      // Reshape to loaded data.
      sparseBlob->ReshapeLike(batch->data_);
      // Copy the data
      caffe_copy(batch->data_.nnz(), batch->data_.gpu_data(),
         sparseBlob->mutable_gpu_data());
      caffe_copy<int>(batch->data_.nnz(), batch->data_.gpu_indices(),
         sparseBlob->mutable_gpu_indices());
      caffe_copy<int>(batch->data_.shape()[0]+1, batch->data_.gpu_ptr(),
         sparseBlob->mutable_gpu_ptr());
    } else {
    LOG(ERROR) << "The top blob in the data layer sparse is not sparse";
    LOG(FATAL) << "fatal error";
    }
 
  DLOG(INFO) << "Prefetch sparse copied (forward)";
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(batch->label_);
    // Copy the labels.
    caffe_copy(batch->label_.count(), batch->label_.gpu_data(),
        top[1]->mutable_gpu_data());
  }

  // Ensure the copy is synchronous wrt the host, so that the next batch isn't
  // copied in meanwhile.
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  prefetch_free_.push(batch);
}

INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingDataLayer);
INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingSparseDataLayer);

}  // namespace caffe
