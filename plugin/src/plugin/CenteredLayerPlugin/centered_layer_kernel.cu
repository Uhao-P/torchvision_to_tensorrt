#include <cuda_fp16.h>
#include <stdio.h>

#include <algorithm>
#include <cmath>

#include "openbayes_cuda_util/common_util.h"
#include "openbayes_cuda_util/cuda_util.h"
#include "centered_layer.h"

namespace openbayes {
namespace plugin {
using namespace openbayes::cuda;
template <typename T>
__global__ void centered_layer_kernel(T *output, const T *input, size_t input_size,
                                  int batch_size,
                                  int num_channels, int WH, T *mean) {
  CUDA_KERNEL_LOOP(i, input_size) {
    const int mean_var_index = i / (num_channels * batch_size);
    T ret = input[i] - mean[mean_var_index];
    output[i] = ret;
  }
}

template <typename T>
void centered_layer(T *output, const T *input, int batch_size,
                        int num_channels, int WH,
                        cudaStream_t stream, void *workspace) {
  size_t word_size = sizeof(T);
  T *mean = (T *)workspace;
  workspace = workspace + openbayes::common::getAlignedSize(
                              batch_size * num_channels * word_size);

  int mean_shape[1] = {batch_size * num_channels * WH};
  bool mean_reduce_dims[1] = {false};

  openbayes::cuda::tensorMean<T>(mean, input, &mean_shape[0],
                                   &mean_reduce_dims[0], 1, stream,
                                   workspace);

  size_t input_size = batch_size * num_channels * WH;

  centered_layer_kernel<T><<<GET_BLOCKS(input_size), CUDA_NUM_THREADS, 0, stream>>>(
      output, input, input_size, batch_size, num_channels, WH,
      mean);
}

template void centered_layer<float>(float *output, const float *input, int batch_size,
                                        int num_channels, int WH,
                                        cudaStream_t stream, void *workspace);


}  // namespace plugin
}  // namespace openbayes
