#pragma once
#include <cuda_runtime.h>

namespace openbayes {
namespace plugin {
template <typename T>
void centered_layer(T* output, const T* input, int batch_size,
                        int num_channels, int WH, 
                        cudaStream_t stream, void* workspace);

}  // namespace plugin
}  // namespace openbayes
