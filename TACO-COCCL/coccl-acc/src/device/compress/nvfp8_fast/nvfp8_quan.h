#ifndef NVFP8_QUAN_H
#define NVFP8_QUAN_H


#include "nccl.h"
#include "device.h"
#include "checks.h"
#include "compressor.h"
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// // Hadamard变换函数声明 - 使用不同的函数名避免冲突
// cudaError_t nvfp8_hadamard_transform_f32(float* data, size_t n, int group_size, cudaStream_t stream);
// cudaError_t nvfp8_hadamard_transform_f16(__half* data, size_t n, int group_size, cudaStream_t stream);
// cudaError_t nvfp8_hadamard_transform_bf16(__nv_bfloat16* data, size_t n, int group_size, cudaStream_t stream);

// cudaError_t nvfp8_inverse_hadamard_transform_f32(float* data, size_t n, int group_size, cudaStream_t stream);
// cudaError_t nvfp8_inverse_hadamard_transform_f16(__half* data, size_t n, int group_size, cudaStream_t stream);
// cudaError_t nvfp8_inverse_hadamard_transform_bf16(__nv_bfloat16* data, size_t n, int group_size, cudaStream_t stream);

#ifdef __cplusplus
extern "C" {
#endif

// void loadConfigPair(const char* configFile, std::pair<const char*, const char*>** configPairs, int* configPairCount);

#ifdef __cplusplus
}
#endif

#endif // NVFP8_QUAN_H