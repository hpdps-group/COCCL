
#ifndef NCCL_REDUCE_EXTEND_KERNEL_H_
#define NCCL_REDUCE_EXTEND_KERNEL_H_

#include "nccl.h"
#include "device.h"
#include "checks.h"
#include <cuda_runtime.h>

void minMaxReduction(const void* input, const size_t chunkCount, void* output, const size_t outputChunkCount, 
    const size_t numChunks, ncclDataType_t datatype, cudaStream_t stream);

#endif