#ifndef NCCL_REDUCE_EXTEND_H_
#define NCCL_REDUCE_EXTEND_H_

#include "device.h"
#include "core.h"

// Thin host wrappers around CUDA reduction kernels used when compressed data
// must be decompressed and reduced outside NCCL's built-in collectives.
ncclResult_t ncclReductionColl(const void* input1, const void* input2, void* output, ncclDataType_t type, ncclRedOp_t op, 
    size_t inputCount, cudaStream_t stream);

ncclResult_t ncclReduceChunk(const void* input, size_t chunkCount, void* output, ncclDataType_t type, int numChunks, 
    cudaStream_t stream);


#endif
