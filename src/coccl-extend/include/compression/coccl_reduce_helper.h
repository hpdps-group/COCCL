#ifndef COCCL_REDUCE_HELPER_H_
#define COCCL_REDUCE_HELPER_H_

#include "nccl.h"

#include <cuda_runtime.h>

// Launches COCCL's built-in elementwise sum kernels. These are internal
// runtime helpers linked into NCCL, not part of the compressor plugin ABI.
ncclResult_t cocclLaunchReductionColl(const void* input1, const void* input2,
                                      void* output,
                                      ncclDataType_t datatype,
                                      size_t inputCount,
                                      cudaStream_t stream);

// Reduces numChunks contiguous arrays of chunkCount elements into one array.
// output may alias the first input chunk.
ncclResult_t cocclLaunchReduceChunk(const void* input, size_t chunkCount,
                                    void* output, ncclDataType_t datatype,
                                    int numChunks, cudaStream_t stream);

#endif  // COCCL_REDUCE_HELPER_H_
