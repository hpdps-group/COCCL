
#include "reduce_extend.h"
#include "info.h"
#include "align.h"

extern "C"{

ncclResult_t launchReductionColl(const void* input1, const void* input2, void* output, ncclDataType_t type, size_t inputCount, 
    cudaStream_t stream);

ncclResult_t launchReduceChunk(const void* input, size_t chunkCount, void* output, ncclDataType_t type, int numChunks, 
    cudaStream_t stream);
}


ncclResult_t ncclReductionColl(const void* input1, const void* input2, void* output, ncclDataType_t type, ncclRedOp_t op, 
    size_t inputCount, cudaStream_t stream)
{
  NCCLCHECK(launchReductionColl(input1, input2, output, type, inputCount, stream));
  return ncclSuccess;
}

ncclResult_t ncclReduceChunk(const void* input, size_t chunkCount, void* output, ncclDataType_t type, int numChunks, 
    cudaStream_t stream)
{
    NCCLCHECK(launchReduceChunk(input, chunkCount, output, type, numChunks, stream));
    return ncclSuccess;
}