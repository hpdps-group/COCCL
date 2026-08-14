#include "core/compression/reduce_extend.h"

#include "core/compression/coccl_reduce_helper.h"

ncclResult_t ncclReductionColl(const void* input1, const void* input2,
                               void* output, ncclDataType_t type,
                               ncclRedOp_t op, size_t inputCount,
                               cudaStream_t stream) {
  // This internal helper currently implements the sum operation used by COCCL.
  (void)op;
  return cocclLaunchReductionColl(input1, input2, output, type, inputCount,
                                  stream);
}

ncclResult_t ncclReduceChunk(const void* input, size_t chunkCount, void* output,
                             ncclDataType_t type, int numChunks,
                             cudaStream_t stream) {
  return cocclLaunchReduceChunk(input, chunkCount, output, type, numChunks,
                                stream);
}
