#ifndef COCCL_COMPRESSOR_RUNTIME_H_
#define COCCL_COMPRESSOR_RUNTIME_H_

#include "compressor_plugin/detail/coccl_compressor_abi.h"
#include "core/training/coccl_training_assist.h"
#include "runtime/coccl_operation.h"
#include "nccl.h"

struct cocclResolvedCompressorPolicy {
  void* compressor = nullptr;
  size_t thresholdBytes = 0;
};

// History belongs to the communicator and decoded layout.
// The slice identity is supplied by the planner, not inferred by the plugin.
struct cocclCompressorScope {
  ncclComm_t comm = nullptr;
  int layout = 0;
  size_t slice = 0;
  size_t slices = 0;
  int localChunkIndex = 0;
};

bool cocclCompressionEnabled();
ncclResult_t cocclResolveCompressorPolicy(
    cocclTrainingRole role, cocclPolicyKey key,
    cocclResolvedCompressorPolicy* resolved);

ncclResult_t cocclQueryCompressorEncodedSizeBound(
    const cocclCompressorPlugin* plugin, const void* config,
    cocclCompressorOperation operation, size_t elements, size_t chunks,
    ncclDataType_t datatype, size_t* encodedBytes);
ncclResult_t cocclGetCompressorEncodedSizeBound(
    void* compressor, cocclCompressorOperation operation,
    size_t elements, size_t chunks, ncclDataType_t datatype,
    size_t* encodedBytes);

bool cocclCompressorSupports(
    void* compressor, cocclCompressorCapability capability);
const cocclCompressorPlugin* cocclCompressorDescriptor(void* compressor);

ncclResult_t cocclExecuteCompressor(
    void* compressor, void* inputCompressor,
    cocclCompressorOperation operation,
    const cocclCompressorView& input, cocclCompressorView* output, int rank,
    size_t reduceChunks, ncclDataType_t originalDatatype,
    size_t originalElements, cudaStream_t stream,
    const cocclCompressorScope* scope = nullptr);

ncclResult_t cocclCompressorRuntimeInit(ncclComm_t comm);
ncclResult_t cocclCompressorRuntimeDestroy(ncclComm_t comm);

#endif
