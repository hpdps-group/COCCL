#ifndef COCCL_COMPRESSOR_RUNTIME_H_
#define COCCL_COMPRESSOR_RUNTIME_H_

#include "compressor_plugin/detail/compressor_abi.h"

#include <memory>

struct ncclComm;
using ncclComm_t = ncclComm*;

// The concrete runtime state remains private to coccl_compressor.cc. Copying a
// handle keeps immutable config, lazy plugin state, and persistent buffers
// alive.
struct cocclCompressorRuntimeState;

struct cocclCompressorHandle {
  std::shared_ptr<cocclCompressorRuntimeState> state;

  explicit operator bool() const { return state != nullptr; }
};

// Takes ownership of non-null parsedConfig on success. On failure the caller
// remains responsible for destroyConfig(parsedConfig).
ncclResult_t cocclCreateCompressorHandle(
    ncclComm_t comm, const cocclCompressorPlugin* compressor,
    void* parsedConfig,
    cocclCompressorHandle* handle);

const cocclCompressorPlugin* cocclCompressorDescriptor(
    const cocclCompressorHandle& handle);

bool cocclCompressorSupports(const cocclCompressorHandle& handle,
                             cocclCompressorCapability capability);

// All codec operations share this lifecycle path. Operation-specific NCCL
// wrappers only populate the fields relevant to their semantic operation.
ncclResult_t cocclExecuteCompressor(
    const cocclCompressorHandle& handle,
    cocclCompressorOperation operation,
    const cocclCompressorDataView& input, cocclCompressorOutputView* output,
    int rank, size_t reduceChunks, ncclDataType_t originalDatatype,
    size_t originalElements, cudaStream_t stream);

#endif
