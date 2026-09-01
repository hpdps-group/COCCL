#include "runtime/coccl_init.h"

#include "checks.h"
#include "core/compression/coccl_compressor_runtime.h"
#include "core/memory/coccl_buffer_management.h"
#include "core/pipeline/coccl_pipeline.h"
#include "core/runtime/coccl_comm.h"

namespace {

void keepFirstError(ncclResult_t candidate, ncclResult_t* result) {
  if (*result == ncclSuccess) *result = candidate;
}

}  // namespace

ncclResult_t cocclInit(ncclComm_t comm) {
  // NCCL 2.27 ignores unsupported images while probing its multi-arch kernels.
  // Clear that probe result before the autotune profiler launches CUDA work.
  const cudaError_t probeError = cudaGetLastError();
  if (probeError != cudaSuccess &&
      probeError != cudaErrorNoKernelImageForDevice) {
    CUDACHECK(probeError);
  }
  NCCLCHECK(cocclCompressorRuntimeInit(comm));
  if (!cocclCompressionEnabled()) return ncclSuccess;
  NCCLCHECK(cocclCommCreate(comm));
  NCCLCHECK(cocclBufferCommInit(comm));
  return ncclSuccess;
}

ncclResult_t cocclPrepareDestroy(ncclComm_t comm) {
  ncclResult_t result = ncclSuccess;
  keepFirstError(cocclPipelineCommDestroy(comm), &result);
  keepFirstError(cocclBufferCommDestroy(comm), &result);
  return result;
}

ncclResult_t cocclDestroy(ncclComm_t comm) {
  ncclResult_t result = ncclSuccess;
  keepFirstError(cocclPrepareDestroy(comm), &result);
  keepFirstError(cocclCommDestroy(comm), &result);
  keepFirstError(cocclCompressorRuntimeDestroy(comm), &result);
  return result;
}
