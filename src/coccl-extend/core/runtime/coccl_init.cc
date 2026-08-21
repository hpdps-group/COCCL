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
  NCCLCHECK(cocclCompressorRuntimeInit(comm));
  if (!cocclCompressionEnabled()) return ncclSuccess;
  NCCLCHECK(cocclCommCreate(comm));
  return cocclBufferCommInit(comm);
}

ncclResult_t cocclDestroy(ncclComm_t comm) {
  ncclResult_t result = ncclSuccess;
  keepFirstError(cocclCommDestroy(comm), &result);
  keepFirstError(cocclPipelineCommDestroy(comm), &result);
  keepFirstError(cocclBufferCommDestroy(comm), &result);
  keepFirstError(cocclCompressorRuntimeDestroy(comm), &result);
  return result;
}
