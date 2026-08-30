#include "core/runtime/coccl_primitive_dispatch.h"

#include "checks.h"
#include "core/compression/coccl_compressor_runtime.h"
#include "core/pipeline/coccl_pipeline.h"
#include "core/runtime/coccl_comm.h"
#include "core/runtime/coccl_prepared_call.h"
#include "comm.h"

ncclResult_t cocclExecuteAllGather(const cocclPreparedCall* prepared) {
  const cocclInfo& info = prepared->info;
  const cocclCompressionScope scope = info.comm->nNodes == 1
      ? cocclCompressionScope::Intra
      : cocclCompressionScope::Default;
  void* const compressor = prepared->compressors.get(scope);
  ncclComm_t communicationComm = info.comm;
  if (info.comm->nNodes > 1 &&
      !cocclCompressorSupports(
          compressor, cocclCompressorCapabilityFramed)) {
    NCCLCHECK(cocclCommGetZeroCtaComm(
        info.comm, &communicationComm));
  }
  const cocclPipelineStage stages[] = {
      cocclPipelineCompress(compressor),
      cocclPipelineAllGather(communicationComm),
      cocclPipelineDecompress(),
  };
  const cocclPipelineSpec spec = {
      "allgather", info.sendbuff, info.recvbuff, info.count, 1,
      info.datatype, info.comm, info.stream, stages,
      (int)(sizeof(stages) / sizeof(stages[0])),
      cocclPipelineInPlaceInputRankChunk,
      cocclPipelineInputContiguous, info.profilerTag};
  return cocclRunPipeline(&spec);
}
