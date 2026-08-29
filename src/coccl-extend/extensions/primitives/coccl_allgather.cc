#include "core/runtime/coccl_primitive_dispatch.h"

#include "core/pipeline/coccl_pipeline.h"
#include "core/runtime/coccl_prepared_call.h"
#include "comm.h"

ncclResult_t cocclExecuteAllGather(const cocclPreparedCall* prepared) {
  const cocclInfo& info = prepared->info;
  const cocclCompressionScope scope = info.comm->nNodes == 1
      ? cocclCompressionScope::Intra
      : cocclCompressionScope::Default;
  const cocclPipelineStage stages[] = {
      cocclPipelineCompress(prepared->compressors.get(scope)),
      cocclPipelineAllGather(info.comm),
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
