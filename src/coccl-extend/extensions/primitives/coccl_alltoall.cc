#include "core/runtime/coccl_primitive_dispatch.h"

#include "core/pipeline/coccl_pipeline.h"
#include "core/runtime/coccl_prepared_call.h"
#include "comm.h"

ncclResult_t cocclExecuteAllToAll(const cocclPreparedCall* prepared) {
  const cocclInfo& info = prepared->info;
  const cocclCompressionScope scope = info.comm->nNodes == 1
      ? cocclCompressionScope::Intra
      : cocclCompressionScope::Default;
  const cocclPipelineStage stages[] = {
      cocclPipelineCompress(prepared->compressors.get(scope)),
      cocclPipelineAllToAll(info.comm),
      cocclPipelineDecompress(),
  };
  const cocclPipelineSpec spec = {
      "alltoall", info.sendbuff, info.recvbuff, info.count,
      (size_t)info.comm->nRanks, info.datatype, info.comm, info.stream,
      stages, (int)(sizeof(stages) / sizeof(stages[0])),
      cocclPipelineInPlaceNone, cocclPipelineInputContiguous,
      info.profilerTag};
  return cocclRunPipeline(&spec);
}
