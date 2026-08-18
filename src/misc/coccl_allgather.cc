#include "coccl_allgather.h"

#include "coccl_pipeline.h"
#include "comm.h"

ncclResult_t cocclExecuteAllGather(const cocclPreparedCall* prepared) {
  const cocclInfo& info = prepared->info;
  const cocclPipelineStage stages[] = {
      cocclPipelineCompress(),
      cocclPipelineAllGather(info.comm),
      cocclPipelineDecompress(),
  };
  const cocclPipelineSpec spec = {
      "allgather", info.sendbuff, info.recvbuff, info.count, 1,
      info.datatype, prepared->policy,
      info.comm, info.stream, stages,
      (int)(sizeof(stages) / sizeof(stages[0])),
      cocclPipelineInPlaceInputRankChunk};
  return cocclRunPipeline(&spec);
}
