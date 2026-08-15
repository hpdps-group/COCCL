#include "coccl_alltoall.h"

#include "coccl_pipeline.h"
#include "coccl_prepared_call.h"
#include "comm.h"

ncclResult_t cocclExecuteAllToAll(const cocclPreparedCall* prepared) {
  const cocclInfo& info = prepared->info;
  const cocclPipelineStage stages[] = {
      cocclPipelineCompress(),
      cocclPipelineAllToAll(info.comm),
      cocclPipelineDecompress(),
  };
  const cocclPipelineSpec spec = {
      "alltoall", info.sendbuff, info.recvbuff, info.count,
      (size_t)info.comm->nRanks, info.datatype, info.comm, info.stream,
      stages, (int)(sizeof(stages) / sizeof(stages[0]))};
  return cocclRunPipeline(&spec);
}
