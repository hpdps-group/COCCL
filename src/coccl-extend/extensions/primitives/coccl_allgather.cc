#include "core/runtime/coccl_primitive_dispatch.h"

#include "core/pipeline/coccl_pipeline.h"
#include "core/runtime/coccl_comm.h"
#include "core/runtime/coccl_prepared_call.h"
#include "comm.h"

namespace {

bool uniformHierarchy(ncclComm_t comm) {
  if (comm->nNodes <= 1 || comm->localRanks <= 1 ||
      comm->nRanks != comm->nNodes * comm->localRanks) {
    return false;
  }
  for (int node = 0; node < comm->nNodes; ++node) {
    if (comm->nodeRanks[node].localRanks != comm->localRanks) return false;
  }
  return true;
}

ncclResult_t executeOneShot(const cocclPreparedCall* prepared) {
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
      cocclPipelineInputContiguous};
  return cocclRunPipeline(&spec);
}

ncclResult_t executeTwoShot(const cocclPreparedCall* prepared) {
  const cocclInfo& info = prepared->info;
  if (!uniformHierarchy(info.comm)) return ncclInvalidUsage;

  cocclHierarchicalComms hierarchy;
  NCCLCHECK(cocclCommGetHierarchicalComms(info.comm, &hierarchy));
  void* const compressor = prepared->compressors.get(
      cocclCompressionScope::Inter);
  const cocclPipelineStage stages[] = {
      cocclPipelineCompress(compressor),
      cocclPipelineAllGather(hierarchy.interComm),
      cocclPipelineAllGather(hierarchy.intraComm),
      cocclPipelineDecompress(),
  };
  const cocclPipelineSpec spec = {
      "allgather-twoshot", info.sendbuff, info.recvbuff, info.count, 1,
      info.datatype, info.comm, info.stream, stages,
      (int)(sizeof(stages) / sizeof(stages[0])),
      cocclPipelineInPlaceInputRankChunk,
      cocclPipelineInputContiguous,
      cocclPipelineOutputHierarchicalAllGather};
  return cocclRunPipeline(&spec);
}

}  // namespace

ncclResult_t cocclExecuteAllGather(const cocclPreparedCall* prepared) {
  switch (prepared->algorithm) {
    case cocclAlgorithmAllGatherOneShot:
      return executeOneShot(prepared);
    case cocclAlgorithmAllGatherTwoShot:
      return executeTwoShot(prepared);
    default:
      return ncclInvalidArgument;
  }
}
