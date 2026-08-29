#include "core/runtime/coccl_primitive_dispatch.h"

#include "checks.h"
#include "extensions/primitives/coccl_hierarchical_reduction.h"
#include "core/pipeline/coccl_pipeline.h"
#include "core/runtime/coccl_comm.h"
#include "core/runtime/coccl_prepared_call.h"
#include "comm.h"

namespace {

ncclResult_t runOneShot(const cocclPreparedCall* prepared) {
  const cocclInfo& info = prepared->info;
  const cocclCompressionScope scope = info.comm->nNodes == 1
      ? cocclCompressionScope::Intra
      : cocclCompressionScope::Default;
  const cocclPipelineStage stages[] = {
      cocclPipelineCompress(prepared->compressors.get(scope)),
      cocclPipelineAllToAll(info.comm),
      cocclPipelineDecompressReduce((size_t)info.comm->nRanks),
  };
  const cocclPipelineSpec spec = {
      "reducescatter-oneshot", info.sendbuff, info.recvbuff, info.count,
      (size_t)info.comm->nRanks, info.datatype, info.comm, info.stream, stages,
      (int)(sizeof(stages) / sizeof(stages[0])),
      cocclPipelineInPlaceOutputRankChunk,
      cocclPipelineInputContiguous, info.profilerTag};
  return cocclRunPipeline(&spec);
}

ncclResult_t runTwoShot(const cocclPreparedCall* prepared) {
  const cocclInfo& info = prepared->info;
  ncclComm_t comm = info.comm;
  cocclHierarchicalComms hierarchy;
  NCCLCHECK(cocclCommGetHierarchicalComms(comm, &hierarchy));

  cocclPipelineStage stages[5];
  const int stageCount = cocclBuildHierarchicalReduction(
      prepared, hierarchy.intraComm, hierarchy.interComm, nullptr, stages);
  const cocclPipelineSpec spec = {
      "reducescatter-twoshot", info.sendbuff, info.recvbuff, info.count,
      (size_t)comm->nRanks, info.datatype, comm, info.stream, stages,
      stageCount,
      cocclPipelineInPlaceOutputRankChunk,
      cocclPipelineInputHierarchicalSwizzle, info.profilerTag};
  return cocclRunPipeline(&spec);
}

}  // namespace

ncclResult_t cocclExecuteReduceScatter(const cocclPreparedCall* prepared) {
  if (prepared->algorithm == cocclAlgorithmReduceScatterOneShot) {
    return runOneShot(prepared);
  }
  ncclComm_t comm = prepared->info.comm;
  if (comm->localRanks <= 0 || comm->nRanks % comm->localRanks != 0) {
    return ncclInvalidArgument;
  }
  return runTwoShot(prepared);
}
