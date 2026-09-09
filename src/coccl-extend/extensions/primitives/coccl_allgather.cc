#include "core/runtime/coccl_primitive_dispatch.h"

#include "checks.h"
#include "core/compression/coccl_compressor_runtime.h"
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
  void* const compressor = prepared->compressors.get(scope);
  ncclComm_t communicationComm = info.comm;
  if (info.comm->nNodes > 1 &&
      !cocclCompressorSupports(
          compressor, cocclCompressorCapabilityFramed)) {
    NCCLCHECK(cocclCommGetZeroCtaComm(
        info.comm, &communicationComm));
  }
  cocclPipelineStage stages[4];
  const cocclPipelineSpec spec = cocclBuildAllGatherSpec(
      info, cocclAlgorithmAllGatherOneShot, compressor,
      communicationComm, nullptr, stages);
  return cocclRunPipeline(&spec);
}

ncclResult_t executeTwoShot(const cocclPreparedCall* prepared) {
  const cocclInfo& info = prepared->info;
  if (!uniformHierarchy(info.comm)) return ncclInvalidUsage;

  cocclHierarchicalComms hierarchy;
  NCCLCHECK(cocclCommGetHierarchicalComms(info.comm, &hierarchy));
  void* const compressor = prepared->compressors.get(
      cocclCompressionScope::Inter);
  cocclPipelineStage stages[4];
  const cocclPipelineSpec spec = cocclBuildAllGatherSpec(
      info, cocclAlgorithmAllGatherTwoShot, compressor,
      hierarchy.interComm, hierarchy.intraComm, stages);
  return cocclRunPipeline(&spec);
}

}  // namespace

cocclPipelineSpec cocclBuildAllGatherSpec(
    const cocclInfo& info, cocclAlgorithmKind algorithm, void* compressor,
    ncclComm_t gatherComm, ncclComm_t intraComm,
    cocclPipelineStage* stages) {
  const bool twoShot = algorithm == cocclAlgorithmAllGatherTwoShot;
  const bool encoded = twoShot || compressor != nullptr;
  int stageCount = 0;
  if (encoded) stages[stageCount++] = cocclPipelineCompress(compressor);
  stages[stageCount++] = cocclPipelineAllGather(gatherComm);
  if (twoShot) stages[stageCount++] = cocclPipelineAllGather(intraComm);
  if (encoded) stages[stageCount++] = cocclPipelineDecompress();
  return {
      twoShot ? "allgather-twoshot" : encoded ? "allgather" : "allgather-native",
      info.sendbuff, info.recvbuff, info.count, 1,
      info.datatype, info.comm, info.stream, stages, stageCount,
      cocclPipelineInPlaceInputRankChunk,
      cocclPipelineInputContiguous, info.profilerTag,
      twoShot ? cocclPipelineOutputHierarchicalAllGather
              : cocclPipelineOutputContiguous};
}

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
