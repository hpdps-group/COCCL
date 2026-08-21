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
  const size_t chunkCount = info.count / (size_t)info.comm->nRanks;
  const cocclCompressionScope scope = info.comm->nNodes == 1
      ? cocclCompressionScope::Intra
      : cocclCompressionScope::Default;
  const cocclPipelineStage stages[] = {
      cocclPipelineCompress(prepared->compressors.get(scope)),
      cocclPipelineAllGather(info.comm),
      cocclPipelineDecompressReduce((size_t)info.comm->nRanks),
  };
  const cocclPipelineSpec spec = {
      "allreduce-oneshot", info.sendbuff, info.recvbuff, chunkCount,
      (size_t)info.comm->nRanks, info.datatype, info.comm, info.stream, stages,
      (int)(sizeof(stages) / sizeof(stages[0])),
      cocclPipelineInPlaceSameBuffer,
      cocclPipelineInputContiguous};
  return cocclRunPipelineSerial(&spec);
}

ncclResult_t runTwoShot(const cocclPreparedCall* prepared) {
  const cocclInfo& info = prepared->info;
  const size_t chunkCount = info.count / (size_t)info.comm->nRanks;
  const cocclCompressionScope scope = info.comm->nNodes == 1
      ? cocclCompressionScope::Intra
      : cocclCompressionScope::Default;
  const cocclPipelineStage stages[] = {
      cocclPipelineCompress(prepared->compressors.get(scope)),
      cocclPipelineAllToAll(info.comm),
      cocclPipelineDecompReduceComp(
          (size_t)info.comm->nRanks, prepared->compressors.get(scope)),
      cocclPipelineAllGather(info.comm),
      cocclPipelineDecompress(),
  };
  const cocclPipelineSpec spec = {
      "allreduce-twoshot", info.sendbuff, info.recvbuff, chunkCount,
      (size_t)info.comm->nRanks, info.datatype, info.comm, info.stream, stages,
      (int)(sizeof(stages) / sizeof(stages[0])),
      cocclPipelineInPlaceSameBuffer,
      cocclPipelineInputContiguous};
  return cocclRunPipeline(&spec);
}

ncclResult_t runTripleShot(const cocclPreparedCall* prepared) {
  const cocclInfo& info = prepared->info;
  ncclComm_t comm = info.comm;
  cocclHierarchicalComms hierarchy;
  NCCLCHECK(cocclCommGetHierarchicalComms(comm, &hierarchy));

  const size_t chunkCount = info.count / (size_t)comm->nRanks;
  void* const finalCompressor =
      prepared->compressors.get(cocclCompressionScope::Default);
  cocclPipelineStage stages[7];
  int stageCount = cocclBuildHierarchicalReduction(
      prepared, hierarchy.intraComm, hierarchy.interComm, finalCompressor,
      stages);
  stages[stageCount++] = cocclPipelineAllGather(comm);
  if (finalCompressor != nullptr) {
    stages[stageCount++] = cocclPipelineDecompress();
  }
  const cocclPipelineSpec spec = {
      "allreduce-tripleshot", info.sendbuff, info.recvbuff, chunkCount,
      (size_t)comm->nRanks, info.datatype, comm, info.stream, stages,
      stageCount,
      cocclPipelineInPlaceSameBuffer,
      cocclPipelineInputHierarchicalSwizzle};
  return cocclRunPipeline(&spec);
}

}  // namespace

ncclResult_t cocclExecuteAllReduce(const cocclPreparedCall* prepared) {
  const cocclInfo& info = prepared->info;
  if (info.count % (size_t)info.comm->nRanks != 0) {
    return ncclInvalidArgument;
  }
  if (prepared->algorithm == cocclAlgorithmAllReduceTripleShot &&
      (info.comm->localRanks <= 0 ||
       info.comm->nRanks % info.comm->localRanks != 0)) {
    return ncclInvalidArgument;
  }
  switch (prepared->algorithm) {
    case cocclAlgorithmAllReduceOneShot:
      return runOneShot(prepared);
    case cocclAlgorithmAllReduceTwoShot:
      return runTwoShot(prepared);
    case cocclAlgorithmAllReduceTripleShot:
      return runTripleShot(prepared);
    default:
      return ncclInternalError;
  }
}
