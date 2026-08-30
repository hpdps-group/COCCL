#include "core/runtime/coccl_primitive_dispatch.h"

#include "checks.h"
#include "core/compression/coccl_compressor_runtime.h"
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
  void* const compressor = prepared->compressors.get(scope);
  ncclComm_t gatherComm = info.comm;
  if (info.comm->nNodes > 1 &&
      !cocclCompressorSupports(
          compressor, cocclCompressorCapabilityFramed)) {
    NCCLCHECK(cocclCommGetZeroCtaComm(info.comm, &gatherComm));
  }
  const cocclPipelineStage stages[] = {
      cocclPipelineCompress(compressor),
      cocclPipelineAllGather(gatherComm),
      cocclPipelineDecompressReduce((size_t)info.comm->nRanks),
  };
  const cocclPipelineSpec spec = {
      "allreduce-oneshot", info.sendbuff, info.recvbuff, chunkCount,
      (size_t)info.comm->nRanks, info.datatype, info.comm, info.stream, stages,
      (int)(sizeof(stages) / sizeof(stages[0])),
      cocclPipelineInPlaceSameBuffer,
      cocclPipelineInputContiguous, info.profilerTag};
  return cocclRunPipelineSerial(&spec);
}

ncclResult_t runTwoShot(const cocclPreparedCall* prepared) {
  const cocclInfo& info = prepared->info;
  const size_t chunkCount = info.count / (size_t)info.comm->nRanks;
  const cocclCompressionScope scope = info.comm->nNodes == 1
      ? cocclCompressionScope::Intra
      : cocclCompressionScope::Default;
  void* const compressor = prepared->compressors.get(scope);
  ncclComm_t gatherComm = info.comm;
  if (info.comm->nNodes > 1 &&
      !cocclCompressorSupports(
          compressor, cocclCompressorCapabilityFramed)) {
    NCCLCHECK(cocclCommGetZeroCtaComm(info.comm, &gatherComm));
  }
  const cocclPipelineStage stages[] = {
      cocclPipelineCompress(compressor),
      cocclPipelineAllToAll(info.comm),
      cocclPipelineDecompReduceComp(
          (size_t)info.comm->nRanks, compressor),
      cocclPipelineAllGather(gatherComm),
      cocclPipelineDecompress(),
  };
  const cocclPipelineSpec spec = {
      "allreduce-twoshot", info.sendbuff, info.recvbuff, chunkCount,
      (size_t)info.comm->nRanks, info.datatype, info.comm, info.stream, stages,
      (int)(sizeof(stages) / sizeof(stages[0])),
      cocclPipelineInPlaceSameBuffer,
      cocclPipelineInputContiguous, info.profilerTag};
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
  ncclComm_t gatherComm = comm;
  if (!cocclCompressorSupports(
          finalCompressor, cocclCompressorCapabilityFramed)) {
    NCCLCHECK(cocclCommGetZeroCtaComm(comm, &gatherComm));
  }
  cocclPipelineStage stages[7];
  int stageCount = cocclBuildHierarchicalReduction(
      prepared, hierarchy.intraComm, hierarchy.interComm, finalCompressor,
      stages);
  stages[stageCount++] = cocclPipelineAllGather(gatherComm);
  if (finalCompressor != nullptr) {
    stages[stageCount++] = cocclPipelineDecompress();
  }
  const cocclPipelineSpec spec = {
      "allreduce-tripleshot", info.sendbuff, info.recvbuff, chunkCount,
      (size_t)comm->nRanks, info.datatype, comm, info.stream, stages,
      stageCount,
      cocclPipelineInPlaceSameBuffer,
      cocclPipelineInputHierarchicalSwizzle, info.profilerTag};
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
