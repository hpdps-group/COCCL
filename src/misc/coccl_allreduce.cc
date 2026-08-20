#include "coccl_allreduce.h"

#include "checks.h"
#include "coccl_hierarchical_reduction.h"
#include "coccl_pipeline.h"
#include "coccl_prepared_call.h"
#include "coccl_training_assist.h"
#include "comm.h"

extern __thread ncclComm_t InterSubComm;
extern __thread ncclComm_t IntraSubComm;

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
  const int localRanks = comm->localRanks;
  if (IntraSubComm == nullptr) {
    NCCLCHECK(ncclCommSplit(
        comm, comm->rank / localRanks, comm->rank, &IntraSubComm, nullptr));
    cocclTrainingAssistUnregister(IntraSubComm);
    NCCLCHECK(ncclCommSplit(
        comm, comm->rank % localRanks, comm->rank, &InterSubComm, nullptr));
    cocclTrainingAssistUnregister(InterSubComm);
  }

  const size_t chunkCount = info.count / (size_t)comm->nRanks;
  void* const finalCompressor =
      prepared->compressors.get(cocclCompressionScope::Default);
  cocclPipelineStage stages[7];
  int stageCount = cocclBuildHierarchicalReduction(
      prepared, IntraSubComm, InterSubComm, finalCompressor, stages);
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
