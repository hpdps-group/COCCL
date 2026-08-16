#include "coccl_allreduce.h"

#include "checks.h"
#include "coccl_pipeline.h"
#include "coccl_prepared_call.h"
#include "comm.h"

extern __thread ncclComm_t InterSubComm;
extern __thread ncclComm_t IntraSubComm;

namespace {

ncclResult_t runOneShot(const cocclPreparedCall* prepared) {
  const cocclInfo& info = prepared->info;
  const size_t chunkCount = info.count / (size_t)info.comm->nRanks;
  const cocclPipelineStage stages[] = {
      cocclPipelineCompress(),
      cocclPipelineAllGather(info.comm),
      cocclPipelineDecompressReduce((size_t)info.comm->nRanks),
  };
  const cocclPipelineSpec spec = {
      "allreduce-oneshot", info.sendbuff, info.recvbuff, chunkCount,
      (size_t)info.comm->nRanks, info.datatype, prepared->policy,
      info.comm, info.stream, stages,
      (int)(sizeof(stages) / sizeof(stages[0]))};
  return cocclRunPipeline(&spec);
}

ncclResult_t runTwoShot(const cocclPreparedCall* prepared) {
  const cocclInfo& info = prepared->info;
  const size_t chunkCount = info.count / (size_t)info.comm->nRanks;
  const cocclPipelineStage stages[] = {
      cocclPipelineCompress(),
      cocclPipelineAllToAll(info.comm),
      cocclPipelineDecompReduceComp((size_t)info.comm->nRanks),
      cocclPipelineAllGather(info.comm),
      cocclPipelineDecompress(),
  };
  const cocclPipelineSpec spec = {
      "allreduce-twoshot", info.sendbuff, info.recvbuff, chunkCount,
      (size_t)info.comm->nRanks, info.datatype, prepared->policy,
      info.comm, info.stream, stages,
      (int)(sizeof(stages) / sizeof(stages[0]))};
  return cocclRunPipeline(&spec);
}

ncclResult_t runTripleShot(const cocclPreparedCall* prepared) {
  const cocclInfo& info = prepared->info;
  ncclComm_t comm = info.comm;
  const int localRanks = comm->localRanks;
  const int nNodes = comm->nRanks / localRanks;
  if (IntraSubComm == nullptr) {
    NCCLCHECK(ncclCommSplit(
        comm, comm->rank / localRanks, comm->rank, &IntraSubComm, nullptr));
    NCCLCHECK(ncclCommSplit(
        comm, comm->rank % localRanks, comm->rank, &InterSubComm, nullptr));
  }

  const size_t chunkCount = info.count / (size_t)comm->nRanks;
  const cocclPipelineStage stages[] = {
      cocclPipelineCompress(),
      cocclPipelineAllToAll(IntraSubComm),
      cocclPipelineDecompReduceComp((size_t)localRanks),
      cocclPipelineAllToAll(InterSubComm),
      cocclPipelineDecompReduceComp((size_t)nNodes),
      cocclPipelineAllGather(comm),
      cocclPipelineDecompress(),
  };
  const cocclPipelineSpec spec = {
      "allreduce-tripleshot", info.sendbuff, info.recvbuff, chunkCount,
      (size_t)comm->nRanks, info.datatype, prepared->policy,
      comm, info.stream, stages,
      (int)(sizeof(stages) / sizeof(stages[0]))};
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
