#include "coccl_reducescatter.h"

#include "checks.h"
#include "coccl_pipeline.h"
#include "coccl_prepared_call.h"
#include "comm.h"

extern __thread ncclComm_t InterSubComm;
extern __thread ncclComm_t IntraSubComm;

namespace {

ncclResult_t runOneShot(const cocclPreparedCall* prepared) {
  const cocclInfo& info = prepared->info;
  const cocclPipelineStage stages[] = {
      cocclPipelineCompress(),
      cocclPipelineAllToAll(info.comm),
      cocclPipelineDecompressReduce((size_t)info.comm->nRanks),
  };
  const cocclPipelineSpec spec = {
      "reducescatter-oneshot", info.sendbuff, info.recvbuff, info.count,
      (size_t)info.comm->nRanks, info.datatype, prepared->policy,
      info.comm, info.stream, stages,
      (int)(sizeof(stages) / sizeof(stages[0])),
      cocclPipelineInPlaceOutputRankChunk,
      cocclPipelineInputContiguous};
  return cocclRunPipeline(&spec);
}

ncclResult_t runTwoShot(const cocclPreparedCall* prepared) {
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

  const cocclPipelineStage stages[] = {
      cocclPipelineCompress(),
      cocclPipelineAllToAll(IntraSubComm),
      cocclPipelineDecompReduceComp((size_t)localRanks),
      cocclPipelineAllToAll(InterSubComm),
      cocclPipelineDecompressReduce((size_t)nNodes),
  };
  const cocclPipelineSpec spec = {
      "reducescatter-twoshot", info.sendbuff, info.recvbuff, info.count,
      (size_t)comm->nRanks, info.datatype, prepared->policy,
      comm, info.stream, stages,
      (int)(sizeof(stages) / sizeof(stages[0])),
      cocclPipelineInPlaceOutputRankChunk,
      cocclPipelineInputHierarchicalSwizzle};
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
