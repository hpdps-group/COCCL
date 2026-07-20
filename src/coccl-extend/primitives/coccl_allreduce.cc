#include "coccl_primitives_internal.h"

NCCL_API(ncclResult_t, ncclAllReduceCompOneShot, const void* sendbuff,
  void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op,
  ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduceCompOneShot(const void* sendbuff, void* recvbuff,
  size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm,
  cudaStream_t stream) {
  if (comm == nullptr || comm->nRanks <= 0) return ncclInvalidArgument;

  const cocclPipelineStage stages[] = {
    cocclPipelineCompress(),
    cocclPipelineAllGather(comm),
    cocclPipelineDecompressReduce((size_t)comm->nRanks),
  };
  const cocclPipelineSpec spec = {
    "allreduce-oneshot", sendbuff, recvbuff,
    DIVUP(count, (size_t)comm->nRanks), (size_t)comm->nRanks, datatype, comm,
    ncclCommOp_t::AllReduce, stream, stages,
    (int)(sizeof(stages) / sizeof(stages[0]))
  };
  NCCLCHECK(cocclRunPipelineSerial(&spec));
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclAllReduceCompTwoShotOverlap,
  const void* sendbuff, void* recvbuff, size_t count,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm,
  cudaStream_t stream);
ncclResult_t ncclAllReduceCompTwoShotOverlap(const void* sendbuff,
  void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op,
  ncclComm* comm, cudaStream_t stream) {
  if (comm->nRanks <= 0 || count % (size_t)comm->nRanks != 0) {
    return ncclInvalidArgument;
  }
  const size_t recvcount = count / comm->nRanks;
  const cocclPipelineStage stages[] = {
    cocclPipelineCompress(),
    cocclPipelineAllToAll(comm),
    cocclPipelineDecompReduceComp((size_t)comm->nRanks),
    cocclPipelineAllGather(comm),
    cocclPipelineDecompress(),
  };
  const cocclPipelineSpec spec = {
    "allreduce-twoshot-overlap", sendbuff, recvbuff, recvcount,
    (size_t)comm->nRanks, datatype, comm, ncclCommOp_t::AllReduce, stream,
    stages, (int)(sizeof(stages) / sizeof(stages[0]))
  };
  NCCLCHECK(cocclRunPipeline(&spec));
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclAllReduceCompTripleShotTLOverlap,
  const void* sendbuff, void* recvbuff, size_t count,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm,
  cudaStream_t stream);
ncclResult_t ncclAllReduceCompTripleShotTLOverlap(const void* sendbuff,
  void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op,
  ncclComm* comm, cudaStream_t stream) {
  if (comm->nRanks <= 0 || count % (size_t)comm->nRanks != 0) {
    return ncclInvalidArgument;
  }
  const size_t recvcount = count / comm->nRanks;
  const int localRanks = comm->localRanks;
  if (localRanks <= 0 || comm->nRanks % localRanks != 0) {
    return ncclInvalidArgument;
  }
  const int nNodes = comm->nRanks / localRanks;
  cocclHierarchicalComms hierarchy = {};
  NCCLCHECK(cocclCommGetHierarchicalComms(comm, &hierarchy));

  const cocclPipelineStage stages[] = {
    cocclPipelineCompress(),
    cocclPipelineAllToAll(hierarchy.intraComm),
    cocclPipelineDecompReduceComp((size_t)localRanks),
    cocclPipelineAllToAll(hierarchy.interComm),
    cocclPipelineDecompReduceComp((size_t)nNodes),
    cocclPipelineAllGather(comm),
    cocclPipelineDecompress(),
  };
  const cocclPipelineSpec spec = {
    "allreduce-tripleshot-tl-overlap", sendbuff, recvbuff, recvcount,
    (size_t)comm->nRanks, datatype, comm, ncclCommOp_t::AllReduce_Inter,
    stream, stages, (int)(sizeof(stages) / sizeof(stages[0]))
  };
  NCCLCHECK(cocclRunPipeline(&spec));
  return ncclSuccess;
}
