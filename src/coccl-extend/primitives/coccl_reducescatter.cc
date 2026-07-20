#include "coccl_primitives_internal.h"

NCCL_API(ncclResult_t, ncclReduceScatterCompOneShotOverlap,
  const void* sendbuff, void* recvbuff, size_t recvcount,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm,
  cudaStream_t stream);
ncclResult_t ncclReduceScatterCompOneShotOverlap(const void* sendbuff,
  void* recvbuff, size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op,
  ncclComm* comm, cudaStream_t stream) {
  const cocclPipelineStage stages[] = {
    cocclPipelineCompress(),
    cocclPipelineAllToAll(comm),
    cocclPipelineDecompressReduce((size_t)comm->nRanks),
  };
  const cocclPipelineSpec spec = {
    "reducescatter-oneshot-overlap", sendbuff, recvbuff, recvcount,
    (size_t)comm->nRanks, datatype, comm, ncclCommOp_t::ReduceScatter,
    stream, stages, (int)(sizeof(stages) / sizeof(stages[0]))
  };
  NCCLCHECK(cocclRunPipeline(&spec));
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclReduceScatterCompTwoShotTLOverlap,
  const void* sendbuff, void* recvbuff, size_t recvcount,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm,
  cudaStream_t stream);
ncclResult_t ncclReduceScatterCompTwoShotTLOverlap(const void* sendbuff,
  void* recvbuff, size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op,
  ncclComm* comm, cudaStream_t stream) {
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
    cocclPipelineDecompressReduce((size_t)nNodes),
  };
  const cocclPipelineSpec spec = {
    "reducescatter-twoshot-tl-overlap", sendbuff, recvbuff, recvcount,
    (size_t)comm->nRanks, datatype, comm,
    ncclCommOp_t::ReduceScatter_Inter, stream, stages,
    (int)(sizeof(stages) / sizeof(stages[0]))
  };
  NCCLCHECK(cocclRunPipeline(&spec));
  return ncclSuccess;
}
