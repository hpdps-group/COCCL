#include "coccl_primitives_internal.h"

NCCL_API(ncclResult_t, ncclAlltoAllCompOverlap, const void* sendbuff,
  void* recvbuff, size_t sendcount, ncclDataType_t datatype, ncclComm_t comm,
  cudaStream_t stream);
ncclResult_t ncclAlltoAllCompOverlap(const void* sendbuff, void* recvbuff,
  size_t sendcount, ncclDataType_t datatype, ncclComm_t comm,
  cudaStream_t stream) {
  const cocclPipelineStage stages[] = {
    cocclPipelineCompress(),
    cocclPipelineAllToAll(comm),
    cocclPipelineDecompress(),
  };
  const cocclPipelineSpec spec = {
    "alltoall-overlap", sendbuff, recvbuff, sendcount,
    (size_t)comm->nRanks, datatype, comm, ncclCommOp_t::AlltoAll, stream,
    stages, (int)(sizeof(stages) / sizeof(stages[0]))
  };
  NCCLCHECK(cocclRunPipeline(&spec));
  return ncclSuccess;
}
