#include "coccl_primitives_internal.h"

NCCL_API(ncclResult_t, ncclAllGatherCompOverlap, const void* sendbuff,
  void* recvbuff, size_t sendcount, ncclDataType_t datatype, ncclComm_t comm,
  cudaStream_t stream);
ncclResult_t ncclAllGatherCompOverlap(const void* sendbuff, void* recvbuff,
  size_t sendcount, ncclDataType_t datatype, ncclComm_t comm,
  cudaStream_t stream) {
  const cocclPipelineStage stages[] = {
    cocclPipelineCompress(),
    cocclPipelineAllGather(comm),
    cocclPipelineDecompress(),
  };
  const cocclPipelineSpec spec = {
    "allgather-overlap", sendbuff, recvbuff, sendcount, 1, datatype, comm,
    ncclCommOp_t::AllGather, stream, stages,
    (int)(sizeof(stages) / sizeof(stages[0]))
  };
  NCCLCHECK(cocclRunPipeline(&spec));
  return ncclSuccess;
}
