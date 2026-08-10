#include "primitives/coccl_primitives_internal.h"

NCCL_API(ncclResult_t, ncclAllToAllComp, const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
ncclResult_t  ncclAllToAllComp(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  
  // Compress all per-peer chunks, exchange them with AllToAll, then decompress
  // the received compressed chunks back into recvbuff.
  size_t compSendCount;
  ncclDataType_t compDatatype;
  CUDACHECK(cudaSetDevice(comm->cudaDev));

  size_t totalSendBytes = 2 * comm->nRanks * sendcount * ncclTypeSize(datatype);
  cocclBufferHandle workspaceBuffer = {};
  NCCLCHECK(cocclGetBuffer(comm, totalSendBytes, &workspaceBuffer));
  void* workspaceBase = workspaceBuffer.ptr;

  // reuse buff may have some wrong, some data may not send/recv sometimes
  // NCCLCHECK(ncclCompress(sendbuff, &sendCompbuff, sendcount, datatype, &compSendCount, &compDatatype, comm->nRanks, comm, cocclDefaultPolicy(cocclOperation::AllToAll), stream));
  NCCLCHECK(ncclCompress(sendbuff, &workspaceBase, sendcount, datatype, &compSendCount, &compDatatype, comm->nRanks, comm->rank, comm, cocclDefaultPolicy(cocclOperation::AllToAll), stream));

  void* sendCompbuff = workspaceBase;
  void* recvCompbuff = (char*)workspaceBase + compSendCount * comm->nRanks * ncclTypeSize(compDatatype);
  
  NCCLCHECK(ncclAllToAllNaive((void*)sendCompbuff, (void*)recvCompbuff, compSendCount, compDatatype, comm, stream));

  NCCLCHECK(ncclDecompress(recvbuff, (char*)recvCompbuff, sendcount, datatype, compSendCount, compDatatype, comm->nRanks, comm, cocclDefaultPolicy(cocclOperation::AllToAll), stream));
  NCCLCHECK(cocclReleaseBuffer(&workspaceBuffer, stream));
  
  return ncclSuccess;
}
