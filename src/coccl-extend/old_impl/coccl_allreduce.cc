#include "legacy/coccl_old_impl_internal.h"

NCCL_API(ncclResult_t, ncclAllReduceCompTwoShot, const void* sendbuff, void* recvbuff, size_t count,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduceCompTwoShot(const void* sendbuff, void* recvbuff, size_t count,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {

  size_t chunkCount = DIVUP(count, comm->nRanks);
 
  // void* sendCompbuff = nullptr;
  // void* recvCompbuff = nullptr;
  
  CUDACHECK(cudaSetDevice(comm->cudaDev));
  size_t compSendCount;
  ncclDataType_t compDatatype;
  size_t totalSendBytes = 2 * count * ncclTypeSize(datatype);
  cocclBufferHandle workspaceBuffer = {};
  NCCLCHECK(cocclGetBuffer(comm, totalSendBytes, &workspaceBuffer));
  void* workspaceBase = workspaceBuffer.ptr;
 
  // reuse buff may have some wrong, some data may not send/recv sometimes
  // NCCLCHECK(ncclCompress(sendbuff, &sendCompbuff, chunkCount, datatype, &compSendCount, &compDatatype, comm->nRanks, comm, cocclDefaultPolicy(cocclOperation::AllReduce), stream));
  NCCLCHECK(ncclCompress(sendbuff, &workspaceBase, chunkCount, datatype,
                            &compSendCount, &compDatatype, comm->nRanks, comm->rank, comm, cocclDefaultPolicy(cocclOperation::AllReduce), stream));

  void* sendCompbuff = workspaceBase;
  void* recvCompbuff = (char*) workspaceBase + compSendCount * comm->nRanks * ncclTypeSize(compDatatype);
  // CUDACHECK(cudaMallocAsync((void**)&recvCompbuff, compSendCount * comm->nRanks * ncclTypeSize(compDatatype), stream));
  //sendCompbuff + comm->nRanks * compSendCount * ncclTypeSize(ncclInt8)
  
  // Two-shot: exchange compressed chunks, reduce/recompress one chunk per rank,
  // then AllGather the compressed reduced chunks.
  NCCLCHECK(ncclAllToAllNaive((void*)sendCompbuff, (void*)recvCompbuff, compSendCount, compDatatype, comm, stream));
  size_t reCompSendCount;
  ncclDataType_t reCompDatatype;
  // DecompReduceComp
  NCCLCHECK(ncclDecompReduceComp((void*)recvCompbuff, &sendCompbuff, count / comm->nRanks, datatype,
              compSendCount, compDatatype, &reCompSendCount, &reCompDatatype, comm->nRanks, comm, cocclDefaultPolicy(cocclOperation::AllReduce), stream));

  struct ncclInfo info = { ncclFuncAllGather, "AllGather",
    sendCompbuff, recvCompbuff, reCompSendCount, reCompDatatype, ncclSum, 0, comm, stream, /* Args */
    ALLGATHER_CHUNKSTEPS, ALLGATHER_SLICESTEPS };
  NCCLCHECK(ncclEnqueueCheck(&info));

  // Decompress
  NCCLCHECK(ncclDecompress(recvbuff, (void*)recvCompbuff, chunkCount, datatype, reCompSendCount, reCompDatatype, comm->nRanks, comm, cocclDefaultPolicy(cocclOperation::AllReduce), stream));
  NCCLCHECK(cocclReleaseBuffer(&workspaceBuffer, stream));

  return ncclSuccess;
}
NCCL_API(ncclResult_t, ncclAllReduceCompTripleShot, const void* sendbuff, void* recvbuff, size_t count,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduceCompTripleShot(const void* sendbuff, void* recvbuff, size_t count,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream){
  size_t recvcount = DIVUP(count, comm->nRanks);

  int nRanks = comm->nRanks;
  int localRanks = comm->localRanks;
  int nNodes = nRanks / localRanks;
  cocclHierarchicalComms hierarchy = {};
  NCCLCHECK(cocclCommGetHierarchicalComms(comm, &hierarchy));
  ncclComm_t IntraSubComm = hierarchy.intraComm;
  ncclComm_t InterSubComm = hierarchy.interComm;
  // INFO(NCCL_INIT, "reducescatter comp twoshot new");
  // void* sendCompbuff = nullptr;
  // void* recvCompbuff = nullptr;
  size_t compSendCount;
  ncclDataType_t compDatatype;
  CUDACHECK(cudaSetDevice(comm->cudaDev));

  size_t totalSendBytes = 2 * (nRanks + nNodes) * recvcount * ncclTypeSize(datatype);
  cocclBufferHandle workspaceBuffer = {};
  NCCLCHECK(cocclGetBuffer(comm, totalSendBytes, &workspaceBuffer));
  NCCLCHECK(cocclRegisterHierarchicalWorkspace(&workspaceBuffer, &hierarchy));
  void* workspaceBase = workspaceBuffer.ptr;
  NCCLCHECK(ncclCompress(sendbuff, &workspaceBase, recvcount, datatype, &compSendCount, &compDatatype, nRanks, comm->rank,
      comm, cocclHierarchicalPolicy(cocclOperation::AllReduce), stream));

  void* intraSendCompbuff = workspaceBase;
  void* intraRecvCompbuff =(char*) workspaceBase + compSendCount * nRanks * ncclTypeSize(compDatatype);
  // reuse buff may have some wrong, some data may not send/recv sometimes
  // swizzle and quan
  // Triple-shot hierarchy: intra-node exchange/reduce, inter-node
  // exchange/reduce, then global AllGather of compressed reduced chunks.
  size_t intraSendCount = compSendCount * nNodes;
  NCCLCHECK(ncclAllToAllNaive((void*)intraSendCompbuff, (void*)intraRecvCompbuff, intraSendCount, compDatatype, IntraSubComm, stream));
  size_t interOffset = 2 * compSendCount * nRanks;
  void* interSendCompbuff = (char*) workspaceBase + interOffset * ncclTypeSize(compDatatype);
  void* interRecvCompbuff = (char*) workspaceBase + (interOffset + compSendCount * nNodes) * ncclTypeSize(compDatatype);
   
  size_t reCompSendCount;
  ncclDataType_t reCompDatatype;
    // DecompReduceComp
  NCCLCHECK(ncclDecompReduceComp((void*)intraRecvCompbuff, &interSendCompbuff, recvcount * nNodes, datatype,
             intraSendCount, compDatatype, &reCompSendCount, &reCompDatatype, localRanks, comm, cocclHierarchicalPolicy(cocclOperation::AllReduce), stream));
    // inter alltoall
  size_t interSendCount = reCompSendCount / nNodes;

  NCCLCHECK(ncclAllToAllNaive((void*)interSendCompbuff, (void*)interRecvCompbuff, interSendCount, compDatatype, InterSubComm, stream));
    
  // DecompReduce
  
  NCCLCHECK(ncclDecompReduceComp((void*)interRecvCompbuff, &intraSendCompbuff, recvcount, datatype,
             interSendCount, compDatatype, &reCompSendCount, &reCompDatatype, nNodes, comm, cocclHierarchicalPolicy(cocclOperation::AllReduce), stream));
    
  struct ncclInfo info = { ncclFuncAllGather, "AllGather",
    workspaceBase, workspaceBase, reCompSendCount, reCompDatatype, ncclSum, 0, comm, stream, /* Args */
    ALLGATHER_CHUNKSTEPS, ALLGATHER_SLICESTEPS };
  NCCLCHECK(ncclEnqueueCheck(&info));

  // Decompress
  NCCLCHECK(ncclDecompress(recvbuff, workspaceBase, recvcount, datatype, reCompSendCount, reCompDatatype, comm->nRanks, comm, cocclHierarchicalPolicy(cocclOperation::AllReduce), stream));
  NCCLCHECK(cocclReleaseBuffer(&workspaceBuffer, stream));
  

  return ncclSuccess;
}
NCCL_API(ncclResult_t, ncclAllReduceCompRing, const void* sendbuff, void* recvbuff, size_t count,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduceCompRing(const void* sendbuff, void* recvbuff, size_t count,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  
  size_t chunkCount = count / comm->nRanks;

  char* r_recvbuf = (char*) recvbuff + comm->rank * chunkCount * ncclTypeSize(datatype);

  // Compose compressed AllReduce from the compressed ReduceScatter ring and the
  // compressed AllGather primitive.
  NCCLCHECK(ncclReduceScatterComp(sendbuff, r_recvbuf, chunkCount, datatype, op, comm, stream));

  NCCLCHECK(ncclAllGatherComp(r_recvbuf, recvbuff, chunkCount, datatype, comm, stream));
  
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclAllReduceCompOptim, const void* sendbuff, void* recvbuff, size_t count,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduceCompOptim(const void* sendbuff, void* recvbuff, size_t count,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream){
  // Size-based selector kept for compatibility with the original COCCL tuning.
  if(count * ncclTypeSize(datatype) < (size_t)1024 * 1024){
    NCCLCHECK(ncclAllReduceOneShot(sendbuff, recvbuff, count, datatype, op, comm, stream));
  }
  else if(count * ncclTypeSize(datatype) < (size_t)1024 * 1024 * 32){
    NCCLCHECK(ncclAllReduceCompTwoShot(sendbuff, recvbuff, count, datatype, op, comm, stream));
  }else{
    NCCLCHECK(ncclAllReduceCompRing(sendbuff, recvbuff, count, datatype, op, comm, stream));
  }
  return ncclSuccess;
}
