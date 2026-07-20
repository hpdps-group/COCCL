#include "coccl_old_impl_internal.h"

NCCL_API(ncclResult_t, ncclReduceScatterCompOneShot, const void* sendbuff, void* recvbuff, size_t recvcount,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclReduceScatterCompOneShot(const void* sendbuff, void* recvbuff, size_t recvcount,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream){

  size_t compSendCount;
  ncclDataType_t compDatatype;
    
  CUDACHECK(cudaSetDevice(comm->cudaDev));
  size_t totalSendBytes = 2 * comm->nRanks * recvcount * ncclTypeSize(datatype);
  cocclBufferHandle workspaceBuffer = {};
  NCCLCHECK(cocclGetBuffer(comm, totalSendBytes, &workspaceBuffer));
  void* workspaceBase = workspaceBuffer.ptr;
  NCCLCHECK(ncclCompress(sendbuff, &workspaceBase, recvcount, datatype, &compSendCount, &compDatatype, comm->nRanks, comm->rank,
    comm, ncclCommOp_t::ReduceScatter, stream));

  void* sendCompbuff = workspaceBase;
  void* recvCompbuff =(char*) workspaceBase + compSendCount * comm->nRanks * ncclTypeSize(compDatatype);
    
  // One-shot path exchanges all compressed chunks, then does decomp-reduce.
  NCCLCHECK(ncclAllToAll((void*)sendCompbuff, (void*)recvCompbuff, compSendCount, compDatatype, comm, stream));

    // DecompReduce
  NCCLCHECK(ncclDecompressReduce((void*)recvbuff, (void*)recvCompbuff, compSendCount, compDatatype, recvcount, datatype, comm->nRanks,
                        comm, ncclCommOp_t::ReduceScatter, stream));
  NCCLCHECK(cocclReleaseBuffer(&workspaceBuffer, stream));
  return ncclSuccess;
}
NCCL_API(ncclResult_t, ncclReduceScatterComp, const void* sendbuff, void* recvbuff, size_t recvcount,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclReduceScatterComp(const void* sendbuff, void* recvbuff, size_t recvcount,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {

  int rightRank = (comm->rank + 1) % comm->nRanks;
  int leftRank = (comm->rank - 1 + comm->nRanks) % comm->nRanks;
  // INFO(NCCL_INIT, "coccl ReduceScatter comp ring");
  CUDACHECK(cudaSetDevice(comm->cudaDev));

  size_t compSendCount;
  ncclDataType_t compDatatype;
  size_t totalSendBytes = (2 + comm->nRanks) * recvcount * ncclTypeSize(datatype);
  cocclBufferHandle workspaceBuffer = {};
  NCCLCHECK(cocclGetBuffer(comm, totalSendBytes, &workspaceBuffer));
  void* workspaceBase = workspaceBuffer.ptr;
  NCCLCHECK(ncclCompress(sendbuff, &workspaceBase, recvcount, datatype, &compSendCount, &compDatatype, comm->nRanks, comm->rank,
  comm, ncclCommOp_t::ReduceScatter, stream));
  // void* reduceSendbuf = (char*) compBuff + comm->nRanks * compSendCount * ncclTypeSize(compDatatype);
  void* reduceRecvbuf = (char*) workspaceBase + (comm->nRanks + 1) * compSendCount * ncclTypeSize(compDatatype);
  void* reducebuff = (char*) workspaceBase + comm->nRanks * compSendCount * ncclTypeSize(compDatatype);

  // Compressed ring path forwards one compressed chunk per step. Middle steps
  // decomp-reduce-recompress before forwarding the partial result.
  for (int r = comm->nRanks - 1; r >= 0; r--) {
    // Ring step 0
    // compress - recv -  send
    int sendIdx = (comm->rank + r) % comm->nRanks;
    int recvIdx = (comm->rank + (r - 1) + comm->nRanks) % comm->nRanks;

    // CUDACHECK(cudaMemcpyAsync(reduceSendbuf, (char*)compBuff + sendIdx * compSendCount * ncclTypeSize(compDatatype), 
    //                                       compSendCount * ncclTypeSize(compDatatype), cudaMemcpyDeviceToDevice, stream));                            
    void* reduceSendbuf = (char*)workspaceBase + sendIdx * compSendCount * ncclTypeSize(compDatatype);
    if(r == comm->nRanks - 1){
      NCCLCHECK(ncclGroupStart());
      NCCLCHECK(ncclRecvNaive((void*)reduceRecvbuf, compSendCount, compDatatype, leftRank, comm, stream));
      NCCLCHECK(ncclSendNaive((void*)reduceSendbuf, compSendCount, compDatatype, rightRank, comm, stream));
      NCCLCHECK(ncclGroupEnd());

    } else if(r > 0) {
      // Ring step 1 ~ N - 2
      CUDACHECK(cudaMemcpyAsync(reducebuff, reduceSendbuf, compSendCount * ncclTypeSize(compDatatype), 
          cudaMemcpyDeviceToDevice, stream)); 
      size_t reCompChunkCount;
      ncclDataType_t reCompDatatype;
      // DecompReduceComp
      NCCLCHECK(ncclDecompReduceComp((void*)reducebuff, (void**)&reduceSendbuf, recvcount, datatype, 
                  compSendCount, compDatatype, &reCompChunkCount, &reCompDatatype, 2, comm, ncclCommOp_t::ReduceScatter, stream));

      NCCLCHECK(ncclGroupStart());
      NCCLCHECK(ncclRecvNaive((void*)reduceRecvbuf, reCompChunkCount, reCompDatatype, leftRank, comm, stream));
      NCCLCHECK(ncclSendNaive((void*)reduceSendbuf, reCompChunkCount, reCompDatatype, rightRank, comm, stream));
      NCCLCHECK(ncclGroupEnd());
    } else {
      // Ring step N - 1
        CUDACHECK(cudaMemcpyAsync(reducebuff, reduceSendbuf, compSendCount * ncclTypeSize(compDatatype), 
            cudaMemcpyDeviceToDevice, stream)); 
      // decompress - reduce
      NCCLCHECK(ncclDecompressReduce((void*)recvbuff, reducebuff, compSendCount, compDatatype, recvcount, datatype, 2,
                        comm, ncclCommOp_t::ReduceScatter, stream));
    }
  }
  NCCLCHECK(cocclReleaseBuffer(&workspaceBuffer, stream));
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclReduceScatterCompTwoShot, const void* sendbuff, void* recvbuff, size_t recvcount,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclReduceScatterCompTwoShot(const void* sendbuff, void* recvbuff, size_t recvcount,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream){
    // there could do inter and intra optimize, multiComm and multiStream

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
  comm, ncclCommOp_t::ReduceScatter_Inter, stream));

  void* intraSendCompbuff = workspaceBase;
  void* intraRecvCompbuff =(char*) workspaceBase + compSendCount * nRanks * ncclTypeSize(compDatatype);
  // reuse buff may have some wrong, some data may not send/recv sometimes
  // swizzle and quan
  // Stage 1 reduces within each node after an intra-node AllToAll.
  size_t intraSendCount = compSendCount * nNodes;
  NCCLCHECK(ncclAllToAll((void*)intraSendCompbuff, (void*)intraRecvCompbuff, intraSendCount, compDatatype, IntraSubComm, stream));
  size_t interOffset = 2 * compSendCount * nRanks;
  void* interSendCompbuff = (char*) workspaceBase + interOffset * ncclTypeSize(compDatatype);
  void* interRecvCompbuff = (char*) workspaceBase + (interOffset + compSendCount * nNodes) * ncclTypeSize(compDatatype);
   
  size_t reCompSendCount;
  ncclDataType_t reCompDatatype;
    // Stage 2 recompresses local partial reductions, exchanges them across
    // nodes, then performs the final decomp-reduce.
  NCCLCHECK(ncclDecompReduceComp((void*)intraRecvCompbuff, &interSendCompbuff, recvcount * nNodes, datatype,
             intraSendCount, compDatatype, &reCompSendCount, &reCompDatatype, localRanks, comm, ncclCommOp_t::ReduceScatter_Inter, stream));
    // inter alltoall
  size_t interSendCount = reCompSendCount / nNodes;

  NCCLCHECK(ncclAllToAll((void*)interSendCompbuff, (void*)interRecvCompbuff, interSendCount, compDatatype, InterSubComm, stream));
    
  // DecompReduce
  NCCLCHECK(ncclDecompressReduce((void*)recvbuff, interRecvCompbuff, interSendCount, reCompDatatype, recvcount, datatype, nNodes,
                        comm, ncclCommOp_t::ReduceScatter_Inter, stream));
  NCCLCHECK(cocclReleaseBuffer(&workspaceBuffer, stream));

  return ncclSuccess;
}

// max reduceScatter sendSize
