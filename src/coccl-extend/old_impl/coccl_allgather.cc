#include "coccl_primitives_internal.h"

NCCL_API(ncclResult_t, ncclAllGatherComp, const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclAllGatherComp(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  
  // Compress
  size_t compSendCount;
  ncclDataType_t compDatatype;
  CUDACHECK(cudaSetDevice(comm->cudaDev));
  size_t totalSendBytes = comm->nRanks * sendcount * ncclTypeSize(datatype);
  cocclBufferHandle workspaceBuffer = {};
  NCCLCHECK(cocclGetBuffer(comm, totalSendBytes, &workspaceBuffer));
  void* workspaceBase = workspaceBuffer.ptr;

  NCCLCHECK(ncclCompress(sendbuff, &workspaceBase,
            sendcount, datatype, &compSendCount, &compDatatype, 1, comm->rank, comm, ncclCommOp_t::AllGather, stream));

  // INFO(NCCL_INIT, "AllgatherComp_datatype_%d_totalcounts_%zu_totalbytes_%zuMB_compSendBytes_%zuMB_rank_%d_nRanks_%d_sendbuff_%p_recvbuff_%p_diff_%p_stream_%p", datatype, sendcount * comm->nRanks, 
  //   sendcount * comm->nRanks * ncclTypeSize(datatype)/ 1024 /1024, compSendCount * comm->nRanks * ncclTypeSize(compDatatype) / 1024/ 1024, 
  //   comm->rank, comm->nRanks, sendbuff, recvbuff, (char*)sendbuff - comm->rank * ncclTypeSize(datatype) * sendcount,(void*)stream);
  // Gather
  struct ncclInfo info = { ncclFuncAllGather, "AllGather",
    workspaceBase, workspaceBase, compSendCount, compDatatype, ncclSum, 0, comm, stream, /* Args */
    ALLGATHER_CHUNKSTEPS, ALLGATHER_SLICESTEPS };
  NCCLCHECK(ncclEnqueueCheck(&info));

  // Decompress
  NCCLCHECK(ncclDecompress(recvbuff, workspaceBase, sendcount, datatype, compSendCount, compDatatype, comm->nRanks, comm, ncclCommOp_t::AllGather, stream));
  NCCLCHECK(cocclReleaseBuffer(&workspaceBuffer, stream));

  return ncclSuccess;
}
// TODO inter- and intra- overlap
NCCL_API(ncclResult_t, ncclAllGatherCompTwoShot, const void* sendbuff, void* recvbuff, size_t sendcount,
  ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclAllGatherCompTwoShot(const void* sendbuff, void* recvbuff, size_t sendcount,
  ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  // optimize
  // Compress
 
  int* allIntraRank = (int*)malloc(comm->localRanks * sizeof(int));
  int* allInterRank = (int*)malloc(comm->nNodes * sizeof(int));
  int interCnt = 0, intraCnt = 0;
  // Precompute rank lists for two-shot exchange: same local rank across nodes
  // first, then peers within the local node.
  for(int r = 0; r < comm->nRanks; r++){
    if(comm->rankToLocalRank[r] == comm->localRank) allInterRank[interCnt++] = r;
    if(comm->rankToNode[r] == comm->node) allIntraRank[intraCnt++] = r;
  }
  size_t compSendCount;
  ncclDataType_t compDatatype;
  CUDACHECK(cudaSetDevice(comm->cudaDev));
  size_t totalSendBytes = (comm->nRanks + 1) * sendcount * ncclTypeSize(datatype);
  cocclBufferHandle workspaceBuffer = {};
  NCCLCHECK(cocclGetBuffer(comm, totalSendBytes, &workspaceBuffer));
  void* workspaceBase = workspaceBuffer.ptr;
  
  // NCCLCHECK(ncclCompress(sendbuff, &sendCompbuff, sendcount, datatype , &compSendCount, &compDatatype, 1, comm, ncclCommOp_t::AllGather, stream));
  NCCLCHECK(ncclCompress(sendbuff, &workspaceBase, sendcount, datatype , &compSendCount, &compDatatype, 1, comm->rank, comm, ncclCommOp_t::AllGather, stream));
  
  void* sendCompbuff=workspaceBase;
  void* recvCompbuff=(char*)workspaceBase + compSendCount * ncclTypeSize(compDatatype);
  // CUDACHECK(cudaMallocAsync((void**)&recvCompbuff, compSendCount * comm->nRanks * ncclTypeSize(compDatatype), stream));
  // Stage 1: exchange each local rank's compressed payload across nodes.
  NCCLCHECK(ncclGroupStart());
  for(int r = 0; r < comm->nNodes; r++){
    int peer = allInterRank[r];
    char* r_sendbuf =(char*) sendCompbuff;
    char* r_recvbuf =(char*) recvCompbuff + peer * compSendCount * ncclTypeSize(compDatatype);
    NCCLCHECK(ncclRecvNaive((void *)r_recvbuf, compSendCount, compDatatype, peer, comm, stream));
    NCCLCHECK(ncclSendNaive((void *)r_sendbuf, compSendCount, compDatatype, peer, comm, stream));
  }
  NCCLCHECK(ncclGroupEnd());

  // Stage 2: redistribute the inter-node results to ranks on the same node.
  NCCLCHECK(ncclGroupStart());
  for(int r = 0; r < comm->localRanks; r++){
    int peer = allIntraRank[r];
    if(peer == comm->rank) continue;
    for(int i = 0; i < comm->nNodes; i++){
      int sendLocation = allInterRank[i];
      int recvLocation = peer%comm->localRanks + i * comm->localRanks;
      char* r_sendbuf = (char*) recvCompbuff + sendLocation * compSendCount * ncclTypeSize(compDatatype);
      char* r_recvbuf = (char*) recvCompbuff + recvLocation * compSendCount * ncclTypeSize(compDatatype);
      NCCLCHECK(ncclSendNaive((void *)r_sendbuf, compSendCount, compDatatype, peer, comm, stream));
      NCCLCHECK(ncclRecvNaive((void *)r_recvbuf, compSendCount, compDatatype, peer, comm, stream));
    }
  }
  NCCLCHECK(ncclGroupEnd());

  // Decompress
  NCCLCHECK(ncclDecompress(recvbuff, (void*)recvCompbuff, sendcount, datatype, compSendCount, compDatatype, comm->nRanks, comm, ncclCommOp_t::AllGather, stream));


  // Free
  free(allInterRank);
  free(allIntraRank);
  NCCLCHECK(cocclReleaseBuffer(&workspaceBuffer, stream));

  return ncclSuccess;
}

// max alltoall sendSize
