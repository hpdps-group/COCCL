#include "coccl_wrapper.h"

// max allgather sendSize
__thread size_t aGMaxSendBytes = 0;
extern __thread void* compBuff;

NCCL_API(ncclResult_t, ncclAllGatherComp, const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclAllGatherComp(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  
  size_t compSendCount;
  ncclDataType_t compDatatype;
  CUDACHECK(cudaSetDevice(comm->cudaDev));
  size_t totalSendBytes = comm->nRanks * sendcount * ncclTypeSize(datatype);
  bool mayUpdateBuff = compBuffHandle == nullptr || totalSendBytes > aGMaxSendBytes;

  NCCLCHECK(ncclCompress(sendbuff, mayUpdateBuff ? &recvbuff: &compBuff, 
            sendcount, datatype, &compSendCount, &compDatatype, 1, ncclCommOp_t::AllGather, stream));
  // update the hold comp buffer
  if(mayUpdateBuff){
    aGMaxSendBytes = totalSendBytes;
    size_t compBuffBytes = compSendCount * comm->nRanks * ncclTypeSize(compDatatype);
    NCCLCHECK(cocclBuffAlloc(&compBuff, compBuffBytes, comm));
    // NCCLCHECK(allocAndRegCompBuff(comm, compBuffBytes));
    CUDACHECK(cudaMemcpy(compBuff, recvbuff, compSendCount * ncclTypeSize(compDatatype), cudaMemcpyDeviceToDevice));
    CUDACHECK(cudaDeviceSynchronize());
  }
  // Gather
  struct ncclInfo info = { ncclFuncAllGather, "AllGather",
    compBuff, compBuff, compSendCount, compDatatype, ncclSum, 0, comm, stream, /* Args */
    ALLGATHER_CHUNKSTEPS, ALLGATHER_SLICESTEPS };
  NCCLCHECK(ncclEnqueueCheck(&info));

  // Decompress
  NCCLCHECK(ncclDecompress(recvbuff, compBuff, sendcount, datatype, compSendCount, compDatatype, comm->nRanks, ncclCommOp_t::AllGather, stream));
  // if(sendbuff !=recvbuff && comm->rank == 0)
  // {
  //   // __nv_bfloat16 *send = (__nv_bfloat16*)malloc(sendcount*comm->nRanks * sizeof(__nv_bfloat16));
  //   __nv_bfloat16 *recv = (__nv_bfloat16*)malloc(sendcount*comm->nRanks * sizeof(__nv_bfloat16));
  //   cudaMemcpyAsync(recv, recvbuff, sendcount*comm->nRanks * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost,stream);
  //   // cudaMemcpyAsync(send, sendbuff, sendcount*comm->nRanks * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost,stream);
  //   cudaStreamSynchronize(stream);
  //   // CUDACHECK(cudaDeviceSynchronize());
  //   for(int r=0;r<comm->nRanks;r++){
  //         for(int i=0;i<sendcount;i++){
  //             INFO(NCCL_INIT, "rank %d chunkid %d elemIdx %d dequan %f", 
  //               comm->rank, r, i, __bfloat162float(recv[r*sendcount+i]));
  //         }
  //         // INFO(NCCL_INIT, "chunkid %d: ", r);
  //     }
  //   // INFO(NCCL_INIT,"rank: %d", comm->rank);
  //   // free(send);
  //   free(recv);
  // }

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
  for(int r = 0; r < comm->nRanks; r++){
    if(comm->rankToLocalRank[r] == comm->localRank) allInterRank[interCnt++] = r;
    if(comm->rankToNode[r] == comm->node) allIntraRank[intraCnt++] = r;
  }
  size_t compSendCount;
  ncclDataType_t compDatatype;
  CUDACHECK(cudaSetDevice(comm->cudaDev));
  size_t totalSendBytes = (comm->nRanks + 1) * sendcount * ncclTypeSize(datatype);
  bool mayUpdateBuff = compBuffHandle == nullptr || totalSendBytes > aGMaxSendBytes;
  
  // NCCLCHECK(ncclCompress(sendbuff, &sendCompbuff, sendcount, datatype , &compSendCount, &compDatatype, 1, ncclCommOp_t::AllGather, stream));
  NCCLCHECK(ncclCompress(sendbuff, mayUpdateBuff ? &recvbuff : &compBuff, sendcount, datatype , &compSendCount, &compDatatype, 1, ncclCommOp_t::AllGather, stream));

  if(mayUpdateBuff){
    aGMaxSendBytes = totalSendBytes;
    size_t compBuffBytes =  (comm->nRanks + 1) * compSendCount * ncclTypeSize(compDatatype);
    // maxCompBuffBytes  = compBuffBytes;
    NCCLCHECK(allocAndRegCompBuff(comm, compBuffBytes));
    CUDACHECK(cudaMemcpy(compBuff, recvbuff, compSendCount * ncclTypeSize(compDatatype), cudaMemcpyDeviceToDevice));
    CUDACHECK(cudaDeviceSynchronize());
  }
  
  void* sendCompbuff=compBuff;
  void* recvCompbuff=(char*)compBuff + compSendCount * ncclTypeSize(compDatatype);
  // CUDACHECK(cudaMallocAsync((void**)&recvCompbuff, compSendCount * comm->nRanks * ncclTypeSize(compDatatype), stream));
  // inter alltoall
  NCCLCHECK(ncclGroupStart());
  for(int r = 0; r < comm->nNodes; r++){
    int peer = allInterRank[r];
    char* r_sendbuf =(char*) sendCompbuff;
    char* r_recvbuf =(char*) recvCompbuff + peer * compSendCount * ncclTypeSize(compDatatype);
    NCCLCHECK(ncclRecv((void *)r_recvbuf, compSendCount, compDatatype, peer, comm, stream));
    NCCLCHECK(ncclSend((void *)r_sendbuf, compSendCount, compDatatype, peer, comm, stream));
  }
  NCCLCHECK(ncclGroupEnd());

  // intra alltoall
  NCCLCHECK(ncclGroupStart());
  for(int r = 0; r < comm->localRanks; r++){
    int peer = allIntraRank[r];
    if(peer == comm->rank) continue;
    for(int i = 0; i < comm->nNodes; i++){
      int sendLocation = allInterRank[i];
      int recvLocation = peer%comm->localRanks + i * comm->localRanks;
      char* r_sendbuf = (char*) recvCompbuff + sendLocation * compSendCount * ncclTypeSize(compDatatype);
      char* r_recvbuf = (char*) recvCompbuff + recvLocation * compSendCount * ncclTypeSize(compDatatype);
      NCCLCHECK(ncclSend((void *)r_sendbuf, compSendCount, compDatatype, peer, comm, stream));
      NCCLCHECK(ncclRecv((void *)r_recvbuf, compSendCount, compDatatype, peer, comm, stream));
    }
  }
  NCCLCHECK(ncclGroupEnd());

  // Decompress
  NCCLCHECK(ncclDecompress(recvbuff, (void*)recvCompbuff, sendcount, datatype, compSendCount, compDatatype, comm->nRanks, ncclCommOp_t::AllGather, stream));


  // Free
  // CUDACHECK(cudaFreeAsync(sendCompbuff, stream));
  // CUDACHECK(cudaFreeAsync(recvCompbuff, stream));
  free(allInterRank);
  free(allIntraRank);

  return ncclSuccess;
}
