#include "coccl_wrapper.h"

// max allreduce sendSize
__thread size_t aRMaxSendSize = 0;
extern __thread void* compBuff;

NCCL_API(ncclResult_t, ncclAllReduceCompOneShot, const void* sendbuff, void* recvbuff, size_t count,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduceCompOneShot(const void* sendbuff, void* recvbuff, size_t count,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  // void* recvTempbuff = nullptr;
  size_t chunkCount = DIVUP(count, comm->nRanks);
  size_t numChunks = comm->nRanks;
  // CUDACHECK(cudaMallocAsync((void**)&recvTempbuff, comm->nRanks * numChunks * chunkCount * ncclTypeSize(datatype), stream));
  // Compress
  
  size_t compSendCount;
  ncclDataType_t compDatatype;
  CUDACHECK(cudaSetDevice(comm->cudaDev));
  size_t totalSendBytes = count * (comm->nRanks + comm->nRanks * comm->nRanks) * ncclTypeSize(datatype);
  bool mayUpdateBuff = compBuffHandle == nullptr || totalSendBytes > aRMaxSendSize;
  // NCCLCHECK(ncclCompress(sendbuff, chunkCount, datatype, &sendCompbuff, &compSendCount, &compDatatype, numChunks, stream));
  NCCLCHECK(ncclCompress(sendbuff, mayUpdateBuff ?  &recvbuff : &compBuff, chunkCount, datatype, &compSendCount, &compDatatype, comm->nRanks, ncclCommOp_t::AllReduce, stream));
  
  if(mayUpdateBuff){
    aRMaxSendSize = totalSendBytes;
    size_t compBuffBytes = compSendCount * (comm->nRanks + comm->nRanks * numChunks) * ncclTypeSize(compDatatype);
    allocAndRegCompBuff(comm, compBuffBytes);
    CUDACHECK(cudaMemcpy(compBuff, recvbuff, compSendCount * comm->nRanks * ncclTypeSize(compDatatype), cudaMemcpyDeviceToDevice));
    CUDACHECK(cudaDeviceSynchronize());
  }
  void* sendCompbuff = compBuff;
  void* recvCompbuff = (char*) compBuff + compSendCount * comm->nRanks * ncclTypeSize(compDatatype);

  // NCCLCHECK(ncclCompress(sendbuff, &sendCompbuff, chunkCount, datatype,  &compSendCount, &compDatatype, comm->nRanks, ncclCommOp_t::AllReduce, stream));

  // CUDACHECK(cudaMallocAsync((void**)&recvCompbuff,  comm->nRanks * numChunks * compSendCount * ncclTypeSize(compDatatype), stream));

  //Gather

  // P2P based - allchunk
  // in RTX 4090 platform it is faster than broadcast based and p2p chunk parallel 50% 
  // size 1K ~ 1M
  NCCLCHECK(ncclGroupStart());

  for(int r = 0; r < comm->nRanks; r++){

    char* r_recvbuf = (char*)recvCompbuff + r * numChunks * compSendCount * ncclTypeSize(compDatatype);
    NCCLCHECK(ncclSend(sendCompbuff, numChunks * compSendCount, compDatatype, r, comm, stream));
    NCCLCHECK(ncclRecv((void*)r_recvbuf, numChunks * compSendCount, compDatatype, r, comm, stream));

  }

  NCCLCHECK(ncclGroupEnd());



  NCCLCHECK(ncclDecompressReduce((void*)recvbuff, recvCompbuff, numChunks * compSendCount, compDatatype, numChunks * chunkCount, datatype, comm->nRanks,
  ncclCommOp_t::AllReduce, stream));
  // NCCLCHECK(ncclDecompress(recvTempbuff, (void*)recvCompbuff, numChunks * chunkCount, datatype, compSendCount, compDatatype, comm->nRanks, ncclCommOp_t::AllReduce, stream));
  // // Reduce chunk
  // NCCLCHECK(ncclReduceChunk(recvTempbuff, numChunks * chunkCount, recvbuff, datatype, comm->nRanks, stream));
  
  // CUDACHECK(cudaFreeAsync(sendCompbuff,stream));
  // CUDACHECK(cudaFreeAsync(recvCompbuff,stream));


  return ncclSuccess;
}



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
  bool mayUpdateBuff = compBuffHandle == nullptr || totalSendBytes > aRMaxSendSize;
 
  // reuse buff may have some wrong, some data may not send/recv sometimes
  // NCCLCHECK(ncclCompress(sendbuff, &sendCompbuff, chunkCount, datatype, &compSendCount, &compDatatype, comm->nRanks, ncclCommOp_t::AllReduce, stream));
  NCCLCHECK(ncclCompress(sendbuff, mayUpdateBuff ?  &recvbuff : &compBuff, chunkCount, datatype, &compSendCount, &compDatatype, comm->nRanks, ncclCommOp_t::AllReduce, stream));
  
  if(mayUpdateBuff){
    aRMaxSendSize = totalSendBytes;
    size_t compBuffBytes = 2 * compSendCount * comm->nRanks * ncclTypeSize(compDatatype);
    allocAndRegCompBuff(comm, compBuffBytes);
    CUDACHECK(cudaMemcpy(compBuff, recvbuff, compSendCount * comm->nRanks * ncclTypeSize(compDatatype), cudaMemcpyDeviceToDevice));
    CUDACHECK(cudaDeviceSynchronize());
  }
  void* sendCompbuff = compBuff;
  void* recvCompbuff = (char*) compBuff + compSendCount * comm->nRanks * ncclTypeSize(compDatatype);
  // CUDACHECK(cudaMallocAsync((void**)&recvCompbuff, compSendCount * comm->nRanks * ncclTypeSize(compDatatype), stream));
  //sendCompbuff + comm->nRanks * compSendCount * ncclTypeSize(ncclInt8)
  
  NCCLCHECK(ncclAllToAll((void*)sendCompbuff, (void*)recvCompbuff, compSendCount, compDatatype, comm, stream));

  size_t reCompSendCount;
  ncclDataType_t reCompDatatype;
  // DecompReduceComp
  NCCLCHECK(ncclDecompReduceComp((void*)recvCompbuff, &sendCompbuff, compSendCount, compDatatype, &reCompSendCount, &reCompDatatype, comm->nRanks,
                        ncclCommOp_t::AllReduce, stream));
  
  struct ncclInfo info = { ncclFuncAllGather, "AllGather",
    sendCompbuff, recvCompbuff, reCompSendCount, reCompDatatype, ncclSum, 0, comm, stream, /* Args */
    ALLGATHER_CHUNKSTEPS, ALLGATHER_SLICESTEPS };
  NCCLCHECK(ncclEnqueueCheck(&info));


  // Decompress
  NCCLCHECK(ncclDecompress(recvbuff, (void*)recvCompbuff, chunkCount, datatype, reCompSendCount, reCompDatatype, comm->nRanks, ncclCommOp_t::AllReduce, stream));
  
  return ncclSuccess;
}


NCCL_API(ncclResult_t, ncclAllReduceCompThripShotNewMultiStream, const void* sendbuff, void* recvbuff, size_t count,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduceCompThripShotNewMultiStream(const void* sendbuff, void* recvbuff, size_t count,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream){

  size_t chunkCount = DIVUP(count, comm->nRanks);
  size_t compSendCount;
  ncclDataType_t compDatatype;
  NCCLCHECK(ncclReduceScatterCompTwoShotNewMulti(sendbuff, (char*)recvbuff + chunkCount * ncclTypeSize(datatype), chunkCount, datatype, op, comm, stream))

  
  NCCLCHECK(ncclCompress((char*)recvbuff + chunkCount * ncclTypeSize(datatype),  &recvbuff, 
            chunkCount, datatype, &compSendCount, &compDatatype, 1, ncclCommOp_t::AllGather_Inter, stream));
  // Gather
  struct ncclInfo info = { ncclFuncAllGather, "AllGather",
    recvbuff, compBuff, compSendCount, compDatatype, ncclSum, 0, comm, stream, /* Args */
    ALLGATHER_CHUNKSTEPS, ALLGATHER_SLICESTEPS };
  NCCLCHECK(ncclEnqueueCheck(&info));

  // Decompress
  NCCLCHECK(ncclDecompress(recvbuff, compBuff, chunkCount, datatype, compSendCount, compDatatype, comm->nRanks, ncclCommOp_t::AllGather_Inter, stream));

  // NCCLCHECK(ncclAllGatherComp(recvbuff, recvbuff, chunkCount, datatype, comm, stream));

  // CUDACHECK(cudaFreeAsync(sendCompbuff, stream));
  // CUDACHECK(cudaFreeAsync(recvCompbuff, stream));
  // free(allInterRank);
  // free(allIntraRank);

  return ncclSuccess;
}


NCCL_API(ncclResult_t, ncclAllReduceCompRing, const void* sendbuff, void* recvbuff, size_t count,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduceCompRing(const void* sendbuff, void* recvbuff, size_t count,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  
  size_t chunkCount = count / comm->nRanks;
  // NCCLCHECK(initCompMemPool(comm->cudaDev, comm->localRanks));
  // NCCLCHECK(initCompressors());

  char* r_recvbuf = (char*) recvbuff + comm->rank * chunkCount * ncclTypeSize(datatype);

  NCCLCHECK(ncclReduceScatterComp(sendbuff, r_recvbuf, chunkCount, datatype, op, comm, stream));

  NCCLCHECK(ncclAllGatherComp(r_recvbuf, recvbuff, chunkCount, datatype, comm, stream));
  
  return ncclSuccess;
}
