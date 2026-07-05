#include "coccl_wrapper.h"

// max reducescatter sendSize
__thread size_t rSMaxSendSize = 0;
extern __thread void* compBuff;

NCCL_API(ncclResult_t, ncclReduceScatterCompOneShot, const void* sendbuff, void* recvbuff, size_t recvcount,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclReduceScatterCompOneShot(const void* sendbuff, void* recvbuff, size_t recvcount,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream){

  if(datatype == ncclDataType_t::ncclFloat16 || datatype == ncclDataType_t::ncclBfloat16){
    void* recvTempbuff = nullptr;
    // CUDACHECK(cudaMallocAsync((void**)&recvTempbuff, comm->nRanks * recvcount * ncclTypeSize(datatype), stream));
    CUDACHECK(cudaMallocFromPoolAsync((void**)&recvTempbuff, comm->nRanks * recvcount * ncclTypeSize(datatype), compMemPool[comm->cudaDev % compMemPoolCnt], stream));
    // Gather
    NCCLCHECK(ncclAllToAllComp(sendbuff, recvTempbuff, recvcount, datatype, comm, stream));
    // Reduce
    NCCLCHECK(ncclReduceChunk(recvTempbuff, recvcount, recvbuff, datatype, comm->nRanks, stream));

    CUDACHECK(cudaFreeAsync(recvTempbuff, stream));
  }
  else if(datatype == ncclDataType_t::ncclFloat32){
    size_t compSendCount;
    ncclDataType_t compDatatype;
    
    CUDACHECK(cudaSetDevice(comm->cudaDev));
    size_t totalSendBytes = 2 * comm->nRanks * recvcount * ncclTypeSize(datatype);
    bool mayUpdateBuff = compBuffHandle == nullptr || totalSendBytes > rSMaxSendSize;

    if(mayUpdateBuff){
      rSMaxSendSize = totalSendBytes;
      void* tempCompbuff = nullptr;
      NCCLCHECK(ncclCompress(sendbuff, &tempCompbuff, recvcount, datatype, &compSendCount, &compDatatype, comm->nRanks, 
        ncclCommOp_t::ReduceScatter, stream));
      size_t compBuffBytes = 2 * compSendCount * comm->nRanks * ncclTypeSize(compDatatype);
      allocAndRegCompBuff(comm, compBuffBytes);
      CUDACHECK(cudaMemcpy(compBuff, tempCompbuff, compSendCount * comm->nRanks * ncclTypeSize(compDatatype), cudaMemcpyDeviceToDevice));
      CUDACHECK(cudaDeviceSynchronize());
      CUDACHECK(cudaFree(tempCompbuff));
    } else {
      NCCLCHECK(ncclCompress(sendbuff, &compBuff, recvcount, datatype, &compSendCount, &compDatatype, comm->nRanks, 
        ncclCommOp_t::ReduceScatter, stream));
    }

    void* sendCompbuff = compBuff;
    void* recvCompbuff =(char*) compBuff + compSendCount * comm->nRanks * ncclTypeSize(compDatatype);
    
    // NCCLCHECK(ncclAllToAll((void*)sendCompbuff, (void*)recvCompbuff, compSendCount, compDatatype, comm, stream));
    NCCLCHECK(ncclAllToAll((void*)sendCompbuff, (void*)recvCompbuff, compSendCount, compDatatype, comm, stream));

    // DecompReduce
    // NCCLCHECK(ncclDecompressReduce((void*)recvbuff, recvCompbuff, compSendCount, compDatatype, recvcount, datatype, comm->nRanks,
    //                     ncclCommOp_t::ReduceScatter, stream));
    NCCLCHECK(ncclDecompressReduce((void*)recvbuff, (void*)recvCompbuff, compSendCount, compDatatype, recvcount, datatype, comm->nRanks,
                        ncclCommOp_t::ReduceScatter, stream));
  }
  
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclReduceScatterComp, const void* sendbuff, void* recvbuff, size_t recvcount,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclReduceScatterComp(const void* sendbuff, void* recvbuff, size_t recvcount,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {

  int rightRank = (comm->rank + 1) % comm->nRanks;
  int leftRank = (comm->rank - 1 + comm->nRanks) % comm->nRanks;
  // INFO(NCCL_INIT, "coccl ReduceScatter comp ring");


  size_t chunkBytes = recvcount * ncclTypeSize(datatype);
  CUDACHECK(cudaSetDevice(comm->cudaDev));

  size_t compSendCount;
  ncclDataType_t compDatatype;
  size_t totalSendBytes = (2 + comm->nRanks) * recvcount * ncclTypeSize(datatype);
  bool mayUpdateBuff = compBuffHandle == nullptr || totalSendBytes > rSMaxSendSize;

  if(mayUpdateBuff){
    rSMaxSendSize = totalSendBytes;
    void* tempCompbuff = nullptr;
    NCCLCHECK(ncclCompress(sendbuff, &tempCompbuff, recvcount, datatype, &compSendCount, &compDatatype, comm->nRanks, 
    ncclCommOp_t::ReduceScatter, stream));
    size_t compBuffBytes = compSendCount * (comm->nRanks + 2) * ncclTypeSize(compDatatype);
    allocAndRegCompBuff(comm, compBuffBytes);
    CUDACHECK(cudaMemcpy(compBuff, tempCompbuff, compSendCount * comm->nRanks * ncclTypeSize(compDatatype), cudaMemcpyDeviceToDevice));
    CUDACHECK(cudaDeviceSynchronize());
    CUDACHECK(cudaFree(tempCompbuff));
  } else {
    NCCLCHECK(ncclCompress(sendbuff, &compBuff, recvcount, datatype, &compSendCount, &compDatatype, comm->nRanks, 
    ncclCommOp_t::ReduceScatter, stream));
  }
  void* reduceSendbuf = (char*) compBuff + comm->nRanks * compSendCount * ncclTypeSize(compDatatype);
  void* reduceRecvbuf = (char*) compBuff + (comm->nRanks + 1) * compSendCount * ncclTypeSize(compDatatype);

  for (int r = comm->nRanks - 1; r >= 0; r--) {
    // Ring step 0
    // compress - recv -  send
    int sendIdx = (comm->rank + r) % comm->nRanks;
    int recvIdx = (comm->rank + (r - 1) + comm->nRanks) % comm->nRanks;

    CUDACHECK(cudaMemcpyAsync(reduceSendbuf, (char*)compBuff + sendIdx * compSendCount * ncclTypeSize(compDatatype), 
                                          compSendCount * ncclTypeSize(compDatatype), cudaMemcpyDeviceToDevice, stream));                            

    if(r == comm->nRanks - 1){
   
      NCCLCHECK(ncclGroupStart());
      NCCLCHECK(ncclRecv((void*)reduceRecvbuf, compSendCount, compDatatype, leftRank, comm, stream));
      NCCLCHECK(ncclSend((void*)reduceSendbuf, compSendCount, compDatatype, rightRank, comm, stream));
      NCCLCHECK(ncclGroupEnd());

    } else if(r > 0) {
      // Ring step 1 ~ N - 2
      // DecompReduceComp
      NCCLCHECK(ncclDecompReduceComp((void*)reduceSendbuf, (void**)&reduceSendbuf, compSendCount, compDatatype, &compSendCount, &compDatatype, 2,
                          ncclCommOp_t::ReduceScatter, stream));

      NCCLCHECK(ncclGroupStart());
      NCCLCHECK(ncclRecv((void*)reduceRecvbuf, compSendCount, compDatatype, leftRank, comm, stream));
      NCCLCHECK(ncclSend((void*)reduceSendbuf, compSendCount, compDatatype, rightRank, comm, stream));
      NCCLCHECK(ncclGroupEnd());
    } else {
      // Ring step N - 1
      // decompress - reduce
      NCCLCHECK(ncclDecompressReduce((void*)recvbuff, reduceSendbuf, compSendCount, compDatatype, recvcount, datatype, 2,
                        ncclCommOp_t::ReduceScatter, stream));
    }
  }
  

  return ncclSuccess;
}



__thread ncclComm_t InterSubComm=nullptr;
__thread ncclComm_t IntraSubComm=nullptr;
NCCL_API(ncclResult_t, ncclReduceScatterCompTwoShotNew, const void* sendbuff, void* recvbuff, size_t recvcount,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclReduceScatterCompTwoShotNew(const void* sendbuff, void* recvbuff, size_t recvcount,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream){
    // there could do inter and intra optimize, multiComm and multiStream

  int nRanks = comm->nRanks;
  int localRanks = comm->localRanks;
  int nNodes = nRanks / localRanks;
  if(InterSubComm == nullptr || IntraSubComm == nullptr){
    //intraSubComm
    NCCLCHECK(ncclCommSplit(comm, comm->rank / localRanks, comm->rank, &IntraSubComm, NULL));
    //interSubComm
    NCCLCHECK(ncclCommSplit(comm, comm->rank % localRanks, comm->rank, &InterSubComm, NULL));
  }
  // INFO(NCCL_INIT, "reducescatter comp twoshot new");
  // void* sendCompbuff = nullptr;
  // void* recvCompbuff = nullptr;
  size_t compSendCount;
  ncclDataType_t compDatatype;
  CUDACHECK(cudaSetDevice(comm->cudaDev));

  size_t totalSendBytes = 2 * (nRanks + nNodes) * recvcount * ncclTypeSize(datatype);
  bool mayUpdateBuff = compBuffHandle == nullptr || totalSendBytes > rSMaxSendSize;

  if(mayUpdateBuff){
    rSMaxSendSize = totalSendBytes;
    void* tempCompbuff = nullptr;
    NCCLCHECK(ncclCompress(sendbuff, &tempCompbuff, recvcount, datatype, &compSendCount, &compDatatype, nRanks, 
    ncclCommOp_t::ReduceScatter_Inter, stream));
    size_t compBuffBytes = 2 * compSendCount * (nRanks + nNodes) * ncclTypeSize(compDatatype);
    allocAndRegCompBuff(comm, compBuffBytes);
    // cudaMemset(compBuff, 0 , 2 * compSendCount * (nRanks + nNodes) * ncclTypeSize(compDatatype));
    CUDACHECK(cudaMemcpy(compBuff, tempCompbuff, compSendCount * nRanks * ncclTypeSize(compDatatype), cudaMemcpyDeviceToDevice));
    CUDACHECK(cudaDeviceSynchronize());
    CUDACHECK(cudaFree(tempCompbuff));
  } else {
    // cudaMemset(compBuff, 0 , 2 * compSendCount * (nRanks + nNodes) * ncclTypeSize(compDatatype));
    NCCLCHECK(ncclCompress(sendbuff, &compBuff, recvcount, datatype, &compSendCount, &compDatatype, nRanks, 
    ncclCommOp_t::ReduceScatter_Inter, stream));
  }

  void* intraSendCompbuff = compBuff;
  void* intraRecvCompbuff =(char*) compBuff + compSendCount * nRanks * ncclTypeSize(compDatatype);
  // reuse buff may have some wrong, some data may not send/recv sometimes
  // swizzle and quan
  // intra alltoall
  size_t intraSendCount = compSendCount * nNodes;
  NCCLCHECK(ncclAllToAll((void*)intraSendCompbuff, (void*)intraRecvCompbuff, intraSendCount, compDatatype, IntraSubComm, stream));
  size_t interOffset = 2 * compSendCount * nRanks;
  void* interSendCompbuff = (char*) compBuff + interOffset * ncclTypeSize(compDatatype);
  void* interRecvCompbuff = (char*) compBuff + (interOffset + compSendCount * nNodes) * ncclTypeSize(compDatatype);
   
  size_t reCompSendCount;
  ncclDataType_t reCompDatatype;
    // DecompReduceComp
  NCCLCHECK(ncclDecompReduceComp((void*)intraRecvCompbuff, &interSendCompbuff, intraSendCount, compDatatype, &reCompSendCount, &reCompDatatype, localRanks,
                      ncclCommOp_t::ReduceScatter_Inter, stream));
    // inter alltoall
  size_t interSendCount = reCompSendCount / nNodes;

  NCCLCHECK(ncclAllToAll((void*)interSendCompbuff, (void*)interRecvCompbuff, interSendCount, compDatatype, InterSubComm, stream));
    
  // DecompReduce
  NCCLCHECK(ncclDecompressReduce((void*)recvbuff, interRecvCompbuff, interSendCount, reCompDatatype, recvcount, datatype, nNodes,
                        ncclCommOp_t::ReduceScatter_Inter, stream));

  

  return ncclSuccess;
}



__thread cudaStream_t* pipelineStream;
__thread cudaEvent_t* pipelineEvent;
NCCL_API(ncclResult_t, ncclReduceScatterCompTwoShotNewMulti, const void* sendbuff, void* recvbuff, size_t recvcount,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclReduceScatterCompTwoShotNewMulti(const void* sendbuff, void* recvbuff, size_t recvcount,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream){

  
  CUDACHECK(cudaSetDevice(comm->cudaDev));
  int nRanks = comm->nRanks;
  int localRanks = comm->localRanks;
  int nNodes = nRanks / localRanks;
  if(InterSubComm == nullptr || IntraSubComm == nullptr){
    pipelineSize = ncclParamPipelineSize();
    pipelineStream = (cudaStream_t*)malloc(sizeof(cudaStream_t) * (pipelineSize));
    pipelineEvent = (cudaEvent_t*)malloc(sizeof(cudaEvent_t) * (pipelineSize));
    for(int i=0; i < pipelineSize; i++){
      CUDACHECK(cudaStreamCreateWithFlags(pipelineStream + i, cudaStreamNonBlocking));
      CUDACHECK(cudaEventCreateWithFlags(pipelineEvent + i, cudaEventDefault));
    }
    //intraSubComm
    NCCLCHECK(ncclCommSplit(comm, comm->rank / localRanks, comm->rank, &IntraSubComm, NULL));
    //interSubComm
    NCCLCHECK(ncclCommSplit(comm, comm->rank % localRanks, comm->rank, &InterSubComm, NULL));
  }
  // INFO(NCCL_INIT, "reducescatter comp twoshot new");
  // void* sendCompbuff = nullptr;
  // void* recvCompbuff = nullptr;
  size_t compSendCount;
  ncclDataType_t compDatatype;

  size_t totalSendBytes = 2 * (nRanks + nNodes) * recvcount * ncclTypeSize(datatype);
  bool mayUpdateBuff = compBuffHandle == nullptr || totalSendBytes > rSMaxSendSize;

  if(mayUpdateBuff){
    rSMaxSendSize = totalSendBytes;
    void* tempCompbuff = nullptr;
    NCCLCHECK(ncclCompress(sendbuff, &tempCompbuff, recvcount, datatype, &compSendCount, &compDatatype, nRanks, 
    ncclCommOp_t::ReduceScatter_Inter, stream));
    size_t compBuffBytes = 2 * compSendCount * (nRanks + nNodes) * ncclTypeSize(compDatatype);
    allocAndRegCompBuff(comm, compBuffBytes);
    // cudaMemset(compBuff, 0 , 2 * compSendCount * (nRanks + nNodes) * ncclTypeSize(compDatatype));
    CUDACHECK(cudaMemcpy(compBuff, tempCompbuff, compSendCount * nRanks * ncclTypeSize(compDatatype), cudaMemcpyDeviceToDevice));
    CUDACHECK(cudaDeviceSynchronize());
    CUDACHECK(cudaFree(tempCompbuff));
    if(mainEvent == nullptr)
      CUDACHECK(cudaEventCreateWithFlags(&mainEvent, cudaEventDefault));
  } else {
    // cudaMemset(compBuff, 0 , 2 * compSendCount * (nRanks + nNodes) * ncclTypeSize(compDatatype));
    NCCLCHECK(ncclCompress(sendbuff, &compBuff, recvcount, datatype, &compSendCount, &compDatatype, nRanks, 
      ncclCommOp_t::ReduceScatter_Inter, stream));
  }

  CUDACHECK(cudaEventRecord(mainEvent, stream));
  size_t pipeCompCount = compSendCount / pipelineSize;
  size_t pipeSendOffset = pipeCompCount * nRanks;
  size_t pipeOffset = pipeCompCount * (nRanks + 2 * nNodes);
  size_t totalSendCount = compSendCount * nRanks;
  for(int i = 0; i < pipelineSize; i++){
    CUDACHECK(cudaStreamWaitEvent(pipelineStream[i], mainEvent, 0));

    void* pipebuff = (char*)compBuff + (totalSendCount + i * pipeOffset) * ncclTypeSize(compDatatype);
    void* intraSendCompbuff = (char*)compBuff + i * pipeSendOffset * ncclTypeSize(compDatatype);
    void* intraRecvCompbuff = pipebuff;
    // reuse buff may have some wrong, some data may not send/recv sometimes
    // swizzle and quan
    // intra alltoall
    size_t intraSendCount = pipeCompCount * nNodes;
    NCCLCHECK(ncclAllToAll((void*)intraSendCompbuff, (void*)intraRecvCompbuff, intraSendCount, compDatatype, IntraSubComm, pipelineStream[i]));

    size_t interOffset = pipeCompCount * nRanks;
    void* interSendCompbuff = (char*) pipebuff + interOffset * ncclTypeSize(compDatatype);
    void* interRecvCompbuff = (char*) pipebuff + (interOffset + pipeCompCount * nNodes) * ncclTypeSize(compDatatype);
    
    size_t reCompSendCount;
    ncclDataType_t reCompDatatype;
      // DecompReduceComp
    NCCLCHECK(ncclDecompReduceComp((void*)intraRecvCompbuff, &interSendCompbuff, intraSendCount, compDatatype, &reCompSendCount, &reCompDatatype, 
                          localRanks, ncclCommOp_t::ReduceScatter_Inter, pipelineStream[i]));
      // inter alltoall
    size_t interSendCount = reCompSendCount / nNodes;

    NCCLCHECK(ncclAllToAll((void*)interSendCompbuff, (void*)interRecvCompbuff, interSendCount, compDatatype, InterSubComm, pipelineStream[i]));

    void* piperecvbuff = (char*) recvbuff + (size_t)i * recvcount / pipelineSize * ncclTypeSize(datatype);
    // DecompReduce
    NCCLCHECK(ncclDecompressReduce((void*)piperecvbuff, interRecvCompbuff, interSendCount, reCompDatatype, recvcount / pipelineSize, datatype, nNodes,
                          ncclCommOp_t::ReduceScatter_Inter, pipelineStream[i]));
  }

  for(int i=0;i<pipelineSize;i++){
    CUDACHECK(cudaEventRecord(pipelineEvent[i], pipelineStream[i]));  
    CUDACHECK(cudaStreamWaitEvent(stream, pipelineEvent[i], 0));
  }

  return ncclSuccess;
}
