
#include "nccl_comp_wrapper.h"
#include "nccl.h"
#include "argcheck.h"
#include "enqueue.h"
#include "compress.h"
#include "reduce_extend.h"
#include "compressor.h"
#include "../graph/topo.h"
#include "coccl_alloc.h"
#define COMPBUFF_EXCESS_SIZE 16
__thread struct parComm* parcomms = nullptr;
extern cudaMemPool_t* compMemPool;
extern size_t compMemPoolCnt;
// maxSendSize for allgather
__thread size_t aGMaxSendBytes = 0;


__thread void* aGbuff = nullptr;
__thread int pipelineDepth = 1;

NCCL_API(ncclResult_t, ncclAllGatherComp, const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclAllGatherComp(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  
  // Compress
  size_t compSendCount;
  ncclDataType_t compDatatype;
  CUDACHECK(cudaSetDevice(comm->cudaDev));
  size_t totalSendBytes = comm->nRanks * sendcount * ncclTypeSize(datatype);
  bool mayUpdateBuff = aGbuff == nullptr || totalSendBytes > aGMaxSendBytes;

  NCCLCHECK(ncclCompress(sendbuff, mayUpdateBuff ? &recvbuff: &aGbuff, 
            sendcount, datatype, &compSendCount, &compDatatype, 1, comm->rank, ncclCommOp_t::AllGather, stream));
  // update the hold comp buffer
  if(mayUpdateBuff){
    aGMaxSendBytes = totalSendBytes;
    size_t compBuffBytes = compSendCount * comm->nRanks * ncclTypeSize(compDatatype);
    NCCLCHECK(cocclBuffAlloc(&aGbuff, compBuffBytes, comm));
    CUDACHECK(cudaMemcpy(aGbuff, recvbuff, compSendCount * ncclTypeSize(compDatatype), cudaMemcpyDeviceToDevice));
    CUDACHECK(cudaDeviceSynchronize());
  }

  // INFO(NCCL_INIT, "AllgatherComp_datatype_%d_totalcounts_%zu_totalbytes_%zuMB_compSendBytes_%zuMB_rank_%d_nRanks_%d_sendbuff_%p_recvbuff_%p_diff_%p_stream_%p", datatype, sendcount * comm->nRanks, 
  //   sendcount * comm->nRanks * ncclTypeSize(datatype)/ 1024 /1024, compSendCount * comm->nRanks * ncclTypeSize(compDatatype) / 1024/ 1024, 
  //   comm->rank, comm->nRanks, sendbuff, recvbuff, (char*)sendbuff - comm->rank * ncclTypeSize(datatype) * sendcount,(void*)stream);
  // Gather
  struct ncclInfo info = { ncclFuncAllGather, "AllGather",
    aGbuff, aGbuff, compSendCount, compDatatype, ncclSum, 0, comm, stream, /* Args */
    ALLGATHER_CHUNKSTEPS, ALLGATHER_SLICESTEPS };
  NCCLCHECK(ncclEnqueueCheck(&info));

  // Decompress
  NCCLCHECK(ncclDecompress(recvbuff, aGbuff, sendcount, datatype, compSendCount, compDatatype, comm->nRanks, ncclCommOp_t::AllGather, stream));

  return ncclSuccess;
}
__thread cudaStream_t AGcompStream = nullptr, AGcommStream = nullptr, AGdecompStream = nullptr;
__thread cudaEvent_t AGcompEvent = nullptr, AGcommEvent = nullptr, AGdecompEvent = nullptr;
__thread cudaEvent_t AGmainEvent;

NCCL_API(ncclResult_t, ncclAllGatherCompOverlap, const void* sendbuff, void* recvbuff, size_t sendcount,
  ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclAllGatherCompOverlap(const void* sendbuff, void* recvbuff, size_t sendcount,
  ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
   // Compress
  size_t compSendCount;
  ncclDataType_t compDatatype;
  CUDACHECK(cudaSetDevice(comm->cudaDev));

  size_t totalSendBytes = (1 + comm->nRanks) * sendcount * ncclTypeSize(datatype);
  NCCLCHECK(cocclBuffAlloc(&aGbuff, totalSendBytes, comm));
  

  pipelineDepth = ncclParamPipelineDepth();


  if(pipelineDepth < 2){
    NCCLCHECK(ncclAllGatherComp(sendbuff, recvbuff, sendcount, datatype, comm, stream));
  }else{
    if(AGcompStream == nullptr){
      CUDACHECK(cudaStreamCreateWithFlags(&AGcompStream, cudaStreamNonBlocking));
      CUDACHECK(cudaStreamCreateWithFlags(&AGcommStream, cudaStreamNonBlocking));
      CUDACHECK(cudaStreamCreateWithFlags(&AGdecompStream, cudaStreamNonBlocking));
      CUDACHECK(cudaEventCreateWithFlags(&AGcompEvent, cudaEventDefault));
      CUDACHECK(cudaEventCreateWithFlags(&AGcommEvent, cudaEventDefault));
      CUDACHECK(cudaEventCreateWithFlags(&AGdecompEvent, cudaEventDefault));
      CUDACHECK(cudaEventCreateWithFlags(&AGmainEvent, cudaEventDefault));
    }
    CUDACHECK(cudaEventRecord(AGmainEvent, stream));
    CUDACHECK(cudaStreamWaitEvent(AGcompStream, AGmainEvent, 0));
    // NCCLCHECK(ncclCompress(sendbuff, &aGbuff, 
    //   sendcount, datatype, &compSendCount, &compDatatype, 1, ncclCommOp_t::AllGather, AGcompStream));
    // printf("dsadasdsadasdsad\n");
    // NCCLCHECK(ncclCompress(sendbuff, &aGbuff, 
    //   sendcount, datatype, &compSendCount, &compDatatype, 1, ncclCommOp_t::AllGather, AGcompStream));
    //   compSendCount /= pipelineDepth;
    // NCCLCHECK(ncclCompress(sendbuff, mayUpdateBuff ? &recvbuff: &aGbuff, 
    //   sendcount, datatype, &compSendCount, &compDatatype, 1, ncclCommOp_t::AllGather, stream));
    for(int i =0 ;i<pipelineDepth; i++){
      void* sbuff = (char*)sendbuff + i * sendcount / pipelineDepth * ncclTypeSize(datatype);
      void* sendCompbuff = (char*) aGbuff + i * sendcount / pipelineDepth  * ncclTypeSize(datatype);
      // void* sendCompbuff = (char*) aGbuff + i * compSendCount  * ncclTypeSize(compDatatype);

      NCCLCHECK(ncclCompress(sbuff, &sendCompbuff, 
          sendcount / pipelineDepth, datatype, &compSendCount, &compDatatype, 1, comm->rank, ncclCommOp_t::AllGather, AGcompStream));
      CUDACHECK(cudaEventRecord(AGcompEvent, AGcompStream));  
      CUDACHECK(cudaStreamWaitEvent(AGcommStream, AGcompEvent, 0));
      void* recvCompbuff = (char*) aGbuff + sendcount * ncclTypeSize(datatype) + i * compSendCount * comm->nRanks * ncclTypeSize(compDatatype);

      struct ncclInfo info = { ncclFuncAllGather, "AllGather",
        sendCompbuff, recvCompbuff, compSendCount, compDatatype, ncclSum, 0, comm, AGcommStream, /* Args */
        ALLGATHER_CHUNKSTEPS, ALLGATHER_SLICESTEPS };
      NCCLCHECK(ncclEnqueueCheck(&info));

      CUDACHECK(cudaEventRecord(AGcommEvent, AGcommStream));  
      CUDACHECK(cudaStreamWaitEvent(AGdecompStream, AGcommEvent, 0));
      void* rbuff = (char*)recvbuff + i * sendcount / pipelineDepth * comm->nRanks  * ncclTypeSize(datatype);
      NCCLCHECK(ncclDecompress(rbuff, (char*)recvCompbuff, sendcount / pipelineDepth, datatype, compSendCount, compDatatype, comm->nRanks, ncclCommOp_t::AllGather, AGdecompStream));
    }
    CUDACHECK(cudaEventRecord(AGdecompEvent, AGdecompStream));
    CUDACHECK(cudaStreamWaitEvent(stream, AGdecompEvent, 0));
  }

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
  bool mayUpdateBuff = aGbuff == nullptr || totalSendBytes > aGMaxSendBytes;
  
  // NCCLCHECK(ncclCompress(sendbuff, &sendCompbuff, sendcount, datatype , &compSendCount, &compDatatype, 1, ncclCommOp_t::AllGather, stream));
  NCCLCHECK(ncclCompress(sendbuff, mayUpdateBuff ? &recvbuff : &aGbuff, sendcount, datatype , &compSendCount, &compDatatype, 1, comm->rank, ncclCommOp_t::AllGather, stream));

  if(mayUpdateBuff){
    aGMaxSendBytes = totalSendBytes;
    size_t compBuffBytes =  (comm->nRanks + 1) * compSendCount * ncclTypeSize(compDatatype);
    NCCLCHECK(cocclBuffAlloc(&aGbuff, compBuffBytes, comm));
    CUDACHECK(cudaMemcpy(aGbuff, recvbuff, compSendCount * ncclTypeSize(compDatatype), cudaMemcpyDeviceToDevice));
    CUDACHECK(cudaDeviceSynchronize());
  }
  
  void* sendCompbuff=aGbuff;
  void* recvCompbuff=(char*)aGbuff + compSendCount * ncclTypeSize(compDatatype);
  // CUDACHECK(cudaMallocAsync((void**)&recvCompbuff, compSendCount * comm->nRanks * ncclTypeSize(compDatatype), stream));
  // inter alltoall
  NCCLCHECK(ncclGroupStart());
  for(int r = 0; r < comm->nNodes; r++){
    int peer = allInterRank[r];
    char* r_sendbuf =(char*) sendCompbuff;
    char* r_recvbuf =(char*) recvCompbuff + peer * compSendCount * ncclTypeSize(compDatatype);
    NCCLCHECK(ncclRecvNaive((void *)r_recvbuf, compSendCount, compDatatype, peer, comm, stream));
    NCCLCHECK(ncclSendNaive((void *)r_sendbuf, compSendCount, compDatatype, peer, comm, stream));
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
      NCCLCHECK(ncclSendNaive((void *)r_sendbuf, compSendCount, compDatatype, peer, comm, stream));
      NCCLCHECK(ncclRecvNaive((void *)r_recvbuf, compSendCount, compDatatype, peer, comm, stream));
    }
  }
  NCCLCHECK(ncclGroupEnd());

  // Decompress
  NCCLCHECK(ncclDecompress(recvbuff, (void*)recvCompbuff, sendcount, datatype, compSendCount, compDatatype, comm->nRanks, ncclCommOp_t::AllGather, stream));


  // Free
  free(allInterRank);
  free(allIntraRank);

  return ncclSuccess;
}

// max alltoall sendSize
__thread size_t a2AMaxSendSize = 0;
__thread void* a2Abuff = nullptr;

NCCL_API(ncclResult_t, ncclAllToAllComp, const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
ncclResult_t  ncclAllToAllComp(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  
  // Compress
  size_t compSendCount;
  ncclDataType_t compDatatype;
  CUDACHECK(cudaSetDevice(comm->cudaDev));

  size_t totalSendBytes = 2 * comm->nRanks * sendcount * ncclTypeSize(datatype);
  NCCLCHECK(cocclBuffAlloc(&a2Abuff, totalSendBytes, comm));

  bool mayUpdateBuff = a2Abuff == nullptr || totalSendBytes > a2AMaxSendSize;
  // reuse buff may have some wrong, some data may not send/recv sometimes
  // NCCLCHECK(ncclCompress(sendbuff, &sendCompbuff, sendcount, datatype, &compSendCount, &compDatatype, comm->nRanks, ncclCommOp_t::AlltoAll, stream));
  NCCLCHECK(ncclCompress(sendbuff, mayUpdateBuff ?  &recvbuff : &a2Abuff, sendcount, datatype, &compSendCount, &compDatatype, comm->nRanks, comm->rank, ncclCommOp_t::AlltoAll, stream));

  // if(mayUpdateBuff){
  //   a2AMaxSendSize = totalSendBytes;
  //   size_t compBuffBytes = 2 * (compSendCount * comm->nRanks * ncclTypeSize(compDatatype));
  //   NCCLCHECK(cocclBuffAlloc(&a2Abuff, compBuffBytes, comm));
  //   CUDACHECK(cudaMemcpy(a2Abuff, recvbuff, compSendCount * comm->nRanks * ncclTypeSize(compDatatype), cudaMemcpyDeviceToDevice));
  //   CUDACHECK(cudaDeviceSynchronize());
  // }
  void* sendCompbuff = a2Abuff;
  void* recvCompbuff = (char*)a2Abuff + compSendCount * comm->nRanks * ncclTypeSize(compDatatype);
  
  NCCLCHECK(ncclAllToAll((void*)sendCompbuff, (void*)recvCompbuff, compSendCount, compDatatype, comm, stream));

  NCCLCHECK(ncclDecompress(recvbuff, (char*)recvCompbuff, sendcount, datatype, compSendCount, compDatatype, comm->nRanks, ncclCommOp_t::AlltoAll, stream));
  
  return ncclSuccess;
}
__thread cudaStream_t A2AcompStream = nullptr, A2AcommStream = nullptr, A2AdecompStream = nullptr;
__thread cudaEvent_t A2AcompEvent = nullptr, A2AcommEvent = nullptr, A2AdecompEvent = nullptr;
__thread cudaEvent_t A2AmainEvent;

// TODO comm- and comp- overlap
NCCL_API(ncclResult_t, ncclAlltoAllCompOverlap, const void* sendbuff, void* recvbuff, size_t sendcount,
  ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
ncclResult_t  ncclAlltoAllCompOverlap(const void* sendbuff, void* recvbuff, size_t sendcount,
  ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {

  // Compress
  size_t compSendCount;
  ncclDataType_t compDatatype;
  CUDACHECK(cudaSetDevice(comm->cudaDev));

  size_t totalSendBytes = 2 * comm->nRanks * sendcount * ncclTypeSize(datatype);
  NCCLCHECK(cocclBuffAlloc(&a2Abuff, totalSendBytes, comm));
  
  pipelineDepth = ncclParamPipelineDepth();

  if(pipelineDepth < 2){
    NCCLCHECK(ncclAllToAllComp(sendbuff, recvbuff, sendcount, datatype, comm, stream));
  }else{
    if(A2AcompStream == nullptr){
      CUDACHECK(cudaStreamCreateWithFlags(&A2AcompStream, cudaStreamNonBlocking));
      CUDACHECK(cudaStreamCreateWithFlags(&A2AcommStream, cudaStreamNonBlocking));
      CUDACHECK(cudaStreamCreateWithFlags(&A2AdecompStream, cudaStreamNonBlocking));
      CUDACHECK(cudaEventCreateWithFlags(&A2AcompEvent, cudaEventDefault));
      CUDACHECK(cudaEventCreateWithFlags(&A2AcommEvent, cudaEventDefault));
      CUDACHECK(cudaEventCreateWithFlags(&A2AdecompEvent, cudaEventDefault));
      CUDACHECK(cudaEventCreateWithFlags(&A2AmainEvent, cudaEventDefault));
    }
    CUDACHECK(cudaEventRecord(A2AmainEvent, stream));
    CUDACHECK(cudaStreamWaitEvent(A2AcompStream, A2AmainEvent, 0));
    for(int i =0 ;i<pipelineDepth; i++){
      void* sbuff = (char*)sendbuff + i * sendcount / pipelineDepth * comm->nRanks * ncclTypeSize(datatype);
      void* sendCompbuff = (char*) a2Abuff + i * sendcount / pipelineDepth * comm->nRanks  * ncclTypeSize(datatype);

      NCCLCHECK(ncclCompress(sbuff, &sendCompbuff, sendcount / pipelineDepth, datatype, &compSendCount, &compDatatype, comm->nRanks, comm->rank, ncclCommOp_t::AlltoAll, A2AcompStream));
      CUDACHECK(cudaEventRecord(A2AcompEvent, A2AcompStream));  
      CUDACHECK(cudaStreamWaitEvent(A2AcommStream, A2AcompEvent, 0));
      void* recvCompbuff = (char*) a2Abuff + sendcount * comm->nRanks * ncclTypeSize(datatype) + i * compSendCount * comm->nRanks * ncclTypeSize(compDatatype);

      NCCLCHECK(ncclAllToAll((void*)sendCompbuff, (void*)recvCompbuff, compSendCount, compDatatype, comm, A2AcommStream));

      CUDACHECK(cudaEventRecord(A2AcommEvent, A2AcommStream));  
      CUDACHECK(cudaStreamWaitEvent(A2AdecompStream, A2AcommEvent, 0));
      void* rbuff = (char*)recvbuff + i * sendcount / pipelineDepth * comm->nRanks  * ncclTypeSize(datatype);
      NCCLCHECK(ncclDecompress(rbuff, (char*)recvCompbuff, sendcount / pipelineDepth, datatype, compSendCount, compDatatype, comm->nRanks, ncclCommOp_t::AlltoAll, A2AdecompStream));
    }
    CUDACHECK(cudaEventRecord(A2AdecompEvent, A2AdecompStream));
    CUDACHECK(cudaStreamWaitEvent(stream, A2AdecompEvent, 0));
  }
 
  return ncclSuccess;
}



__thread size_t rSMaxSendSize = 0;
__thread void* rSbuff = nullptr;
NCCL_API(ncclResult_t, ncclReduceScatterCompOneShot, const void* sendbuff, void* recvbuff, size_t recvcount,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclReduceScatterCompOneShot(const void* sendbuff, void* recvbuff, size_t recvcount,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream){

  size_t compSendCount;
  ncclDataType_t compDatatype;
    
  CUDACHECK(cudaSetDevice(comm->cudaDev));
  size_t totalSendBytes = 2 * comm->nRanks * recvcount * ncclTypeSize(datatype);
  bool mayUpdateBuff = rSbuff == nullptr || totalSendBytes > rSMaxSendSize;
  // printf("totalSendBytes %lu rSMaxSendSize %lu mayUpdateBuff %d\n", totalSendBytes, rSMaxSendSize, mayUpdateBuff);
  if(mayUpdateBuff){
    rSMaxSendSize = totalSendBytes;
    void* tempCompbuff = nullptr;
    NCCLCHECK(ncclCompress(sendbuff, &tempCompbuff, recvcount, datatype, &compSendCount, &compDatatype, comm->nRanks, comm->rank,
      ncclCommOp_t::ReduceScatter, stream));
    size_t compBuffBytes = 2 * compSendCount * comm->nRanks * ncclTypeSize(compDatatype);
    // printf("compBuffBytes")
    NCCLCHECK(cocclBuffAlloc(&rSbuff, compBuffBytes, comm));
    CUDACHECK(cudaMemcpy(rSbuff, tempCompbuff, compSendCount * comm->nRanks * ncclTypeSize(compDatatype), cudaMemcpyDeviceToDevice));
    CUDACHECK(cudaDeviceSynchronize());
    CUDACHECK(cudaFree(tempCompbuff));
  } else {
    // printf("Asdsadasd\n");
    NCCLCHECK(ncclCompress(sendbuff, &rSbuff, recvcount, datatype, &compSendCount, &compDatatype, comm->nRanks, comm->rank,
      ncclCommOp_t::ReduceScatter, stream));
  }

  void* sendCompbuff = rSbuff;
  void* recvCompbuff =(char*) rSbuff + compSendCount * comm->nRanks * ncclTypeSize(compDatatype);
    
  NCCLCHECK(ncclAllToAll((void*)sendCompbuff, (void*)recvCompbuff, compSendCount, compDatatype, comm, stream));

    // DecompReduce
  NCCLCHECK(ncclDecompressReduce((void*)recvbuff, (void*)recvCompbuff, compSendCount, compDatatype, recvcount, datatype, comm->nRanks,
                        ncclCommOp_t::ReduceScatter, stream));
  return ncclSuccess;
}
__thread cudaStream_t RScompStream = nullptr, RScommStream = nullptr, RSdecompStream = nullptr;
__thread cudaEvent_t RScompEvent = nullptr, RScommEvent = nullptr, RSdecompEvent = nullptr;
__thread cudaEvent_t RSmainEvent;

NCCL_API(ncclResult_t, ncclReduceScatterCompOneShotOverlap, const void* sendbuff, void* recvbuff, size_t recvcount,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclReduceScatterCompOneShotOverlap(const void* sendbuff, void* recvbuff, size_t recvcount,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream){

  size_t compSendCount;
  ncclDataType_t compDatatype;
    
  CUDACHECK(cudaSetDevice(comm->cudaDev));
  size_t totalSendBytes = 2 * comm->nRanks * recvcount * ncclTypeSize(datatype);
  NCCLCHECK(cocclBuffAlloc(&rSbuff, totalSendBytes, comm));

  
  pipelineDepth = ncclParamPipelineDepth();

  if(pipelineDepth < 2){
    NCCLCHECK(ncclReduceScatterCompOneShot(sendbuff, recvbuff, recvcount, datatype, op, comm, stream));
  }else{
      if(RScompStream == nullptr){
        CUDACHECK(cudaStreamCreateWithFlags(&RScompStream, cudaStreamNonBlocking));
        CUDACHECK(cudaStreamCreateWithFlags(&RScommStream, cudaStreamNonBlocking));
        CUDACHECK(cudaStreamCreateWithFlags(&RSdecompStream, cudaStreamNonBlocking));
        CUDACHECK(cudaEventCreateWithFlags(&RScompEvent, cudaEventDefault));
        CUDACHECK(cudaEventCreateWithFlags(&RScommEvent, cudaEventDefault));
        CUDACHECK(cudaEventCreateWithFlags(&RSdecompEvent, cudaEventDefault));
        CUDACHECK(cudaEventCreateWithFlags(&RSmainEvent, cudaEventDefault));
      }
      CUDACHECK(cudaEventRecord(RSmainEvent, stream));
      CUDACHECK(cudaStreamWaitEvent(RScompStream, RSmainEvent, 0));
      for(int i =0 ;i<pipelineDepth; i++){
        void* sbuff = (char*)sendbuff + i * recvcount / pipelineDepth * comm->nRanks * ncclTypeSize(datatype);
        void* sendCompbuff = (char*) rSbuff + i * recvcount / pipelineDepth * comm->nRanks  * ncclTypeSize(datatype);
  
        NCCLCHECK(ncclCompress(sbuff, &sendCompbuff, recvcount / pipelineDepth, datatype, &compSendCount, &compDatatype, comm->nRanks, comm->rank, ncclCommOp_t::ReduceScatter, RScompStream));
        CUDACHECK(cudaEventRecord(RScompEvent, RScompStream));  
        CUDACHECK(cudaStreamWaitEvent(RScommStream, RScompEvent, 0));
        void* recvCompbuff = (char*) rSbuff + recvcount * comm->nRanks * ncclTypeSize(datatype) + i * compSendCount * comm->nRanks * ncclTypeSize(compDatatype);
  
        NCCLCHECK(ncclAllToAll((void*)sendCompbuff, (void*)recvCompbuff, compSendCount, compDatatype, comm, RScommStream));
  
        CUDACHECK(cudaEventRecord(RScommEvent, RScommStream));  
        CUDACHECK(cudaStreamWaitEvent(RSdecompStream, RScommEvent, 0));
        void* rbuff = (char*)recvbuff + i * recvcount / pipelineDepth * ncclTypeSize(datatype);
        NCCLCHECK(ncclDecompressReduce((void*)rbuff, (void*)recvCompbuff, compSendCount, compDatatype, recvcount / pipelineDepth, datatype, comm->nRanks,
            ncclCommOp_t::ReduceScatter, RSdecompStream));
      }
      CUDACHECK(cudaEventRecord(RSdecompEvent, RSdecompStream));
      CUDACHECK(cudaStreamWaitEvent(stream, RSdecompEvent, 0));

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
  bool mayUpdateBuff = rSbuff == nullptr || totalSendBytes > rSMaxSendSize;

  if(mayUpdateBuff){
    rSMaxSendSize = totalSendBytes;
    void* tempCompbuff = nullptr;
    NCCLCHECK(ncclCompress(sendbuff, &tempCompbuff, recvcount, datatype, &compSendCount, &compDatatype, comm->nRanks, comm->rank, 
    ncclCommOp_t::ReduceScatter, stream));
    size_t compBuffBytes = compSendCount * (comm->nRanks + 2) * ncclTypeSize(compDatatype);
    NCCLCHECK(cocclBuffAlloc(&rSbuff, compBuffBytes, comm));
    CUDACHECK(cudaMemcpy(rSbuff, tempCompbuff, compSendCount * comm->nRanks * ncclTypeSize(compDatatype), cudaMemcpyDeviceToDevice));
    CUDACHECK(cudaDeviceSynchronize());
    CUDACHECK(cudaFree(tempCompbuff));
  } else {
    NCCLCHECK(ncclCompress(sendbuff, &rSbuff, recvcount, datatype, &compSendCount, &compDatatype, comm->nRanks, comm->rank, 
    ncclCommOp_t::ReduceScatter, stream));
  }
  // void* reduceSendbuf = (char*) compBuff + comm->nRanks * compSendCount * ncclTypeSize(compDatatype);
  void* reduceRecvbuf = (char*) rSbuff + (comm->nRanks + 1) * compSendCount * ncclTypeSize(compDatatype);
  void* reducebuff = (char*) rSbuff + comm->nRanks * compSendCount * ncclTypeSize(compDatatype);

  for (int r = comm->nRanks - 1; r >= 0; r--) {
    // Ring step 0
    // compress - recv -  send
    int sendIdx = (comm->rank + r) % comm->nRanks;
    int recvIdx = (comm->rank + (r - 1) + comm->nRanks) % comm->nRanks;

    // CUDACHECK(cudaMemcpyAsync(reduceSendbuf, (char*)compBuff + sendIdx * compSendCount * ncclTypeSize(compDatatype), 
    //                                       compSendCount * ncclTypeSize(compDatatype), cudaMemcpyDeviceToDevice, stream));                            
    void* reduceSendbuf = (char*)rSbuff + sendIdx * compSendCount * ncclTypeSize(compDatatype);
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
                  compSendCount, compDatatype, &reCompChunkCount, &reCompDatatype, 2, ncclCommOp_t::ReduceScatter, stream));

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
                        ncclCommOp_t::ReduceScatter, stream));
    }
  }
  
  return ncclSuccess;
}

__thread ncclComm_t InterSubComm=nullptr;
__thread ncclComm_t IntraSubComm=nullptr;
NCCL_API(ncclResult_t, ncclReduceScatterCompTwoShot, const void* sendbuff, void* recvbuff, size_t recvcount,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclReduceScatterCompTwoShot(const void* sendbuff, void* recvbuff, size_t recvcount,
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
  bool mayUpdateBuff = rSbuff == nullptr || totalSendBytes > rSMaxSendSize;

  if(mayUpdateBuff){
    rSMaxSendSize = totalSendBytes;
    void* tempCompbuff = nullptr;
    NCCLCHECK(ncclCompress(sendbuff, &tempCompbuff, recvcount, datatype, &compSendCount, &compDatatype, nRanks, comm->rank,
    ncclCommOp_t::ReduceScatter_Inter, stream));
    size_t compBuffBytes = 2 * compSendCount * (nRanks + nNodes) * ncclTypeSize(compDatatype);
    NCCLCHECK(cocclBuffAlloc(&rSbuff, compBuffBytes, comm));
    // cudaMemset(compBuff, 0 , 2 * compSendCount * (nRanks + nNodes) * ncclTypeSize(compDatatype));
    CUDACHECK(cudaMemcpy(rSbuff, tempCompbuff, compSendCount * nRanks * ncclTypeSize(compDatatype), cudaMemcpyDeviceToDevice));
    CUDACHECK(cudaDeviceSynchronize());
    CUDACHECK(cudaFree(tempCompbuff));
  } else {
    // cudaMemset(compBuff, 0 , 2 * compSendCount * (nRanks + nNodes) * ncclTypeSize(compDatatype));
    NCCLCHECK(ncclCompress(sendbuff, &rSbuff, recvcount, datatype, &compSendCount, &compDatatype, nRanks, comm->rank, 
    ncclCommOp_t::ReduceScatter_Inter, stream));
  }

  void* intraSendCompbuff = rSbuff;
  void* intraRecvCompbuff =(char*) rSbuff + compSendCount * nRanks * ncclTypeSize(compDatatype);
  // reuse buff may have some wrong, some data may not send/recv sometimes
  // swizzle and quan
  // intra alltoall
  size_t intraSendCount = compSendCount * nNodes;
  NCCLCHECK(ncclAllToAll((void*)intraSendCompbuff, (void*)intraRecvCompbuff, intraSendCount, compDatatype, IntraSubComm, stream));
  size_t interOffset = 2 * compSendCount * nRanks;
  void* interSendCompbuff = (char*) rSbuff + interOffset * ncclTypeSize(compDatatype);
  void* interRecvCompbuff = (char*) rSbuff + (interOffset + compSendCount * nNodes) * ncclTypeSize(compDatatype);
   
  size_t reCompSendCount;
  ncclDataType_t reCompDatatype;
    // DecompReduceComp
  NCCLCHECK(ncclDecompReduceComp((void*)intraRecvCompbuff, &interSendCompbuff, recvcount * nNodes, datatype,
             intraSendCount, compDatatype, &reCompSendCount, &reCompDatatype, localRanks, ncclCommOp_t::ReduceScatter_Inter, stream));
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
__thread cudaEvent_t mainEvent;
NCCL_API(ncclResult_t, ncclReduceScatterCompTwoShotOverlap, const void* sendbuff, void* recvbuff, size_t recvcount,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclReduceScatterCompTwoShotOverlap(const void* sendbuff, void* recvbuff, size_t recvcount,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream){

  
  CUDACHECK(cudaSetDevice(comm->cudaDev));
  int nRanks = comm->nRanks;
  int localRanks = comm->localRanks;
  int nNodes = nRanks / localRanks;
  if(InterSubComm == nullptr || IntraSubComm == nullptr){
    pipelineDepth = ncclParamPipelineDepth();
    pipelineStream = (cudaStream_t*)malloc(sizeof(cudaStream_t) * (pipelineDepth));
    pipelineEvent = (cudaEvent_t*)malloc(sizeof(cudaEvent_t) * (pipelineDepth));
    for(int i=0; i < pipelineDepth; i++){
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
  bool mayUpdateBuff = rSbuff == nullptr || totalSendBytes > rSMaxSendSize;

  if(mayUpdateBuff){
    rSMaxSendSize = totalSendBytes;
    void* tempCompbuff = nullptr;
    NCCLCHECK(ncclCompress(sendbuff, &tempCompbuff, recvcount, datatype, &compSendCount, &compDatatype, nRanks, comm->rank, 
    ncclCommOp_t::ReduceScatter_Inter, stream));
    size_t compBuffBytes = 2 * compSendCount * (nRanks + nNodes) * ncclTypeSize(compDatatype);
    NCCLCHECK(cocclBuffAlloc(&rSbuff, compBuffBytes, comm));
    // cudaMemset(compBuff, 0 , 2 * compSendCount * (nRanks + nNodes) * ncclTypeSize(compDatatype));
    CUDACHECK(cudaMemcpy(rSbuff, tempCompbuff, compSendCount * nRanks * ncclTypeSize(compDatatype), cudaMemcpyDeviceToDevice));
    CUDACHECK(cudaDeviceSynchronize());
    CUDACHECK(cudaFree(tempCompbuff));
    if(mainEvent == nullptr)
      CUDACHECK(cudaEventCreateWithFlags(&mainEvent, cudaEventDefault));
  } else {
    // cudaMemset(compBuff, 0 , 2 * compSendCount * (nRanks + nNodes) * ncclTypeSize(compDatatype));
    NCCLCHECK(ncclCompress(sendbuff, &rSbuff, recvcount, datatype, &compSendCount, &compDatatype, nRanks, comm->rank, 
      ncclCommOp_t::ReduceScatter_Inter, stream));
  }

  CUDACHECK(cudaEventRecord(mainEvent, stream));
  size_t pipeCompCount = compSendCount / pipelineDepth;
  size_t pipeSendOffset = pipeCompCount * nRanks;
  size_t pipeOffset = pipeCompCount * (nRanks + 2 * nNodes);
  size_t totalSendCount = compSendCount * nRanks;
  for(int i = 0; i < pipelineDepth; i++){
    CUDACHECK(cudaStreamWaitEvent(pipelineStream[i], mainEvent, 0));

    void* pipebuff = (char*)rSbuff + (totalSendCount + i * pipeOffset) * ncclTypeSize(compDatatype);
    void* intraSendCompbuff = (char*)rSbuff + i * pipeSendOffset * ncclTypeSize(compDatatype);
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
    NCCLCHECK(ncclDecompReduceComp((void*)intraRecvCompbuff, &interSendCompbuff, recvcount / pipelineDepth * nNodes, datatype,
                intraSendCount, compDatatype, &reCompSendCount, &reCompDatatype, localRanks, ncclCommOp_t::ReduceScatter_Inter, 
                pipelineStream[i]));
      // inter alltoall
    size_t interSendCount = reCompSendCount / nNodes;

    NCCLCHECK(ncclAllToAll((void*)interSendCompbuff, (void*)interRecvCompbuff, interSendCount, compDatatype, InterSubComm, pipelineStream[i]));

    void* piperecvbuff = (char*) recvbuff + (size_t)i * recvcount / pipelineDepth * ncclTypeSize(datatype);
    // DecompReduce
    NCCLCHECK(ncclDecompressReduce((void*)piperecvbuff, interRecvCompbuff, interSendCount, reCompDatatype, recvcount / pipelineDepth, datatype, nNodes,
                          ncclCommOp_t::ReduceScatter_Inter, pipelineStream[i]));
  }

  for(int i=0;i<pipelineDepth;i++){
    CUDACHECK(cudaEventRecord(pipelineEvent[i], pipelineStream[i]));  
    CUDACHECK(cudaStreamWaitEvent(stream, pipelineEvent[i], 0));
  }

  return ncclSuccess;
}
__thread cudaStream_t RScompStream_Inter = nullptr, RScommStream_Inter = nullptr, RSdecompStream_Inter = nullptr;
__thread cudaEvent_t RScompEvent_Inter = nullptr, RScommEvent_Inter = nullptr, RSdecompEvent_Inter = nullptr;

NCCL_API(ncclResult_t, ncclReduceScatterCompTwoShotTLOverlap, const void* sendbuff, void* recvbuff, size_t recvcount,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclReduceScatterCompTwoShotTLOverlap(const void* sendbuff, void* recvbuff, size_t recvcount,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream){
  pipelineDepth = ncclParamPipelineDepth();
  if(pipelineDepth < 2){
    NCCLCHECK(ncclReduceScatterCompTwoShot(sendbuff, recvbuff, recvcount, datatype, op, comm, stream));
    return ncclSuccess;
  }
  CUDACHECK(cudaSetDevice(comm->cudaDev));
  int nRanks = comm->nRanks;
  int localRanks = comm->localRanks;
  int nNodes = nRanks / localRanks;
  if(InterSubComm == nullptr || IntraSubComm == nullptr){
    pipelineDepth = ncclParamPipelineDepth();
    pipelineStream = (cudaStream_t*)malloc(sizeof(cudaStream_t) * (pipelineDepth));
    pipelineEvent = (cudaEvent_t*)malloc(sizeof(cudaEvent_t) * (pipelineDepth));
    for(int i=0; i < pipelineDepth; i++){
      CUDACHECK(cudaStreamCreateWithFlags(pipelineStream + i, cudaStreamNonBlocking));
      CUDACHECK(cudaEventCreateWithFlags(pipelineEvent + i, cudaEventDefault));
    }
    //intraSubComm
    NCCLCHECK(ncclCommSplit(comm, comm->rank / localRanks, comm->rank, &IntraSubComm, NULL));
    //interSubComm
    NCCLCHECK(ncclCommSplit(comm, comm->rank % localRanks, comm->rank, &InterSubComm, NULL));
  }

  size_t compSendCount;
  ncclDataType_t compDatatype;

  size_t totalSendBytes = 2 * (nRanks + nNodes) * recvcount * ncclTypeSize(datatype);
  if(RScompStream == nullptr){
    CUDACHECK(cudaStreamCreateWithFlags(&RScompStream, cudaStreamNonBlocking));
    CUDACHECK(cudaStreamCreateWithFlags(&RScommStream, cudaStreamNonBlocking));
    CUDACHECK(cudaStreamCreateWithFlags(&RSdecompStream, cudaStreamNonBlocking));
    CUDACHECK(cudaEventCreateWithFlags(&RScompEvent, cudaEventDefault));
    CUDACHECK(cudaEventCreateWithFlags(&RScommEvent, cudaEventDefault));
    CUDACHECK(cudaEventCreateWithFlags(&RSdecompEvent, cudaEventDefault));

    CUDACHECK(cudaStreamCreateWithFlags(&RScompStream_Inter, cudaStreamNonBlocking));
    CUDACHECK(cudaStreamCreateWithFlags(&RScommStream_Inter, cudaStreamNonBlocking));
    CUDACHECK(cudaStreamCreateWithFlags(&RSdecompStream_Inter, cudaStreamNonBlocking));
    CUDACHECK(cudaEventCreateWithFlags(&RScompEvent_Inter, cudaEventDefault));
    CUDACHECK(cudaEventCreateWithFlags(&RScommEvent_Inter, cudaEventDefault));
    CUDACHECK(cudaEventCreateWithFlags(&RSdecompEvent_Inter, cudaEventDefault));

    CUDACHECK(cudaEventCreateWithFlags(&RSmainEvent, cudaEventDefault));
  }
  NCCLCHECK(cocclBuffAlloc(&rSbuff, totalSendBytes, comm));
  

  CUDACHECK(cudaEventRecord(RSmainEvent, stream));
  CUDACHECK(cudaStreamWaitEvent(RScompStream, RSmainEvent, 0));
  void* intrabuff =(char* ) rSbuff;
  void* interbuff =(char* ) rSbuff + 2 * recvcount * nRanks * ncclTypeSize(datatype);
  size_t intraRecvOffset = recvcount * nRanks * ncclTypeSize(datatype);
  size_t interRecvOffset = recvcount * nNodes * ncclTypeSize(datatype);
  
  for(size_t i =0 ;i<pipelineDepth; i++){
    void* sbuff = (char*)sendbuff + i * recvcount / pipelineDepth * nRanks * ncclTypeSize(datatype);
    void* intraSendCompbuff = (char*) intrabuff + i * recvcount / pipelineDepth * nRanks * ncclTypeSize(datatype);

    NCCLCHECK(ncclCompress(sbuff, &intraSendCompbuff, recvcount / pipelineDepth, datatype, &compSendCount,
                              &compDatatype, nRanks, comm->rank, ncclCommOp_t::ReduceScatter_Inter, RScompStream));

    CUDACHECK(cudaEventRecord(RScompEvent, RScompStream));  
    CUDACHECK(cudaStreamWaitEvent(RScommStream, RScompEvent, 0));
    void* intraRecvCompbuff = (char*) intrabuff + intraRecvOffset + i * compSendCount * nNodes * localRanks * ncclTypeSize(compDatatype);
    size_t intraSendCount = compSendCount * nNodes;

    NCCLCHECK(ncclAllToAll((void*)intraSendCompbuff, (void*)intraRecvCompbuff, intraSendCount, compDatatype, IntraSubComm, RScommStream));
    CUDACHECK(cudaEventRecord(RScommEvent, RScommStream));  
    CUDACHECK(cudaStreamWaitEvent(RSdecompStream, RScommEvent, 0));
    // size_t interOffset = pipeCompCount * nRanks;
    // void* interSendCompbuff = (char*) pipebuff + interOffset * ncclTypeSize(compDatatype);
    // void* interRecvCompbuff = (char*) pipebuff + (interOffset + pipeCompCount * nNodes) * ncclTypeSize(compDatatype);
    void* interSendCompbuff = (char*) interbuff + i * recvcount * nNodes / pipelineDepth * ncclTypeSize(datatype);
    size_t reCompSendCount;
    ncclDataType_t reCompDatatype;
    NCCLCHECK(ncclDecompReduceComp((void*)intraRecvCompbuff, &interSendCompbuff, compSendCount * nNodes, datatype,
                intraSendCount, compDatatype, &reCompSendCount, &reCompDatatype, localRanks, ncclCommOp_t::ReduceScatter_Inter, 
                RSdecompStream));
    CUDACHECK(cudaEventRecord(RSdecompEvent, RSdecompStream));  
    CUDACHECK(cudaStreamWaitEvent(RScommStream, RSdecompEvent, 0));
    void* interRecvCompbuff = (char*) interbuff + interRecvOffset + i * reCompSendCount * ncclTypeSize(reCompDatatype);
    
    //  (interOffset + pipeCompCount * nNodes) * ncclTypeSize(compDatatype);
    size_t interSendCount = reCompSendCount / nNodes;

    NCCLCHECK(ncclAllToAll((void*)interSendCompbuff, (void*)interRecvCompbuff, interSendCount, reCompDatatype, InterSubComm, RScommStream));
    CUDACHECK(cudaEventRecord(RScommEvent_Inter, RScommStream));  
    CUDACHECK(cudaStreamWaitEvent(RSdecompStream, RScommEvent_Inter, 0));
    void* piperecvbuff = (char*) recvbuff + i * recvcount / pipelineDepth * ncclTypeSize(datatype);

    // DecompReduce
    NCCLCHECK(ncclDecompressReduce((void*)piperecvbuff, interRecvCompbuff, interSendCount, reCompDatatype, recvcount / pipelineDepth, datatype, nNodes,
                          ncclCommOp_t::ReduceScatter_Inter, RSdecompStream));
  }
  CUDACHECK(cudaEventRecord(RSdecompEvent_Inter, RSdecompStream));
  CUDACHECK(cudaStreamWaitEvent(stream, RSdecompEvent_Inter, 0));


  return ncclSuccess;
}
// max reduceScatter sendSize
__thread size_t aRMaxSendSize = 0;
__thread void* aRbuff = nullptr;
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
  bool mayUpdateBuff = aRbuff == nullptr || totalSendBytes > aRMaxSendSize;
  // NCCLCHECK(ncclCompress(sendbuff, chunkCount, datatype, &sendCompbuff, &compSendCount, &compDatatype, numChunks, stream));
  NCCLCHECK(ncclCompress(sendbuff, mayUpdateBuff ?  &recvbuff : &aRbuff, chunkCount, datatype, &compSendCount, &compDatatype, comm->nRanks, comm->rank, ncclCommOp_t::AllReduce, stream));
  
  if(mayUpdateBuff){
    aRMaxSendSize = totalSendBytes;
    size_t compBuffBytes = compSendCount * (comm->nRanks + comm->nRanks * numChunks) * ncclTypeSize(compDatatype);
    NCCLCHECK(cocclBuffAlloc(&aRbuff, compBuffBytes, comm));
    CUDACHECK(cudaMemcpy(aRbuff, recvbuff, compSendCount * comm->nRanks * ncclTypeSize(compDatatype), cudaMemcpyDeviceToDevice));
    CUDACHECK(cudaDeviceSynchronize());
  }
  void* sendCompbuff = aRbuff;
  void* recvCompbuff = (char*) aRbuff + compSendCount * comm->nRanks * ncclTypeSize(compDatatype);

  // NCCLCHECK(ncclCompress(sendbuff, &sendCompbuff, chunkCount, datatype,  &compSendCount, &compDatatype, comm->nRanks, ncclCommOp_t::AllReduce, stream));

  // CUDACHECK(cudaMallocAsync((void**)&recvCompbuff,  comm->nRanks * numChunks * compSendCount * ncclTypeSize(compDatatype), stream));

  //Gather

  // P2P based - allchunk
  // in RTX 4090 platform it is faster than broadcast based and p2p chunk parallel 50% 
  // size 1K ~ 1M
  NCCLCHECK(ncclGroupStart());

  for(int r = 0; r < comm->nRanks; r++){

    char* r_recvbuf = (char*)recvCompbuff + r * numChunks * compSendCount * ncclTypeSize(compDatatype);
    NCCLCHECK(ncclSendNaive(sendCompbuff, numChunks * compSendCount, compDatatype, r, comm, stream));
    NCCLCHECK(ncclRecvNaive((void*)r_recvbuf, numChunks * compSendCount, compDatatype, r, comm, stream));

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
  bool mayUpdateBuff = aRbuff == nullptr || totalSendBytes > aRMaxSendSize;
 
  // reuse buff may have some wrong, some data may not send/recv sometimes
  // NCCLCHECK(ncclCompress(sendbuff, &sendCompbuff, chunkCount, datatype, &compSendCount, &compDatatype, comm->nRanks, ncclCommOp_t::AllReduce, stream));
  NCCLCHECK(ncclCompress(sendbuff, mayUpdateBuff ?  &recvbuff : &aRbuff, chunkCount, datatype, 
                            &compSendCount, &compDatatype, comm->nRanks, comm->rank, ncclCommOp_t::AllReduce, stream));
  
  if(mayUpdateBuff){
    aRMaxSendSize = totalSendBytes;
    size_t compBuffBytes = 2 * compSendCount * comm->nRanks * ncclTypeSize(compDatatype);
    NCCLCHECK(cocclBuffAlloc(&aRbuff, compBuffBytes, comm));
    CUDACHECK(cudaMemcpy(aRbuff, recvbuff, compSendCount * comm->nRanks * ncclTypeSize(compDatatype), cudaMemcpyDeviceToDevice));
    CUDACHECK(cudaDeviceSynchronize());
  }
  void* sendCompbuff = aRbuff;
  void* recvCompbuff = (char*) aRbuff + compSendCount * comm->nRanks * ncclTypeSize(compDatatype);
  // CUDACHECK(cudaMallocAsync((void**)&recvCompbuff, compSendCount * comm->nRanks * ncclTypeSize(compDatatype), stream));
  //sendCompbuff + comm->nRanks * compSendCount * ncclTypeSize(ncclInt8)
  
  NCCLCHECK(ncclAllToAll((void*)sendCompbuff, (void*)recvCompbuff, compSendCount, compDatatype, comm, stream));
  size_t reCompSendCount;
  ncclDataType_t reCompDatatype;
  // DecompReduceComp
  NCCLCHECK(ncclDecompReduceComp((void*)recvCompbuff, &sendCompbuff, count / comm->nRanks, datatype,
              compSendCount, compDatatype, &reCompSendCount, &reCompDatatype, comm->nRanks, ncclCommOp_t::AllReduce, stream));

  struct ncclInfo info = { ncclFuncAllGather, "AllGather",
    sendCompbuff, recvCompbuff, reCompSendCount, reCompDatatype, ncclSum, 0, comm, stream, /* Args */
    ALLGATHER_CHUNKSTEPS, ALLGATHER_SLICESTEPS };
  NCCLCHECK(ncclEnqueueCheck(&info));

  // Decompress
  NCCLCHECK(ncclDecompress(recvbuff, (void*)recvCompbuff, chunkCount, datatype, reCompSendCount, reCompDatatype, comm->nRanks, ncclCommOp_t::AllReduce, stream));

  return ncclSuccess;
}
__thread cudaStream_t ARcompStream = nullptr, ARcommStream = nullptr, ARdecompStream = nullptr, ARAGStream = nullptr, ARAGdecompStream = nullptr;
__thread cudaEvent_t ARcompEvent = nullptr, ARcommEvent = nullptr, ARdecompEvent = nullptr, ARAGEvent = nullptr, ARAGdecompEvent = nullptr;
__thread cudaEvent_t ARmainEvent;

NCCL_API(ncclResult_t, ncclAllReduceCompTwoShotOverlap, const void* sendbuff, void* recvbuff, size_t count,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduceCompTwoShotOverlap(const void* sendbuff, void* recvbuff, size_t count,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {

  size_t recvcount = count / comm->nRanks;
  size_t compSendCount;
  ncclDataType_t compDatatype;
    
  CUDACHECK(cudaSetDevice(comm->cudaDev));
  size_t totalSendBytes = 2 * comm->nRanks * recvcount * ncclTypeSize(datatype);
  NCCLCHECK(cocclBuffAlloc(&aRbuff, totalSendBytes, comm));

  
  pipelineDepth = ncclParamPipelineDepth();

  if(pipelineDepth < 2){
    NCCLCHECK(ncclAllReduceCompTwoShot(sendbuff, recvbuff, count, datatype, op, comm, stream));
  }else{
      if(ARcompStream == nullptr){
        CUDACHECK(cudaStreamCreateWithFlags(&ARcompStream, cudaStreamNonBlocking));
        CUDACHECK(cudaStreamCreateWithFlags(&ARcommStream, cudaStreamNonBlocking));
        CUDACHECK(cudaStreamCreateWithFlags(&ARdecompStream, cudaStreamNonBlocking));
        CUDACHECK(cudaStreamCreateWithFlags(&ARAGStream, cudaStreamNonBlocking));
        CUDACHECK(cudaStreamCreateWithFlags(&ARAGdecompStream, cudaStreamNonBlocking));
        CUDACHECK(cudaEventCreateWithFlags(&ARcompEvent, cudaEventDefault));
        CUDACHECK(cudaEventCreateWithFlags(&ARcommEvent, cudaEventDefault));
        CUDACHECK(cudaEventCreateWithFlags(&ARdecompEvent, cudaEventDefault));
        CUDACHECK(cudaEventCreateWithFlags(&ARAGEvent, cudaEventDefault));
        CUDACHECK(cudaEventCreateWithFlags(&ARAGdecompEvent, cudaEventDefault));
        CUDACHECK(cudaEventCreateWithFlags(&ARmainEvent, cudaEventDefault));
      }

      CUDACHECK(cudaEventRecord(ARmainEvent, stream));
      CUDACHECK(cudaStreamWaitEvent(ARcompStream, ARmainEvent, 0));
      for(int i =0 ;i<pipelineDepth; i++){
        void* sbuff = (char*)sendbuff + i * recvcount / pipelineDepth * comm->nRanks * ncclTypeSize(datatype);
        void* sendCompbuff = (char*) aRbuff + i * recvcount / pipelineDepth * comm->nRanks  * ncclTypeSize(datatype);

        NCCLCHECK(ncclCompress(sbuff, &sendCompbuff, recvcount / pipelineDepth, datatype, &compSendCount, &compDatatype, comm->nRanks, comm->rank, ncclCommOp_t::AllReduce, ARcompStream));
        CUDACHECK(cudaEventRecord(ARcompEvent, ARcompStream));  
        CUDACHECK(cudaStreamWaitEvent(ARcommStream, ARcompEvent, 0));
        void* recvCompbuff = (char*) aRbuff + recvcount * comm->nRanks * ncclTypeSize(datatype) + i * compSendCount * comm->nRanks * ncclTypeSize(compDatatype);

        NCCLCHECK(ncclAllToAll((void*)sendCompbuff, (void*)recvCompbuff, compSendCount, compDatatype, comm, ARcommStream));
        CUDACHECK(cudaEventRecord(ARcommEvent, ARcommStream));  
        CUDACHECK(cudaStreamWaitEvent(ARdecompStream, ARcommEvent, 0));

        size_t reCompSendCount;
        ncclDataType_t reCompDatatype;
        // DecompReduceComp
        NCCLCHECK(ncclDecompReduceComp((void*)recvCompbuff, &sendCompbuff, recvcount / pipelineDepth, datatype,
              compSendCount, compDatatype, &reCompSendCount, &reCompDatatype, comm->nRanks, ncclCommOp_t::AllReduce, ARdecompStream));
        CUDACHECK(cudaEventRecord(ARdecompEvent, ARdecompStream));  
        CUDACHECK(cudaStreamWaitEvent(ARAGStream, ARdecompEvent, 0));
        struct ncclInfo info = { ncclFuncAllGather, "AllGather",
          sendCompbuff, recvCompbuff, reCompSendCount, reCompDatatype, ncclSum, 0, comm, ARAGStream, /* Args */
          ALLGATHER_CHUNKSTEPS, ALLGATHER_SLICESTEPS };
        NCCLCHECK(ncclEnqueueCheck(&info));
        CUDACHECK(cudaEventRecord(ARAGEvent, ARAGStream));  
        CUDACHECK(cudaStreamWaitEvent(ARAGdecompStream, ARAGEvent, 0));
        void* rbuff = (char*)recvbuff + i * recvcount / pipelineDepth * ncclTypeSize(datatype);
        // Decompress
        NCCLCHECK(ncclDecompress(rbuff, (void*)recvCompbuff, recvcount / pipelineDepth, datatype, reCompSendCount, reCompDatatype, comm->nRanks, ncclCommOp_t::AllReduce, ARAGdecompStream));
      }
      CUDACHECK(cudaEventRecord(ARAGdecompEvent, ARAGdecompStream));
      CUDACHECK(cudaStreamWaitEvent(stream, ARAGdecompEvent, 0));

  }
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
  NCCLCHECK(cocclBuffAlloc(&aRbuff, totalSendBytes, comm));
  NCCLCHECK(ncclCompress(sendbuff, &aRbuff, recvcount, datatype, &compSendCount, &compDatatype, nRanks, comm->rank,
      ncclCommOp_t::AllReduce_Inter, stream));

  void* intraSendCompbuff = aRbuff;
  void* intraRecvCompbuff =(char*) aRbuff + compSendCount * nRanks * ncclTypeSize(compDatatype);
  // reuse buff may have some wrong, some data may not send/recv sometimes
  // swizzle and quan
  // intra alltoall
  size_t intraSendCount = compSendCount * nNodes;
  NCCLCHECK(ncclAllToAll((void*)intraSendCompbuff, (void*)intraRecvCompbuff, intraSendCount, compDatatype, IntraSubComm, stream));
  size_t interOffset = 2 * compSendCount * nRanks;
  void* interSendCompbuff = (char*) aRbuff + interOffset * ncclTypeSize(compDatatype);
  void* interRecvCompbuff = (char*) aRbuff + (interOffset + compSendCount * nNodes) * ncclTypeSize(compDatatype);
   
  size_t reCompSendCount;
  ncclDataType_t reCompDatatype;
    // DecompReduceComp
  NCCLCHECK(ncclDecompReduceComp((void*)intraRecvCompbuff, &interSendCompbuff, recvcount * nNodes, datatype,
             intraSendCount, compDatatype, &reCompSendCount, &reCompDatatype, localRanks, ncclCommOp_t::AllReduce_Inter, stream));
    // inter alltoall
  size_t interSendCount = reCompSendCount / nNodes;

  NCCLCHECK(ncclAllToAll((void*)interSendCompbuff, (void*)interRecvCompbuff, interSendCount, compDatatype, InterSubComm, stream));
    
  // DecompReduce
  
  NCCLCHECK(ncclDecompReduceComp((void*)interRecvCompbuff, &intraSendCompbuff, recvcount, datatype,
             interSendCount, compDatatype, &reCompSendCount, &reCompDatatype, nNodes, ncclCommOp_t::AllReduce_Inter, stream));
    
  struct ncclInfo info = { ncclFuncAllGather, "AllGather",
    aRbuff, aRbuff, reCompSendCount, reCompDatatype, ncclSum, 0, comm, stream, /* Args */
    ALLGATHER_CHUNKSTEPS, ALLGATHER_SLICESTEPS };
  NCCLCHECK(ncclEnqueueCheck(&info));

  // Decompress
  NCCLCHECK(ncclDecompress(recvbuff, aRbuff, recvcount, datatype, reCompSendCount, reCompDatatype, comm->nRanks, ncclCommOp_t::AllReduce_Inter, stream));
  

  return ncclSuccess;
}
__thread cudaStream_t  ARcommStream_Inter = nullptr, ARdecompStream_Inter = nullptr;
__thread cudaEvent_t ARcommEvent_Inter = nullptr, ARdecompEvent_Inter = nullptr;


NCCL_API(ncclResult_t, ncclAllReduceCompTripleShotTLOverlap, const void* sendbuff, void* recvbuff, size_t count,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduceCompTripleShotTLOverlap(const void* sendbuff, void* recvbuff, size_t count,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream){
    size_t recvcount = DIVUP(count, comm->nRanks);

    pipelineDepth = ncclParamPipelineDepth();
    if(pipelineDepth < 2){
      NCCLCHECK(ncclAllReduceCompTripleShot(sendbuff, recvbuff, count, datatype, op, comm, stream));
    }else{
    CUDACHECK(cudaSetDevice(comm->cudaDev));
    int nRanks = comm->nRanks;
    int localRanks = comm->localRanks;
    int nNodes = nRanks / localRanks;
    if(InterSubComm == nullptr || IntraSubComm == nullptr){
      pipelineDepth = ncclParamPipelineDepth();
      pipelineStream = (cudaStream_t*)malloc(sizeof(cudaStream_t) * (pipelineDepth));
      pipelineEvent = (cudaEvent_t*)malloc(sizeof(cudaEvent_t) * (pipelineDepth));
      for(int i=0; i < pipelineDepth; i++){
        CUDACHECK(cudaStreamCreateWithFlags(pipelineStream + i, cudaStreamNonBlocking));
        CUDACHECK(cudaEventCreateWithFlags(pipelineEvent + i, cudaEventDefault));
      }
      //intraSubComm
      NCCLCHECK(ncclCommSplit(comm, comm->rank / localRanks, comm->rank, &IntraSubComm, NULL));
      //interSubComm
      NCCLCHECK(ncclCommSplit(comm, comm->rank % localRanks, comm->rank, &InterSubComm, NULL));
    }
  
    size_t compSendCount;
    ncclDataType_t compDatatype;
  
    size_t totalSendBytes = 2 * (nRanks + nNodes) * recvcount * ncclTypeSize(datatype);
    if(ARcompStream == nullptr){
      CUDACHECK(cudaStreamCreateWithFlags(&ARcompStream, cudaStreamNonBlocking));
      CUDACHECK(cudaStreamCreateWithFlags(&ARcommStream, cudaStreamNonBlocking));
      CUDACHECK(cudaStreamCreateWithFlags(&ARdecompStream, cudaStreamNonBlocking));
      CUDACHECK(cudaEventCreateWithFlags(&ARcompEvent, cudaEventDefault));
      CUDACHECK(cudaEventCreateWithFlags(&ARcommEvent, cudaEventDefault));
      CUDACHECK(cudaEventCreateWithFlags(&ARdecompEvent, cudaEventDefault));
      CUDACHECK(cudaStreamCreateWithFlags(&ARAGdecompStream, cudaStreamNonBlocking));
      CUDACHECK(cudaStreamCreateWithFlags(&ARAGStream, cudaStreamNonBlocking));

      CUDACHECK(cudaStreamCreateWithFlags(&RScompStream_Inter, cudaStreamNonBlocking));
      CUDACHECK(cudaStreamCreateWithFlags(&ARcommStream_Inter, cudaStreamNonBlocking));
      CUDACHECK(cudaStreamCreateWithFlags(&ARdecompStream_Inter, cudaStreamNonBlocking));
      CUDACHECK(cudaEventCreateWithFlags(&RScompEvent_Inter, cudaEventDefault));
      CUDACHECK(cudaEventCreateWithFlags(&ARcommEvent_Inter, cudaEventDefault));
      CUDACHECK(cudaEventCreateWithFlags(&ARdecompEvent_Inter, cudaEventDefault));
      CUDACHECK(cudaEventCreateWithFlags(&ARAGdecompEvent, cudaEventDefault));
      CUDACHECK(cudaEventCreateWithFlags(&ARAGEvent, cudaEventDefault));

      CUDACHECK(cudaEventCreateWithFlags(&ARmainEvent, cudaEventDefault));
    }
    NCCLCHECK(cocclBuffAlloc(&aRbuff, totalSendBytes, comm));
    
  
    CUDACHECK(cudaEventRecord(ARmainEvent, stream));
    CUDACHECK(cudaStreamWaitEvent(ARcompStream, ARmainEvent, 0));
    void* intrabuff =(char* ) aRbuff;
    void* interbuff =(char* ) aRbuff + 2 * recvcount * nRanks * ncclTypeSize(datatype);
    size_t intraRecvOffset = recvcount * nRanks * ncclTypeSize(datatype);
    size_t interRecvOffset = recvcount * nNodes * ncclTypeSize(datatype);
    
    for(size_t i =0 ;i<pipelineDepth; i++){
      void* sbuff = (char*)sendbuff + i * recvcount / pipelineDepth * nRanks * ncclTypeSize(datatype);
      void* intraSendCompbuff = (char*) intrabuff + i * recvcount / pipelineDepth * nRanks * ncclTypeSize(datatype);
  
      NCCLCHECK(ncclCompress(sbuff, &intraSendCompbuff, recvcount / pipelineDepth, datatype, &compSendCount,
                                &compDatatype, nRanks, comm->rank, ncclCommOp_t::AllReduce_Inter, ARcompStream));
  
      CUDACHECK(cudaEventRecord(ARcompEvent, ARcompStream));  
      CUDACHECK(cudaStreamWaitEvent(ARcommStream, ARcompEvent, 0));
      void* intraRecvCompbuff = (char*) intrabuff + intraRecvOffset + i * compSendCount * nNodes * localRanks * ncclTypeSize(compDatatype);
      size_t intraSendCount = compSendCount * nNodes;
  
      NCCLCHECK(ncclAllToAll((void*)intraSendCompbuff, (void*)intraRecvCompbuff, intraSendCount, compDatatype, IntraSubComm, ARcommStream));
      CUDACHECK(cudaEventRecord(ARcommEvent, ARcommStream));  
      CUDACHECK(cudaStreamWaitEvent(ARdecompStream, ARcommEvent, 0));
      // size_t interOffset = pipeCompCount * nRanks;
      // void* interSendCompbuff = (char*) pipebuff + interOffset * ncclTypeSize(compDatatype);
      // void* interRecvCompbuff = (char*) pipebuff + (interOffset + pipeCompCount * nNodes) * ncclTypeSize(compDatatype);
      void* interSendCompbuff = (char*) interbuff + i * recvcount * nNodes / pipelineDepth * ncclTypeSize(datatype);
      size_t reCompSendCount;
      ncclDataType_t reCompDatatype;
      NCCLCHECK(ncclDecompReduceComp((void*)intraRecvCompbuff, &interSendCompbuff, compSendCount * nNodes, datatype,
                  intraSendCount, compDatatype, &reCompSendCount, &reCompDatatype, localRanks, ncclCommOp_t::AllReduce_Inter, 
                  ARdecompStream));
      CUDACHECK(cudaEventRecord(ARdecompEvent, ARdecompStream));  
      CUDACHECK(cudaStreamWaitEvent(ARcommStream_Inter, ARdecompEvent, 0));
      void* interRecvCompbuff = (char*) interbuff + interRecvOffset + i * reCompSendCount * ncclTypeSize(reCompDatatype);
      
      //  (interOffset + pipeCompCount * nNodes) * ncclTypeSize(compDatatype);
      size_t interSendCount = reCompSendCount / nNodes;
  
      NCCLCHECK(ncclAllToAll((void*)interSendCompbuff, (void*)interRecvCompbuff, interSendCount, reCompDatatype, InterSubComm, ARcommStream_Inter));
      CUDACHECK(cudaEventRecord(ARcommEvent_Inter, ARcommStream_Inter));  
      CUDACHECK(cudaStreamWaitEvent(ARdecompStream_Inter, ARcommEvent_Inter, 0));
  
      // // DecompReduce
      // NCCLCHECK(ncclDecompressReduce((void*)piperecvbuff, interRecvCompbuff, interSendCount, reCompDatatype, recvcount / pipelineDepth, datatype, nNodes,
      //                       ncclCommOp_t::AllReduce_Inter, ARdecompStream_Inter));

      void* agSendCompbuff = (char*) intrabuff + i * interSendCount * nRanks * ncclTypeSize(datatype);
      NCCLCHECK(ncclDecompReduceComp((void*)interRecvCompbuff, &agSendCompbuff, recvcount / pipelineDepth, datatype,
            interSendCount, compDatatype, &reCompSendCount, &reCompDatatype, nNodes, ncclCommOp_t::AllReduce_Inter, ARdecompStream_Inter));
      CUDACHECK(cudaEventRecord(ARdecompEvent_Inter, ARdecompStream_Inter));  
      CUDACHECK(cudaStreamWaitEvent(ARAGStream, ARdecompEvent_Inter, 0));
            
        // NCCLCHECK(ncclDecompReduceComp((void*)interRecvCompbuff, &intraSendCompbuff, recvcount * nNodes, datatype,
        //      intraSendCount, compDatatype, &reCompSendCount, &reCompDatatype, localRanks, ncclCommOp_t::AllReduce_Inter, stream));
      struct ncclInfo info = { ncclFuncAllGather, "AllGather",
        agSendCompbuff, agSendCompbuff, reCompSendCount, reCompDatatype, ncclSum, 0, comm, ARAGStream, /* Args */
        ALLGATHER_CHUNKSTEPS, ALLGATHER_SLICESTEPS };
      NCCLCHECK(ncclEnqueueCheck(&info));
      CUDACHECK(cudaEventRecord(ARAGEvent, ARAGStream));  
      CUDACHECK(cudaStreamWaitEvent(ARAGdecompStream, ARAGEvent, 0));
      void* rbuff = (char*)recvbuff + i * recvcount / pipelineDepth * ncclTypeSize(datatype);
      NCCLCHECK(ncclDecompress(rbuff, (void*)agSendCompbuff, recvcount / pipelineDepth, datatype, reCompSendCount, reCompDatatype, comm->nRanks, ncclCommOp_t::AllReduce_Inter, ARAGdecompStream));
      // Decompress
    }
    CUDACHECK(cudaEventRecord(ARAGdecompEvent, ARAGdecompStream));
    CUDACHECK(cudaStreamWaitEvent(stream, ARAGdecompEvent, 0));
  }
  
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclAllReduceCompRing, const void* sendbuff, void* recvbuff, size_t count,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduceCompRing(const void* sendbuff, void* recvbuff, size_t count,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  
  size_t chunkCount = count / comm->nRanks;

  char* r_recvbuf = (char*) recvbuff + comm->rank * chunkCount * ncclTypeSize(datatype);

  NCCLCHECK(ncclReduceScatterComp(sendbuff, r_recvbuf, chunkCount, datatype, op, comm, stream));

  NCCLCHECK(ncclAllGatherComp(r_recvbuf, recvbuff, chunkCount, datatype, comm, stream));
  
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclAllReduceCompOptim, const void* sendbuff, void* recvbuff, size_t count,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduceCompOptim(const void* sendbuff, void* recvbuff, size_t count,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream){
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

__thread size_t srMaxSendSize = 0;
extern __thread int ncclGroupDepth;
struct compMeta_t{
  size_t compCount = 0;
  ncclDataType_t compDatatype;
};
__thread void* sendCompbuff = nullptr;

__thread void* sendBWDCompbuff = nullptr;
compMeta_t* hCompSendMeta = nullptr, *dCompSendMeta = nullptr;
compMeta_t* hCompBWDSendMeta = nullptr, *dCompBWDSendMeta = nullptr;

__thread ncclComm_t fwdComm = nullptr;
__thread ncclComm_t bwdComm = nullptr;

__thread cudaStream_t fwdStream = nullptr;
__thread cudaEvent_t fwdEvent = nullptr;

__thread cudaStream_t bwdStream = nullptr;
__thread cudaEvent_t bwdEvent = nullptr;

__thread cudaEvent_t mEvent = nullptr;

NCCL_API(ncclResult_t, ncclSendComp, const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
  ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclSendComp(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
  ncclComm_t comm, cudaStream_t stream) {
  
  ncclCommOp_t compop = comm->rank < peer ? ncclCommOp_t::SendRecv : ncclCommOp_t::SendRecv_BWD;

  compMeta_t** hMeta = comm->rank < peer ? &hCompSendMeta : &hCompBWDSendMeta;
  compMeta_t** dMeta = comm->rank < peer ? &dCompSendMeta : &dCompBWDSendMeta;

  CUDACHECK(cudaSetDevice(comm->cudaDev));
  int tempdepth = ncclGroupDepth;
  while(ncclGroupDepth > 0) NCCLCHECK(ncclGroupEnd());
 
  if(fwdComm == nullptr){
    NCCLCHECK(ncclCommSplit(comm, 0, comm->rank, &fwdComm, NULL));
    CUDACHECK(cudaStreamCreateWithFlags(&fwdStream, cudaStreamNonBlocking));
    CUDACHECK(cudaEventCreateWithFlags(&fwdEvent, cudaEventDefault));
  }
  if(bwdComm == nullptr){
    NCCLCHECK(ncclCommSplit(comm, 1, comm->rank, &bwdComm, NULL));
    CUDACHECK(cudaStreamCreateWithFlags(&bwdStream, cudaStreamNonBlocking));
    CUDACHECK(cudaEventCreateWithFlags(&bwdEvent, cudaEventDefault));
  }

  if(*hMeta == nullptr){
    cudaHostAlloc((void**)hMeta, sizeof(compMeta_t), cudaHostAllocWriteCombined);
    cudaMalloc((void**)dMeta, sizeof(compMeta_t));
  }

  INFO(NCCL_INIT, "SendComp_datatype_%d_sendbytes_%zuMB_rank_%d_peer_%d_nRanks_%d_stream_%p", datatype, count * ncclTypeSize(datatype)/ 1024 /1024, comm->rank, peer, comm->nRanks, (void*)stream);

  size_t totalSendBytes = 2 * count * ncclTypeSize(datatype);
  void** buff = comm->rank < peer ? &sendCompbuff : &sendBWDCompbuff;
  ncclComm_t sendComm = comm->rank < peer ? fwdComm: bwdComm;
  cudaStream_t sendStream = comm->rank < peer ? fwdStream: bwdStream;
  cudaEvent_t sendEvent = comm->rank < peer ? fwdEvent: bwdEvent;
  
 
  if(mEvent == nullptr)
    CUDACHECK(cudaEventCreateWithFlags(&mEvent, cudaEventDefault));
  CUDACHECK(cudaEventRecord(mEvent, stream));  
  CUDACHECK(cudaStreamWaitEvent(sendStream, mEvent, 0));
  
  NCCLCHECK(cocclBuffAlloc(buff, totalSendBytes, sendComm));

  
  NCCLCHECK(ncclCompress(sendbuff, buff, count, datatype, &((*hMeta)->compCount), &((*hMeta)->compDatatype), 1, comm->rank,
      compop, sendStream));
  CUDACHECK(cudaMemcpyAsync(*dMeta, *hMeta, sizeof(compMeta_t), cudaMemcpyHostToDevice, sendStream));
  CUDACHECK(cudaStreamSynchronize(sendStream));

  NCCLCHECK(ncclSendNaive(*dMeta, sizeof(compMeta_t), ncclDataType_t::ncclInt8, peer, sendComm, sendStream));
  
  NCCLCHECK(ncclSendNaive(*buff, (*hMeta)->compCount, (*hMeta)->compDatatype, peer, sendComm, sendStream));
  
  while(tempdepth-- > 0) NCCLCHECK(ncclGroupStart());

  return ncclSuccess;
}
__thread void* recvCompbuff = nullptr;
__thread void* recvBWDCompbuff = nullptr;

compMeta_t* hCompRecvMeta = nullptr, *dCompRecvMeta = nullptr;
compMeta_t* hCompBWDRecvMeta = nullptr, *dCompBWDRecvMeta = nullptr;

NCCL_API(ncclResult_t, ncclRecvDecomp, void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
  ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclRecvDecomp(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
  ncclComm_t comm, cudaStream_t stream) {
  
  ncclCommOp_t compop = comm->rank > peer ? ncclCommOp_t::SendRecv : ncclCommOp_t::SendRecv_BWD;

  CUDACHECK(cudaSetDevice(comm->cudaDev));
  int tempdepth = ncclGroupDepth;
  while(ncclGroupDepth > 0) NCCLCHECK(ncclGroupEnd());

  compMeta_t** hMeta = comm->rank > peer ? &hCompRecvMeta : &hCompBWDRecvMeta;
  compMeta_t** dMeta = comm->rank > peer ? &dCompRecvMeta : &dCompBWDRecvMeta;

  if(*hMeta == nullptr){
    cudaHostAlloc((void **)hMeta, sizeof(compMeta_t), cudaHostAllocDefault);
    cudaMalloc((void**)dMeta, sizeof(compMeta_t));
  }
  
  if(fwdComm == nullptr){
    NCCLCHECK(ncclCommSplit(comm, 0, comm->rank, &fwdComm, NULL));
    CUDACHECK(cudaStreamCreateWithFlags(&fwdStream, cudaStreamNonBlocking));
    CUDACHECK(cudaEventCreateWithFlags(&fwdEvent, cudaEventDefault));

  }
  if(bwdComm == nullptr){
    NCCLCHECK(ncclCommSplit(comm, 1, comm->rank, &bwdComm, NULL));
    CUDACHECK(cudaStreamCreateWithFlags(&bwdStream, cudaStreamNonBlocking));
    CUDACHECK(cudaEventCreateWithFlags(&bwdEvent, cudaEventDefault));
  }

  void** buff = comm->rank > peer ? &recvCompbuff : &recvBWDCompbuff;
  ncclComm_t recvComm = comm->rank > peer ? fwdComm: bwdComm;
  cudaStream_t recvStream = comm->rank > peer ? fwdStream: bwdStream;
  cudaEvent_t recvEvent = comm->rank > peer ? fwdEvent: bwdEvent;

  INFO(NCCL_INIT, "RecvComp_datatype_%d_recvbuff_%zuMB_rank_%d_peer_%d_nRanks_%d_stream_%p", datatype, count * ncclTypeSize(datatype)/ 1024 /1024, comm->rank, peer, comm->nRanks, (void*)stream);

  
  NCCLCHECK(ncclRecvNaive(*dMeta, sizeof(compMeta_t), ncclDataType_t::ncclInt8, peer, recvComm, recvStream));
  
  CUDACHECK(cudaMemcpyAsync(*hMeta, *dMeta, sizeof(compMeta_t), cudaMemcpyDeviceToHost, recvStream));
  CUDACHECK(cudaStreamSynchronize(recvStream));
  
 
  if((*hMeta)->compCount > 0){
    
    size_t totalSendBytes = (*hMeta)->compCount * ncclTypeSize((*hMeta)->compDatatype);
    NCCLCHECK(cocclBuffAlloc(buff, totalSendBytes, recvComm));
    

    NCCLCHECK(ncclRecvNaive(*buff, (*hMeta)->compCount, (*hMeta)->compDatatype, peer, recvComm, recvStream));

   
    NCCLCHECK(ncclDecompress(recvbuff, *buff, count, datatype, (*hMeta)->compCount, (*hMeta)->compDatatype, 
      1, compop, recvStream));
  }

  CUDACHECK(cudaEventRecord(recvEvent, recvStream));  
  CUDACHECK(cudaStreamWaitEvent(stream, recvEvent, 0));
  while(tempdepth-- > 0) NCCLCHECK(ncclGroupStart());

  return ncclSuccess;
}

