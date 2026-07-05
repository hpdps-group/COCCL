
#include "coll_extend_p2p.h"
#include "nccl.h"
#include "argcheck.h"
#include "enqueue.h"
#include "compress.h"
#include "reduce_extend.h"

// extern __thread cudaMemPool_t compMemPool;

// NCCL_API(ncclResult_t, ncclAllToAll, const void* sendbuff, void* recvbuff, size_t sendcount,
//     ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
// ncclResult_t  ncclAllToAll(const void* sendbuff, void* recvbuff, size_t sendcount,
//     ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
//   NCCLCHECK(ncclGroupStart());
//   for (size_t r = 0; r < comm->nRanks ; r++){
//     // NCCLCHECK(ncclGroupStart());
//     char* r_sendbuf =(char*) sendbuff + r * sendcount*ncclTypeSize(datatype);
//     char* r_recvbuf =(char*) recvbuff + r * sendcount*ncclTypeSize(datatype);
//     NCCLCHECK(ncclRecv((void *)r_recvbuf, sendcount, datatype, r, comm, stream));
//     NCCLCHECK(ncclSend((void *)r_sendbuf, sendcount, datatype, r, comm, stream));
//     // NCCLCHECK(ncclGroupEnd());
//   }
//   NCCLCHECK(ncclGroupEnd());
//   return ncclSuccess;
// }

NCCL_API(ncclResult_t, ncclReduceScatterRing, const void* sendbuff, void* recvbuff, size_t recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclReduceScatterRing(const void* sendbuff, void* recvbuff, size_t recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
    
    int rightRank = (comm->rank + 1) % comm->nRanks;
    int leftRank = (comm->rank - 1 + comm->nRanks) % comm->nRanks;
    size_t chunkBytes = recvcount * ncclTypeSize(datatype);
    CUDACHECK(cudaSetDevice(comm->cudaDev));
    void* recvTempbuff;
    void* sendTempbuff;
    CUDACHECK(cudaMallocAsync((void**)&recvTempbuff, comm->nRanks * chunkBytes, stream));
    CUDACHECK(cudaMallocAsync((void**)&sendTempbuff, comm->nRanks * chunkBytes, stream));
    CUDACHECK(cudaMemcpyAsync(sendTempbuff, sendbuff, comm->nRanks * chunkBytes, cudaMemcpyDeviceToDevice, stream));
  
    for (int r = comm->nRanks - 1; r >= 0; r--) {
      int sendIdx = (comm->rank + r) % comm->nRanks;
      int recvIdx = (comm->rank + (r - 1) + comm->nRanks) % comm->nRanks;
      
      char* r_sendbuf =(char*) sendTempbuff + sendIdx * chunkBytes;
      char* r_recvbuf =(char*) recvTempbuff + recvIdx * chunkBytes;
      if(r == comm->nRanks - 1){
        // Ring step 0
  
        NCCLCHECK(ncclGroupStart());
        NCCLCHECK(ncclRecv((void*)r_recvbuf, recvcount, datatype, leftRank, comm, stream));
        NCCLCHECK(ncclSend((void*)r_sendbuf, recvcount, datatype, rightRank, comm, stream));
        NCCLCHECK(ncclGroupEnd());
      } 
      else if(r > 0) {
        // Ring step 1 ~ N-2 
        // recv reduce send

        int prevRecvIdx = (recvIdx + 1) % comm->nRanks;
        // int nextRecvIdx = (recvIdx - 1 + comm->nRanks) % comm->nRanks;
        char* r_prevRecvbuf =(char*) recvTempbuff + prevRecvIdx * chunkBytes;
        NCCLCHECK(ncclReductionColl((void*)r_sendbuf, (void*)r_prevRecvbuf, 
              (void*)r_sendbuf, datatype, op, recvcount, stream));
        
        NCCLCHECK(ncclGroupStart());
        NCCLCHECK(ncclRecv((void*)r_recvbuf, recvcount, datatype, leftRank, comm, stream));
        NCCLCHECK(ncclSend((void*)r_sendbuf, recvcount, datatype, rightRank, comm, stream));
        NCCLCHECK(ncclGroupEnd());
      } 
      else {
        // Ring step N

        int prevRecvIdx = (recvIdx + 1) % comm->nRanks;
        // int nextRecvIdx = (recvIdx - 1 + comm->nRanks) % comm->nRanks;
        char* r_prevRecvbuf =(char*) recvTempbuff + prevRecvIdx * chunkBytes;
        NCCLCHECK(ncclReductionColl((void*)r_sendbuf, (void*)r_prevRecvbuf, 
              (void*)recvbuff, datatype, op, recvcount, stream));
      }
    }
  
    CUDACHECK(cudaFreeAsync(recvTempbuff, stream));
    CUDACHECK(cudaFreeAsync(sendTempbuff, stream));
  
    return ncclSuccess;
}


NCCL_API(ncclResult_t, ncclAllReduceRing, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduceRing(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
   
  size_t chunkCount = count / comm->nRanks;

  char* r_recvbuf = (char*) recvbuff + comm->rank *  chunkCount * ncclTypeSize(datatype);

  NCCLCHECK(ncclReduceScatterRing(sendbuff, r_recvbuf, chunkCount, datatype, op, comm, stream));

  NCCLCHECK(ncclAllGather(r_recvbuf, recvbuff, chunkCount, datatype, comm, stream));

  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclAllReduceOneShot, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduceOneShot(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  // NCCLCHECK(initCompMemPool(comm->cudaDev, comm->localRanks));


  void* recvTempbuff = nullptr;
  size_t chunkCount = DIVUP(count, comm->nRanks);
  size_t numChunks = comm->nRanks;
  // CUDACHECK(cudaMallocFromPoolAsync((void**)&recvTempbuff, comm->nRanks * numChunks * chunkCount * ncclTypeSize(datatype), compMemPool, stream));
  CUDACHECK(cudaMallocAsync((void**)&recvTempbuff, comm->nRanks * numChunks * chunkCount * ncclTypeSize(datatype), stream));

  //gather

  // broadcast based - allchunk 
  // NCCLCHECK(ncclGroupStart());
  // for(int r = 0; r < comm->nRanks; r++){
  //   char* r_recvbuf = (char*)recvTempbuff + r * count * ncclTypeSize(datatype);
  //   NCCLCHECK(ncclBroadcast((void*)sendbuff, (void*)r_recvbuf, count, datatype, r, comm, stream));
  // }
  // NCCLCHECK(ncclGroupEnd());

  // broadcast based - chunk parallel
  // NCCLCHECK(ncclGroupStart());
  // for(int r = 0; r < comm->nRanks; r++){
  //   for(int chunkIdx = 0; chunkIdx < numChunks; chunkIdx++){
  //     char* r_sendbuf = (char*)sendbuff + chunkIdx * chunkCount * ncclTypeSize(datatype);
  //     char* r_recvbuf = (char*)recvTempbuff + (r * comm->nRanks + chunkIdx) * chunkCount * ncclTypeSize(datatype);
  //     NCCLCHECK(ncclBroadcast((void*)r_sendbuf, (void*)r_recvbuf, chunkCount, datatype, r, comm, stream));
  //   }
  // }
  // NCCLCHECK(ncclGroupEnd());


  // P2P based - allchunk
  // in RTX 4090 platform it is faster than broadcast based and p2p chunk parallel 50% 
  // size 1K ~ 1M
  NCCLCHECK(ncclGroupStart());
  for(int r = 0; r < comm->nRanks; r++){
    char* r_recvbuf = (char*)recvTempbuff + r * count * ncclTypeSize(datatype);
    NCCLCHECK(ncclSend(sendbuff, count, datatype, r, comm, stream));
    NCCLCHECK(ncclRecv((void*)r_recvbuf, count, datatype, r, comm, stream));
  }
  NCCLCHECK(ncclGroupEnd());


  // p2p based - chunk parallel
  // NCCLCHECK(ncclGroupStart());

  // for(int chunkIdx = 0; chunkIdx < numChunks; chunkIdx++){
  //   char* r_sendbuf =  (char*)sendbuff + chunkIdx * chunkCount * ncclTypeSize(datatype);
  //     for(int r = 0; r < comm->nRanks; r++){
  //       char* r_recvbuf = (char*)recvTempbuff + (r * comm->nRanks + chunkIdx) * chunkCount * ncclTypeSize(datatype);
  //       NCCLCHECK(ncclSend((void*)r_sendbuf, chunkCount, datatype, r, comm, stream));
  //       NCCLCHECK(ncclRecv((void*)r_recvbuf, chunkCount, datatype, r, comm, stream));
  //     }
  // }
  // NCCLCHECK(ncclGroupEnd());


  // reduce chunk
  NCCLCHECK(ncclReduceChunk(recvTempbuff, numChunks * chunkCount, recvbuff, datatype, comm->nRanks, stream));

  CUDACHECK(cudaFreeAsync(recvTempbuff, stream));
  
  return ncclSuccess;
}

/*
  1. rank 0 gather all chunk
  2. rank 0 reduce all chunk
  3. rank 0 scatter all chunk
*/
NCCL_API(ncclResult_t, ncclAllReduceTwoShotR0, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduceTwoShotR0(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  

  size_t chunkCount = DIVUP(count, comm->nRanks);
  size_t numChunks = comm->nRanks;
  void* recvTempbuff = nullptr;
  if(comm->rank==0)
    CUDACHECK(cudaMallocAsync((void**)&recvTempbuff, comm->nRanks * numChunks * chunkCount * ncclTypeSize(datatype), stream));  
  
  // gather 
  // p2p base - allchunk
  // in RTX 4090 platform it is faster than p2p chunk parallel 10% 
  // size 1M ~ 32M
  NCCLCHECK(ncclGroupStart());
  NCCLCHECK(ncclSend(sendbuff, count, datatype, 0, comm, stream));
  if (comm->rank == 0) {
    for (int r=0; r<comm->nRanks; r++) {
      NCCLCHECK(ncclRecv(((char*)recvTempbuff)+r*count*ncclTypeSize(datatype), count, datatype, r, comm, stream));
    }
  }
  NCCLCHECK(ncclGroupEnd());

  // p2p base - chunk parallel
  // NCCLCHECK(ncclGroupStart());
  // for(int chunkIdx = 0; chunkIdx < numChunks; chunkIdx++){
  //     char* r_sendbuf =  (char*)sendbuff + chunkIdx * chunkCount * ncclTypeSize(datatype);
  //     NCCLCHECK(ncclSend((void*)r_sendbuf, chunkCount, datatype, 0, comm, stream));
  // }
  // if(comm->rank == 0){
  //   for(int r = 0; r < comm->nRanks; r++){
  //     for(int chunkIdx = 0; chunkIdx < numChunks; chunkIdx++){
  //       char* r_recvbuf = (char*)recvTempbuff + (r * comm->nRanks + chunkIdx) * chunkCount * ncclTypeSize(datatype);
  //       NCCLCHECK(ncclRecv((void*)r_recvbuf, chunkCount, datatype, r, comm, stream));
  //     }
  //   }
  // }
  // NCCLCHECK(ncclGroupEnd());
  
  //reduce
  if(comm->rank == 0)
    NCCLCHECK(ncclReduceChunk(recvTempbuff, numChunks * chunkCount, recvbuff, datatype, comm->nRanks, stream));

  //scatter

  // broadcast base - allchunk
  // NCCLCHECK(ncclGroupStart());
  // NCCLCHECK(ncclBroadcast(recvbuff, recvbuff, count, datatype, 0, comm, stream));
  // NCCLCHECK(ncclGroupEnd());

  // broadcast base - chunk parallel
  // in RTX 4090 platform it is faster than p2p based and all chunk broadcast 30% 
  // size 1M ~ 32M
  NCCLCHECK(ncclGroupStart());
  for(int chunkIdx = 0; chunkIdx < numChunks; chunkIdx++){
    char* r_sendbuf = (char*)recvbuff + chunkIdx * chunkCount * ncclTypeSize(datatype);
    char* r_recvbuf = (char*)recvbuff + chunkIdx * chunkCount * ncclTypeSize(datatype);
    NCCLCHECK(ncclBroadcast((void*)r_sendbuf, (void*)r_recvbuf, chunkCount, datatype, 0, comm, stream));
  }
  NCCLCHECK(ncclGroupEnd());
  

  // p2p base - allchunk
  // if(comm->rank == 0){
  //   for(int r = 1; r< comm->nRanks; r++){
  //     NCCLCHECK(ncclSend(((char*)recvbuff), count, datatype, r, comm, stream));
  //   }
  // }else{
  //   NCCLCHECK(ncclRecv(recvbuff, count, datatype, 0, comm, stream));
  // }

  // p2p base - chunk parallel
  // if(comm->rank == 0){
  //   for(int chunkIdx = 0; chunkIdx < numChunks; chunkIdx++){
  //     char* r_sendbuf =  (char*)recvbuff + chunkIdx * chunkCount * ncclTypeSize(datatype);
  //     for(int r = 1; r < comm->nRanks; r++){
  //       NCCLCHECK(ncclSend((void*)r_sendbuf, chunkCount, datatype, r, comm, stream));
  //     }
  //   }
  // }else{
  //   for(int chunkIdx = 0; chunkIdx < numChunks; chunkIdx++){
  //     char* r_recvbuf =  (char*)recvbuff + chunkIdx * chunkCount * ncclTypeSize(datatype);
  //     NCCLCHECK(ncclRecv((void*)r_recvbuf, chunkCount, datatype, 0, comm, stream));
  //   }
  // }
  
  if(comm->rank == 0)
    CUDACHECK(cudaFreeAsync(recvTempbuff, stream));
  return ncclSuccess;
}

/*
  all to all based reduce
  in RTX 4090 platform it is faster than rank0 gather all 2.2x

  1. rank i gather chunk i
  2. rank i reduce chunk i
  3. rank i broadcast chunk i
*/ 
NCCL_API(ncclResult_t, ncclAllReduceTwoShot, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduceTwoShot(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {

  size_t chunkCount = DIVUP(count, comm->nRanks);
  // size_t numChunks = comm->nRanks;
  // void* recvTempbuff = nullptr;
  // CUDACHECK(cudaMallocAsync((void**)&recvTempbuff, comm->nRanks * chunkCount * ncclTypeSize(datatype), stream));
  // NCCLCHECK(ncclAlltoAll(sendbuff, recvTempbuff, chunkCount, datatype, comm, stream));
  NCCLCHECK(ncclAllToAll(sendbuff, recvbuff, chunkCount, datatype, comm, stream));

  // reduce
  // NCCLCHECK(ncclReduceChunk(recvTempbuff, chunkCount, recvTempbuff, datatype, comm->nRanks, stream));

  NCCLCHECK(ncclReduceChunk(recvbuff, chunkCount, recvbuff, datatype, comm->nRanks, stream));

  // broadcast base
  // NCCLCHECK(ncclGroupStart());
  // for(int r = 0; r < comm->nRanks; r++){
  //   char* r_sendbuf = (char*)recvTempbuff;
  //   char* r_recvbuf = (char*)recvbuff + r * chunkCount * ncclTypeSize(datatype);
  //   NCCLCHECK(ncclBroadcast((void*)r_sendbuf, (void*)r_recvbuf, chunkCount, datatype, r, comm, stream));
  // }
  // NCCLCHECK(ncclGroupEnd());

  // p2p base - chunk 
  // in RTX 4090 platform it is faster than broadcast based 20% 
  cudaMemcpyAsync((char*)recvbuff + comm->rank * chunkCount * ncclTypeSize(datatype), recvbuff, chunkCount * ncclTypeSize(datatype),
    cudaMemcpyDeviceToDevice, stream);
  // size 1M ~ 32M
  NCCLCHECK(ncclGroupStart());
  for(int r = 0; r <  comm->nRanks; r++){
    if(r == comm->rank) continue;
    // char* r_sendbuf = (char*)recvbuff;
    char* r_sendbuf = (char*)recvbuff + comm->rank * chunkCount * ncclTypeSize(datatype);
    char* r_recvbuf = (char*)recvbuff + r * chunkCount * ncclTypeSize(datatype);
      NCCLCHECK(ncclSend((void*)r_sendbuf, chunkCount, datatype, r, comm, stream));
      NCCLCHECK(ncclRecv((void*)r_recvbuf, chunkCount, datatype, r, comm, stream));
  }
  NCCLCHECK(ncclGroupEnd());


  // CUDACHECK(cudaFreeAsync(recvTempbuff, stream));

  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclAllReduceOptim, const void* sendbuff, void* recvbuff, size_t count,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduceOptim(const void* sendbuff, void* recvbuff, size_t count,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream){
  if(count * ncclTypeSize(datatype)< (size_t)1024 * 1024){
    NCCLCHECK(ncclAllReduceOneShot(sendbuff, recvbuff, count, datatype, op, comm, stream));
  }
  else if(count * ncclTypeSize(datatype)< (size_t)1024 * 1024 * 32){
    NCCLCHECK(ncclAllReduceTwoShot(sendbuff, recvbuff, count, datatype, op, comm, stream));
  }else{
    NCCLCHECK(ncclAllReduceRing(sendbuff, recvbuff, count, datatype, op, comm, stream));
  }
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclAllGatherRing, const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclAllGatherRing(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
    int rightRank = (comm->rank + 1) % comm->nRanks;
    int leftRank = (comm->rank - 1 + comm->nRanks) % comm->nRanks;
    size_t chunkBytes = sendcount * ncclTypeSize(datatype);
    // step 0 local copy
    CUDACHECK(cudaMemcpyAsync((char*)recvbuff + comm->rank * chunkBytes, sendbuff, chunkBytes, cudaMemcpyDeviceToDevice, stream));
    // step 1 ~ N - 2  
    for(int r = 0; r < comm->nRanks - 1; r++){
        int sendIdx = (comm->rank - r + comm->nRanks) % comm->nRanks;
        int recvIdx = (comm->rank - (r + 1) + comm->nRanks) % comm->nRanks;
        char* r_sendbuf =(char*) recvbuff + sendIdx * chunkBytes;
        char* r_recvbuf =(char*) recvbuff + recvIdx * chunkBytes;
        NCCLCHECK(ncclGroupStart());
        NCCLCHECK(ncclRecv((void*)r_recvbuf, sendcount, datatype, leftRank, comm, stream));
        NCCLCHECK(ncclSend((void*)r_sendbuf, sendcount, datatype, rightRank, comm, stream));
        NCCLCHECK(ncclGroupEnd());
    }

    return ncclSuccess;
}
