#include "coccl_wrapper.h"

// max alltoall sendSize
__thread size_t a2AMaxSendSize = 0;
extern __thread void* compBuff;

NCCL_API(ncclResult_t, ncclAllToAllComp, const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
ncclResult_t  ncclAllToAllComp(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  
  // Compress
  size_t compSendCount;
  ncclDataType_t compDatatype;
  CUDACHECK(cudaSetDevice(comm->cudaDev));

  size_t totalSendBytes = 2 * comm->nRanks * sendcount * ncclTypeSize(datatype);
  bool mayUpdateBuff = compBuffHandle == nullptr || totalSendBytes > a2AMaxSendSize;
  // reuse buff may have some wrong, some data may not send/recv sometimes
  // NCCLCHECK(ncclCompress(sendbuff, &sendCompbuff, sendcount, datatype, &compSendCount, &compDatatype, comm->nRanks, ncclCommOp_t::AlltoAll, stream));
  NCCLCHECK(ncclCompress(sendbuff, mayUpdateBuff ?  &recvbuff : &compBuff, sendcount, datatype, &compSendCount, &compDatatype, comm->nRanks, ncclCommOp_t::AlltoAll, stream));
  CUDACHECK(cudaDeviceSynchronize());

  if(mayUpdateBuff){
    a2AMaxSendSize = totalSendBytes;
    size_t compBuffBytes = 2 * (compSendCount * comm->nRanks * ncclTypeSize(compDatatype));
    allocAndRegCompBuff(comm, compBuffBytes);
    CUDACHECK(cudaMemcpy(compBuff, recvbuff, compSendCount * comm->nRanks * ncclTypeSize(compDatatype), cudaMemcpyDeviceToDevice));
    CUDACHECK(cudaDeviceSynchronize());
  }
  void* sendCompbuff = compBuff;
  void* recvCompbuff = (char*)compBuff + compSendCount * comm->nRanks * ncclTypeSize(compDatatype);
  
  NCCLCHECK(ncclAllToAll((void*)sendCompbuff, (void*)recvCompbuff, compSendCount, compDatatype, comm, stream));

  NCCLCHECK(ncclDecompress(recvbuff, (char*)recvCompbuff, sendcount, datatype, compSendCount, compDatatype, comm->nRanks, ncclCommOp_t::AlltoAll, stream));
 
  return ncclSuccess;
}
// TODO comm- and comp- overlap
NCCL_API(ncclResult_t, ncclAlltoAllCompMultiStream, const void* sendbuff, void* recvbuff, size_t sendcount,
  ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
ncclResult_t  ncclAlltoAllCompMultiStream(const void* sendbuff, void* recvbuff, size_t sendcount,
  ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {

  // Compress
  size_t compSendCount;
  ncclDataType_t compDatatype;
  CUDACHECK(cudaSetDevice(comm->cudaDev));

  // if(allIntraRank ==nullptr){
  //   allIntraRank = (int*)malloc(comm->localRanks * sizeof(int));
  //   allInterRank = (int*)malloc(comm->nNodes * sizeof(int));
  //   int interCnt = 0, intraCnt = 0;
  //   for(int r = 0; r < comm->nRanks; r++){
  //     if(comm->rank % 4 == r % 4) allInterRank[interCnt++] = r;
  //     if(comm->rank / 4 == r / 4) allIntraRank[intraCnt++] = r;
  //   }
  // }

  NCCLCHECK(initParallelComms(comm));

  void** sendCompbuff;
  void** recvCompbuff;
  // sendCompbuff=(void**)calloc(comm->nRanks, sizeof(void*));
  // recvCompbuff=(void**)calloc(comm->nRanks, sizeof(void*));
  sendCompbuff=(void**)calloc(2, sizeof(void*));
  recvCompbuff=(void**)calloc(2, sizeof(void*));
  // for(int commId =0; commId < 2;commId++){
  //   NCCLCHECK(ncclCompress((char*)sendbuff + commId * comm->nRanks/2 * sendcount * ncclTypeSize(datatype), sendcount, datatype, &sendCompbuff[commId], &compSendCount, &compDatatype, comm->nRanks/2, parcomms[commId].stream));
  //   CUDACHECK(cudaMallocAsync(&recvCompbuff[commId], compSendCount * comm->nRanks/2 * ncclTypeSize(compDatatype), parcomms[commId].stream));
  // }


  for(int commId =0; commId < 2;commId++){
    // for (int r= comm->nRanks/2 *commId;r<comm->nRanks/2*(commId + 1); r++){
    //   int sendIdx = (comm->rank + r)%comm->nRanks;
    //   int recvIdx = (comm->rank - r + comm->nRanks)%comm->nRanks;
    //   NCCLCHECK(ncclCompress((char*)sendbuff + sendIdx * sendcount * ncclTypeSize(datatype), sendcount, datatype, &sendCompbuff[sendIdx], &compSendCount, &compDatatype, 1, parcomms[commId].stream));
    //   CUDACHECK(cudaMallocAsync(&recvCompbuff[recvIdx], compSendCount * 1 * ncclTypeSize(compDatatype), parcomms[commId].stream));
    // }
    NCCLCHECK(ncclCompress((char*)sendbuff + commId * comm->nRanks/2 * sendcount * ncclTypeSize(datatype), &sendCompbuff[commId], 
              sendcount, datatype, &compSendCount, &compDatatype, comm->nRanks/2, ncclCommOp_t::AlltoAll, parcomms[commId].stream));

    // NCCLCHECK(ncclCompress((char*)sendbuff + r * sendcount * ncclTypeSize(datatype), &sendCompbuff[r], sendcount, datatype, &compSendCount, &compDatatype, 1, compstreams[r]));

    CUDACHECK(cudaMallocAsync(&recvCompbuff[commId], compSendCount * comm->nRanks/2 * ncclTypeSize(compDatatype), parcomms[commId].stream));
  }
  //   for(int r= comm->nRanks/2 *commId;r<comm->nRanks/2*(commId + 1); r++){
  //     int sendIdx = (comm->rank + r)%comm->nRanks;
  //     int recvIdx = (comm->rank - r + comm->nRanks)%comm->nRanks;
  //     NCCLCHECK(ncclCompress((char*)sendbuff + sendIdx * sendcount * ncclTypeSize(datatype), sendcount, datatype, &sendCompbuff[sendIdx], &compSendCount, &compDatatype, 1, parcomms[r].stream));
  //     CUDACHECK(cudaMallocAsync(&recvCompbuff[recvIdx], compSendCount * 1 * ncclTypeSize(compDatatype), parcomms[r].stream));
  //   }
  // }
  // NCCLCHECK(ncclGroupStart());
  // for(int commId =0; commId < 2;commId++){
  //   for (int r= comm->nRanks/2 *commId;r<comm->nRanks/2*(commId + 1); r++){
  //     int sendIdx = (comm->rank + r)%comm->nRanks;
  //     int recvIdx = (comm->rank - r + comm->nRanks)%comm->nRanks;
      
  //     // char* r_sendbuf =sendCompbuff[sendIdx];
  //     // char* r_recvbuf =recvCompbuff[recvIdx];
  //     char* r_sendbuf =(char*) sendbuff + sendIdx * sendcount * ncclTypeSize(datatype);
  //     char* r_recvbuf =(char*) recvbuff + recvIdx * sendcount * ncclTypeSize(datatype);

  //     // NCCLCHECK(ncclSend((void *) r_sendbuf, compSendCount, compDatatype, sendIdx, parcomms[r].subcomm, parcomms[r].stream));
  //     // NCCLCHECK(ncclRecv((void *) r_recvbuf, compSendCount, compDatatype, recvIdx, parcomms[r].subcomm, parcomms[r].stream));
  //     NCCLCHECK(ncclSend((void *) r_sendbuf, sendcount, datatype, sendIdx, parcomms[commId].subcomm, parcomms[commId].stream));
  //     NCCLCHECK(ncclRecv((void *) r_recvbuf, sendcount, datatype, recvIdx, parcomms[commId].subcomm, parcomms[commId].stream));
  //   }
  // }
  // NCCLCHECK(ncclGroupEnd());
  NCCLCHECK(ncclGroupStart());
  for(int commId =0; commId < 2;commId++){
    for (int r= comm->nRanks/2 *commId;r<comm->nRanks/2*(commId + 1); r++){
          int sendIdx = (comm->rank + r)%comm->nRanks;
          int recvIdx = (comm->rank - r + comm->nRanks)%comm->nRanks;
      
      // char* r_sendbuf =(char*) sendCompbuff[sendIdx];
      // char* r_recvbuf =(char*) recvCompbuff[recvIdx];
      char* r_sendbuf =(char*) sendCompbuff[commId] + r %( comm->nRanks/2 )* compSendCount * ncclTypeSize(compDatatype);
      char* r_recvbuf =(char*) recvCompbuff[commId] + r %( comm->nRanks/2 ) * compSendCount * ncclTypeSize(compDatatype);

      // NCCLCHECK(ncclSend((void *) r_sendbuf, compSendCount, compDatatype, sendIdx, parcomms[r].subcomm, parcomms[r].stream));
      // NCCLCHECK(ncclRecv((void *) r_recvbuf, compSendCount, compDatatype, recvIdx, parcomms[r].subcomm, parcomms[r].stream));
      NCCLCHECK(ncclSend((void *) r_sendbuf, compSendCount, compDatatype, sendIdx, parcomms[commId].subcomm, parcomms[commId].stream));
      NCCLCHECK(ncclRecv((void *) r_recvbuf, compSendCount, compDatatype, recvIdx, parcomms[commId].subcomm, parcomms[commId].stream));
    }
  }
  NCCLCHECK(ncclGroupEnd());

   for(int commId = 0; commId < 2; commId++){
    // for (int r= comm->nRanks/2 *commId;r<comm->nRanks/2*(commId + 1); r++){
    //   int sendIdx = (comm->rank + r)%comm->nRanks;
    //   int recvIdx = (comm->rank - r + comm->nRanks)%comm->nRanks;
    //   NCCLCHECK(ncclDecompress((void*)recvCompbuff[recvIdx], compSendCount, compDatatype, (char*)recvbuff + recvIdx * sendcount * ncclTypeSize(datatype), sendcount, datatype, 1, parcomms[commId].stream));
    //   CUDACHECK(cudaFreeAsync(sendCompbuff[sendIdx], parcomms[commId].stream));
    //   CUDACHECK(cudaFreeAsync(recvCompbuff[recvIdx], parcomms[commId].stream));
    // }
    // NCCLCHECK(ncclDecompress(recvbuff, recvCompbuff, sendcount, datatype, compSendCount, compDatatype, comm->nRanks, stream));

    NCCLCHECK(ncclDecompress((char*)recvbuff + commId * comm->nRanks/2 * sendcount * ncclTypeSize(datatype), (void*)recvCompbuff[commId], 
              sendcount, datatype, compSendCount, compDatatype,  comm->nRanks/2, ncclCommOp_t::AlltoAll, parcomms[commId].stream));
    CUDACHECK(cudaFreeAsync(sendCompbuff[commId], parcomms[commId].stream));
    CUDACHECK(cudaFreeAsync(recvCompbuff[commId], parcomms[commId].stream));
  }
  
  // Decompress
  // new algo
  // for(int r=0;r<comm->nRanks;r++){
  //   int sendIdx = (comm->rank + r)%comm->nRanks;
  //   int recvIdx = (comm->rank - r + comm->nRanks)%comm->nRanks;
  //   NCCLCHECK(ncclDecompress((void*)recvCompbuff[recvIdx], compSendCount, compDatatype, (char*)recvbuff + recvIdx * sendcount * ncclTypeSize(datatype), sendcount, datatype, 1, parcomms[r].stream));
  //   CUDACHECK(cudaFreeAsync(sendCompbuff[sendIdx], parcomms[r].stream));
  //   CUDACHECK(cudaFreeAsync(recvCompbuff[recvIdx], parcomms[r].stream));
  // }
  
  for(int commId=0;commId<2;commId++){
    CUDACHECK(cudaEventRecord(parcomms[commId].event,parcomms[commId].stream));
    CUDACHECK(cudaStreamWaitEvent(stream, parcomms[commId].event, 0));
    // cudaStreamSynchronize(parcomms[r].stream);
  }


  // free(sendCompbuff);
  // free(recvCompbuff);


  // // old algo
  // for(int r=0;r<comm->nRanks;r++){
  //   NCCLCHECK(ncclCompress((char*)sendbuff + r * sendcount * ncclTypeSize(datatype), sendcount, datatype, &sendCompbuff[r], &compSendCount, &compDatatype, 1, streams[r]));
  //   CUDACHECK(cudaMallocAsync(&recvCompbuff[r], compSendCount * 1 * ncclTypeSize(compDatatype), streams[comm->rank]));
  //   // NCCLCHECK(ncclDecompress((void*)sendCompbuff[i], compSendCount, compDatatype, (char*)recvbuff + i * sendcount * ncclTypeSize(datatype), sendcount, datatype, 1, streams[i]));
  // }
  // // old algo
  // NCCLCHECK(ncclGroupStart());
  // for (size_t r = 0; r < comm->nRanks ; r++){
  //   char* r_sendbuf =(char*) sendCompbuff[r];
  //   char* r_recvbuf =(char*) recvCompbuff[r];
  //   NCCLCHECK(ncclSend((void *)r_sendbuf, compSendCount, compDatatype, r, subcomms[r], streams[r]));
  //   NCCLCHECK(ncclRecv((void *)r_recvbuf, compSendCount, compDatatype, r, subcomms[comm->rank], streams[comm->rank]));
  // }
  // NCCLCHECK(ncclGroupEnd());

  // // old algo
  // for(int r=0;r<comm->nRanks;r++){
  //   int sendIdx = (comm->rank + r)%comm->nRanks;
  //   int recvIdx = (comm->rank - r + comm->nRanks)%comm->nRanks;
  //   // NCCLCHECK(ncclCompress((char*)sendbuff + i * sendcount * ncclTypeSize(datatype), sendcount, datatype, &sendCompbuff[i], &compSendCount, &compDatatype,  1, streams[i]));
  //   NCCLCHECK(ncclDecompress((void*)recvCompbuff[r], compSendCount, compDatatype, (char*)recvbuff + r * sendcount * ncclTypeSize(datatype), sendcount, datatype, 1, streams[comm->rank]));
  //   CUDACHECK(cudaFreeAsync(sendCompbuff[sendIdx], streams[r]));
  //   CUDACHECK(cudaFreeAsync(recvCompbuff[recvIdx], streams[comm->rank]));
  // }
  // for(int r=0;r<4;r++){
  //   // CUDACHECK(cudaStreamDestroy(streams+i));
  //   // CUDACHECK(cudaStreamDestroy(streams+i));
  //   CUDACHECK(cudaEventRecord(parcomms[r].event,parcomms[r].stream));
  //   CUDACHECK(cudaStreamWaitEvent(stream, parcomms[r].event, 0));
  // }


  // two rank tests
    // NCCLCHECK(ncclGroupStart());
    // for(int r = 0; r< 4;r++){
    //   int sendIdx = (comm->rank + r)%4;
    //   int recvIdx = (comm->rank - r + 4)%4;

    //   // char* r_sendbuf =(char*) sendbuff + sendIdx * sendcount * ncclTypeSize(datatype);
    //   // char* r_recvbuf =(char*) recvbuff + recvIdx * sendcount * ncclTypeSize(datatype);
    //         // NCCLCHECK(ncclSend((void *)r_sendbuf, compSendCount, compDatatype, r, subcomms[r], streams[r]));
    //         // NCCLCHECK(ncclRecv((void *)r_recvbuf, compSendCount, compDatatype, r, subcomms[comm->rank], streams[comm->rank]));
    //         // NCCLCHECK(ncclGroupStart());
    //     char* r_sendbuf =(char*) sendCompbuff[sendIdx];
    //     char* r_recvbuf =(char*) recvCompbuff[recvIdx];

    //   // multi
    //   // NCCLCHECK(ncclSend((void *) r_sendbuf, compSendCount, compDatatype, sendIdx, parcomms[r].subcomm, parcomms[r].stream));
    //   // NCCLCHECK(ncclRecv((void *) r_recvbuf, compSendCount, compDatatype, recvIdx, parcomms[r].subcomm, parcomms[r].stream));
    // }
    // NCCLCHECK(ncclGroupEnd());

  // cudaEvent_t mainevent;
  // CUDACHECK(cudaEventCreateWithFlags(&mainevent, cudaEventDefault));
  // CUDACHECK(cudaEventRecord(mainevent,stream));

  // for(int r=0;r<comm->nRanks;r++){
  //   // CUDACHECK(cudaFree(sendCompbuff+i))
  //   CUDACHECK(cudaStreamWaitEvent(parcomms[r].stream, mainevent, 0));
  // }



  // for(int r=0;r<comm->nRanks;r++){
  //   int sendIdx = (comm->rank + r)%comm->nRanks;
  //   int recvIdx = (comm->rank - r + comm->nRanks)%comm->nRanks;
  //   // NCCLCHECK(ncclCompress((char*)sendbuff + i * sendcount * ncclTypeSize(datatype), sendcount, datatype, &sendCompbuff[i], &compSendCount, &compDatatype,  1, streams[i]));
  //   NCCLCHECK(ncclDecompress((void*)recvCompbuff[recvIdx], compSendCount, compDatatype, (char*)recvbuff + recvIdx * sendcount * ncclTypeSize(datatype), sendcount, datatype, 1, parcomms[r].stream));
  //   CUDACHECK(cudaFreeAsync(sendCompbuff[sendIdx], parcomms[r].stream));
  //   CUDACHECK(cudaFreeAsync(recvCompbuff[recvIdx], parcomms[r].stream));
  // }


    // NCCLCHECK(ncclGroupStart());
    // for(int r = 0; r< 4;r++){

    //   char* r_sendbuf =(char*) sendbuff + r * sendcount * ncclTypeSize(datatype);
    //   char* r_recvbuf =(char*) recvbuff + r * sendcount * ncclTypeSize(datatype);

    //   // multi
    //   NCCLCHECK(ncclSend((void *) r_sendbuf, sendcount, datatype, r, parcomms[comm->rank].subcomm, parcomms[comm->rank].stream));
    //   NCCLCHECK(ncclRecv((void *) r_recvbuf, sendcount, datatype, r, parcomms[r].subcomm, parcomms[r].stream));
    
    // }
    // NCCLCHECK(ncclGroupEnd());



  // for(int i=0;i<comm->nRanks;i++){
  //   // CUDACHECK(cudaFree(sendCompbuff+i))
  //   CUDACHECK(cudaEventRecord(events[i],streams[i]));
  //   CUDACHECK(cudaStreamWaitEvent(stream, events[i], 0));
  // }


  // NCCLCHECK(ncclGroupStart());
  //   for (size_t r = 0; r < comm->nRanks ; r++){
  //     // char* r_sendbuf =(char*) sendbuff + r * sendcount*ncclTypeSize(datatype);
  //     // char* r_recvbuf =(char*) recvbuff + r * sendcount*ncclTypeSize(datatype);
  //     char* r_sendbuf =(char*) sendCompbuff[r];
  //     char* r_recvbuf =(char*) sendCompbuff[r];
  //       // NCCLCHECK(ncclGroupStart());
  //       // NCCLCHECK(ncclSend((void *)r_sendbuf, compSendCount, compDatatype, r, subcomms[r], streams[r]));
  //       // NCCLCHECK(ncclRecv((void *)r_recvbuf, compSendCount, compDatatype, r, subcomms[r], streams[r]));
  //       NCCLCHECK(ncclSend((void *)r_sendbuf, compSendCount, compDatatype, r, subcomms[comm->rank * comm->nRanks + r], streams[r]));
  //       NCCLCHECK(ncclRecv((void *)r_recvbuf, compSendCount, compDatatype, r, subcomms[comm->rank * comm->nRanks + r], streams[r]));
  //       // NCCLCHECK(ncclGroupEnd());
  //       // CUDACHECK(cudaEventRecord(events[2 * r],streams[2 * r]));
  //       // CUDACHECK(cudaEventRecord(events[2 * r + 1],streams[2 * r + 1]));
  //       // CUDACHECK(cudaStreamWaitEvent(stream, events[2 * r], 0));
  //       // CUDACHECK(cudaStreamWaitEvent(stream, events[2 * r + 1], 0));
  //   }
  //   NCCLCHECK(ncclGroupEnd());


    // for(int r=0;r<comm->nRanks;r++){
    //   // CUDACHECK(cudaStreamDestroy(streams+i));
    //   // CUDACHECK(cudaStreamDestroy(streams+i));
    //   CUDACHECK(cudaEventRecord(events[r],streams[r]));
    //   CUDACHECK(cudaStreamWaitEvent(stream, events[r], 0));
    // }





  // NCCLCHECK(ncclDecompress((void*)sendCompbuff, compSendCount, compDatatype, recvbuff, sendcount, datatype, comm->nRanks, stream));

  // Free
  // for(int r=0;r<4;r++){
  //   CUDACHECK(cudaStreamSynchronize(parcomms[r].stream));
  // }

  // for(int r=0;r<comm->nRanks;r++){
  //     // CUDACHECK(cudaStreamDestroy(streams+i));
  //     // CUDACHECK(cudaStreamDestroy(streams+i));
  //     CUDACHECK(cudaEventRecord(parcomms[r].event,parcomms[r].stream));
  //     CUDACHECK(cudaStreamWaitEvent(stream, parcomms[r].event, 0));
  // }

  return ncclSuccess;
}
