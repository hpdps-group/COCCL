
#include "nccl_comp_wrapper.h"
#include "nccl.h"
#include "argcheck.h"
#include "enqueue.h"
#include "compress.h"
#include "reduce_extend.h"
#include "../graph/topo.h"
#include "coccl_alloc.h"
#include "coccl_buffer_management.h"
#include "coccl_runtime.h"
#define COMPBUFF_EXCESS_SIZE 16

namespace {

class CocclWorkspace {
 public:
  explicit CocclWorkspace(cudaStream_t stream) : stream_(stream) {}

  ~CocclWorkspace() {
    if (handle_.slice == nullptr) return;
    // The normal path releases asynchronously after the user stream has
    // joined all internal streams. On an execution error, drain submitted work
    // before returning the slice to the pool.
    (void)cudaDeviceSynchronize();
    (void)cocclReleaseBuffer(&handle_, stream_);
  }

  ncclResult_t acquire(ncclComm_t comm, size_t bytes) {
    return cocclGetBuffer(comm, bytes, stream_, &handle_);
  }

  void* ptr() const { return handle_.ptr; }

  ncclResult_t release() {
    return cocclReleaseBuffer(&handle_, stream_);
  }

 private:
  CocclWorkspace(const CocclWorkspace&);
  CocclWorkspace& operator=(const CocclWorkspace&);

  cocclBufferHandle handle_;
  cudaStream_t stream_;
};

ncclResult_t replaySend(const void* buffer, size_t count,
                        ncclDataType_t datatype, int peer,
                        ncclComm_t comm, cudaStream_t stream) {
  cocclInfo info;
  info.sendbuff = buffer;
  info.count = count;
  info.datatype = datatype;
  info.peer = peer;
  info.func = ncclFuncSend;
  info.operation = cocclOperation::SendRecv;
  info.comm = comm;
  info.stream = stream;
  return cocclReplayNativeCall(info);
}

ncclResult_t replayRecv(void* buffer, size_t count,
                        ncclDataType_t datatype, int peer,
                        ncclComm_t comm, cudaStream_t stream) {
  cocclInfo info;
  info.recvbuff = buffer;
  info.count = count;
  info.datatype = datatype;
  info.peer = peer;
  info.func = ncclFuncRecv;
  info.operation = cocclOperation::SendRecv;
  info.comm = comm;
  info.stream = stream;
  return cocclReplayNativeCall(info);
}

}  // namespace

__thread struct parComm* parcomms = nullptr;
// maxSendSize for allgather
__thread size_t aGMaxSendBytes = 0;


__thread void* aGbuff = nullptr;

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
    NCCLCHECK(replayRecv((void *)r_recvbuf, compSendCount, compDatatype, peer, comm, stream));
    NCCLCHECK(replaySend((void *)r_sendbuf, compSendCount, compDatatype, peer, comm, stream));
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
      NCCLCHECK(replaySend((void *)r_sendbuf, compSendCount, compDatatype, peer, comm, stream));
      NCCLCHECK(replayRecv((void *)r_recvbuf, compSendCount, compDatatype, peer, comm, stream));
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
  NCCLCHECK(ncclDecompressReduce((void*)recvbuff, (void*)recvCompbuff, compSendCount, compDatatype, recvcount, datatype, comm->nRanks, comm->nRanks,
                        ncclCommOp_t::ReduceScatter, stream, comm));
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
      NCCLCHECK(replayRecv((void*)reduceRecvbuf, compSendCount, compDatatype, leftRank, comm, stream));
      NCCLCHECK(replaySend((void*)reduceSendbuf, compSendCount, compDatatype, rightRank, comm, stream));
      NCCLCHECK(ncclGroupEnd());

    } else if(r > 0) {
      // Ring step 1 ~ N - 2
      CUDACHECK(cudaMemcpyAsync(reducebuff, reduceSendbuf, compSendCount * ncclTypeSize(compDatatype), 
          cudaMemcpyDeviceToDevice, stream)); 
      size_t reCompChunkCount;
      ncclDataType_t reCompDatatype;
      // DecompReduceComp
      NCCLCHECK(ncclDecompReduceComp((void*)reducebuff, (void**)&reduceSendbuf, recvcount, datatype, 
                  compSendCount, compDatatype, &reCompChunkCount, &reCompDatatype, 2, 2, ncclCommOp_t::ReduceScatter, stream, comm));

      NCCLCHECK(ncclGroupStart());
      NCCLCHECK(replayRecv((void*)reduceRecvbuf, reCompChunkCount, reCompDatatype, leftRank, comm, stream));
      NCCLCHECK(replaySend((void*)reduceSendbuf, reCompChunkCount, reCompDatatype, rightRank, comm, stream));
      NCCLCHECK(ncclGroupEnd());
    } else {
      // Ring step N - 1
        CUDACHECK(cudaMemcpyAsync(reducebuff, reduceSendbuf, compSendCount * ncclTypeSize(compDatatype), 
            cudaMemcpyDeviceToDevice, stream)); 
      // decompress - reduce
      NCCLCHECK(ncclDecompressReduce((void*)recvbuff, reducebuff, compSendCount, compDatatype, recvcount, datatype, 2, 2,
                        ncclCommOp_t::ReduceScatter, stream, comm));
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
             intraSendCount, compDatatype, &reCompSendCount, &reCompDatatype, localRanks, localRanks, ncclCommOp_t::ReduceScatter_Inter, stream, comm));
    // inter alltoall
  size_t interSendCount = reCompSendCount / nNodes;

  NCCLCHECK(ncclAllToAll((void*)interSendCompbuff, (void*)interRecvCompbuff, interSendCount, compDatatype, InterSubComm, stream));
    
  // DecompReduce
  NCCLCHECK(ncclDecompressReduce((void*)recvbuff, interRecvCompbuff, interSendCount, reCompDatatype, recvcount, datatype, nNodes, nNodes,
                        ncclCommOp_t::ReduceScatter_Inter, stream, comm));

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
  // NCCLCHECK(ncclCompress(sendbuff, chunkCount, datatype, &sendCompbuff, &compSendCount, &compDatatype, numChunks, stream));
  void* initialOutput = recvbuff;
  NCCLCHECK(ncclCompress(sendbuff, &initialOutput, chunkCount, datatype,
                         &compSendCount, &compDatatype, comm->nRanks,
                         comm->rank, ncclCommOp_t::AllReduce, stream));
  size_t compBuffBytes = compSendCount *
      (comm->nRanks + comm->nRanks * numChunks) *
      ncclTypeSize(compDatatype);
  CocclWorkspace workspace(stream);
  NCCLCHECK(workspace.acquire(comm, compBuffBytes));
  void* aRbuff = workspace.ptr();
  CUDACHECK(cudaMemcpy(aRbuff, recvbuff,
                       compSendCount * comm->nRanks *
                           ncclTypeSize(compDatatype),
                       cudaMemcpyDeviceToDevice));
  CUDACHECK(cudaDeviceSynchronize());
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
    NCCLCHECK(replaySend(sendCompbuff, numChunks * compSendCount, compDatatype, r, comm, stream));
    NCCLCHECK(replayRecv((void*)r_recvbuf, numChunks * compSendCount, compDatatype, r, comm, stream));

  }

  NCCLCHECK(ncclGroupEnd());



  NCCLCHECK(ncclDecompressReduce((void*)recvbuff, recvCompbuff, numChunks * compSendCount, compDatatype, numChunks * chunkCount, datatype, comm->nRanks, comm->nRanks,
  ncclCommOp_t::AllReduce, stream, comm));
  // NCCLCHECK(ncclDecompress(recvTempbuff, (void*)recvCompbuff, numChunks * chunkCount, datatype, compSendCount, compDatatype, comm->nRanks, ncclCommOp_t::AllReduce, stream));
  // // Reduce chunk
  // NCCLCHECK(ncclReduceChunk(recvTempbuff, numChunks * chunkCount, recvbuff, datatype, comm->nRanks, stream));
  
  // CUDACHECK(cudaFreeAsync(sendCompbuff,stream));
  // CUDACHECK(cudaFreeAsync(recvCompbuff,stream));


  return workspace.release();
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
              compSendCount, compDatatype, &reCompSendCount, &reCompDatatype, comm->nRanks, comm->nRanks, ncclCommOp_t::AllReduce, stream, comm));

  struct ncclInfo info = { ncclFuncAllGather, "AllGather",
    sendCompbuff, recvCompbuff, reCompSendCount, reCompDatatype, ncclSum, 0, comm, stream, /* Args */
    ALLGATHER_CHUNKSTEPS, ALLGATHER_SLICESTEPS };
  NCCLCHECK(ncclEnqueueCheck(&info));

  // Decompress
  NCCLCHECK(ncclDecompress(recvbuff, (void*)recvCompbuff, chunkCount, datatype, reCompSendCount, reCompDatatype, comm->nRanks, ncclCommOp_t::AllReduce, stream));

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
  CocclWorkspace workspace(stream);
  NCCLCHECK(workspace.acquire(comm, totalSendBytes));
  void* aRbuff = workspace.ptr();
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
             intraSendCount, compDatatype, &reCompSendCount, &reCompDatatype, localRanks, localRanks, ncclCommOp_t::AllReduce_Inter, stream, comm));
    // inter alltoall
  size_t interSendCount = reCompSendCount / nNodes;

  NCCLCHECK(ncclAllToAll((void*)interSendCompbuff, (void*)interRecvCompbuff, interSendCount, compDatatype, InterSubComm, stream));
    
  // DecompReduce
  
  NCCLCHECK(ncclDecompReduceComp((void*)interRecvCompbuff, &intraSendCompbuff, recvcount, datatype,
             interSendCount, compDatatype, &reCompSendCount, &reCompDatatype, nNodes, nNodes, ncclCommOp_t::AllReduce_Inter, stream, comm));
    
  struct ncclInfo info = { ncclFuncAllGather, "AllGather",
    aRbuff, aRbuff, reCompSendCount, reCompDatatype, ncclSum, 0, comm, stream, /* Args */
    ALLGATHER_CHUNKSTEPS, ALLGATHER_SLICESTEPS };
  NCCLCHECK(ncclEnqueueCheck(&info));

  // Decompress
  NCCLCHECK(ncclDecompress(recvbuff, aRbuff, recvcount, datatype, reCompSendCount, reCompDatatype, comm->nRanks, ncclCommOp_t::AllReduce_Inter, stream));
  

  return workspace.release();
}
