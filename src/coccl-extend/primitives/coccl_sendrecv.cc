#include "coccl_primitives_internal.h"
#include "coccl_training_assist.h"
#include "group.h"

// Keep the original Send/Recv structure local to this primitive: two
// thread-local split communicators provide independent forward/backward lanes.

struct compMeta_t {
  size_t compCount = 0;
  ncclDataType_t compDatatype = ncclInt8;
};

__thread ncclComm_t fwdComm = nullptr;
__thread ncclComm_t bwdComm = nullptr;

__thread cudaStream_t fwdStream = nullptr;
__thread cudaEvent_t fwdEvent = nullptr;

__thread cudaStream_t bwdStream = nullptr;
__thread cudaEvent_t bwdEvent = nullptr;

__thread cudaEvent_t mEvent = nullptr;

__thread compMeta_t* hCompSendMeta = nullptr;
__thread compMeta_t* dCompSendMeta = nullptr;
__thread compMeta_t* hCompBWDSendMeta = nullptr;
__thread compMeta_t* dCompBWDSendMeta = nullptr;

__thread compMeta_t* hCompRecvMeta = nullptr;
__thread compMeta_t* dCompRecvMeta = nullptr;
__thread compMeta_t* hCompBWDRecvMeta = nullptr;
__thread compMeta_t* dCompBWDRecvMeta = nullptr;

static ncclResult_t cocclEnsureSendRecvLane(ncclComm_t comm, bool forward) {
  if (comm == nullptr) return ncclInvalidArgument;

  ncclComm_t* laneComm = forward ? &fwdComm : &bwdComm;
  cudaStream_t* laneStream = forward ? &fwdStream : &bwdStream;
  cudaEvent_t* laneEvent = forward ? &fwdEvent : &bwdEvent;
  int color = forward ? 0 : 1;

  if (*laneComm == nullptr) {
    NCCLCHECK(ncclCommSplit(comm, color, comm->rank, laneComm, NULL));
    cocclTrainingAssistUnregister(*laneComm);
    CUDACHECK(cudaStreamCreateWithFlags(laneStream, cudaStreamNonBlocking));
    CUDACHECK(cudaEventCreateWithFlags(laneEvent, cudaEventDefault));
  }

  return ncclSuccess;
}

static ncclResult_t cocclEnsureSendRecvMeta(compMeta_t** hostMeta, compMeta_t** deviceMeta,
                                            unsigned int hostFlags) {
  if (hostMeta == nullptr || deviceMeta == nullptr) return ncclInvalidArgument;

  if (*hostMeta == nullptr) {
    CUDACHECK(cudaHostAlloc((void**)hostMeta, sizeof(compMeta_t), hostFlags));
    CUDACHECK(cudaMalloc((void**)deviceMeta, sizeof(compMeta_t)));
  }

  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclSendComp, const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
  ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclSendComp(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
  ncclComm_t comm, cudaStream_t stream) {
  if (comm == nullptr) return ncclInvalidArgument;
  if (ncclGroupDepth > 0) {
    WARN("COCCL compressed Send cannot execute inside an outer NCCL group");
    return ncclGroupErrCheck(ncclInvalidUsage);
  }

  bool forward = comm->rank < peer;
  ncclCommOp_t compop = forward ? ncclCommOp_t::SendRecv : ncclCommOp_t::SendRecv_BWD;

  compMeta_t** hMeta = forward ? &hCompSendMeta : &hCompBWDSendMeta;
  compMeta_t** dMeta = forward ? &dCompSendMeta : &dCompBWDSendMeta;

  CUDACHECK(cudaSetDevice(comm->cudaDev));

  NCCLCHECK(cocclEnsureSendRecvLane(comm, true));
  NCCLCHECK(cocclEnsureSendRecvLane(comm, false));
  NCCLCHECK(cocclEnsureSendRecvMeta(hMeta, dMeta, cudaHostAllocWriteCombined));

  INFO(NCCL_INIT, "SendComp_datatype_%d_sendbytes_%zuMB_rank_%d_peer_%d_nRanks_%d_stream_%p",
       datatype, count * ncclTypeSize(datatype) / 1024 / 1024, comm->rank, peer, comm->nRanks,
       (void*)stream);

  size_t totalSendBytes = 2 * count * ncclTypeSize(datatype);
  ncclComm_t sendComm = forward ? fwdComm : bwdComm;
  cudaStream_t sendStream = forward ? fwdStream : bwdStream;

  if (mEvent == nullptr) {
    CUDACHECK(cudaEventCreateWithFlags(&mEvent, cudaEventDefault));
  }
  CUDACHECK(cudaEventRecord(mEvent, stream));
  CUDACHECK(cudaStreamWaitEvent(sendStream, mEvent, 0));

  cocclBufferHandle sendBuffer = {};
  NCCLCHECK(cocclGetBufferForComm(comm, sendComm, totalSendBytes, &sendBuffer));
  void* compBuff = sendBuffer.ptr;

  NCCLCHECK(ncclCompress(sendbuff, &compBuff, count, datatype, &((*hMeta)->compCount),
                         &((*hMeta)->compDatatype), 1, comm->rank, comm, compop, sendStream));
  CUDACHECK(cudaMemcpyAsync(*dMeta, *hMeta, sizeof(compMeta_t), cudaMemcpyHostToDevice, sendStream));
  CUDACHECK(cudaStreamSynchronize(sendStream));

  NCCLCHECK(ncclSendNaive(*dMeta, sizeof(compMeta_t), ncclDataType_t::ncclInt8, peer, sendComm,
                          sendStream));
  NCCLCHECK(ncclSendNaive(compBuff, (*hMeta)->compCount, (*hMeta)->compDatatype, peer, sendComm,
                          sendStream));
  NCCLCHECK(cocclReleaseBuffer(&sendBuffer, sendStream));

  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclRecvDecomp, void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
  ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclRecvDecomp(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
  ncclComm_t comm, cudaStream_t stream) {
  if (comm == nullptr) return ncclInvalidArgument;
  if (ncclGroupDepth > 0) {
    WARN("COCCL compressed Recv cannot execute inside an outer NCCL group");
    return ncclGroupErrCheck(ncclInvalidUsage);
  }

  bool forward = comm->rank > peer;
  ncclCommOp_t compop = forward ? ncclCommOp_t::SendRecv : ncclCommOp_t::SendRecv_BWD;

  compMeta_t** hMeta = forward ? &hCompRecvMeta : &hCompBWDRecvMeta;
  compMeta_t** dMeta = forward ? &dCompRecvMeta : &dCompBWDRecvMeta;

  CUDACHECK(cudaSetDevice(comm->cudaDev));

  NCCLCHECK(cocclEnsureSendRecvMeta(hMeta, dMeta, cudaHostAllocDefault));
  NCCLCHECK(cocclEnsureSendRecvLane(comm, true));
  NCCLCHECK(cocclEnsureSendRecvLane(comm, false));

  ncclComm_t recvComm = forward ? fwdComm : bwdComm;
  cudaStream_t recvStream = forward ? fwdStream : bwdStream;
  cudaEvent_t recvEvent = forward ? fwdEvent : bwdEvent;

  INFO(NCCL_INIT, "RecvComp_datatype_%d_recvbuff_%zuMB_rank_%d_peer_%d_nRanks_%d_stream_%p",
       datatype, count * ncclTypeSize(datatype) / 1024 / 1024, comm->rank, peer, comm->nRanks,
       (void*)stream);

  NCCLCHECK(ncclRecvNaive(*dMeta, sizeof(compMeta_t), ncclDataType_t::ncclInt8, peer, recvComm,
                          recvStream));

  CUDACHECK(cudaMemcpyAsync(*hMeta, *dMeta, sizeof(compMeta_t), cudaMemcpyDeviceToHost, recvStream));
  CUDACHECK(cudaStreamSynchronize(recvStream));

  if ((*hMeta)->compCount > 0) {
    size_t totalSendBytes = (*hMeta)->compCount * ncclTypeSize((*hMeta)->compDatatype);
    cocclBufferHandle recvBuffer = {};
    NCCLCHECK(cocclGetBufferForComm(comm, recvComm, totalSendBytes, &recvBuffer));
    void* compBuff = recvBuffer.ptr;

    NCCLCHECK(ncclRecvNaive(compBuff, (*hMeta)->compCount, (*hMeta)->compDatatype, peer, recvComm,
                            recvStream));

    NCCLCHECK(ncclDecompress(recvbuff, compBuff, count, datatype, (*hMeta)->compCount,
                             (*hMeta)->compDatatype, 1, comm, compop, recvStream));
    NCCLCHECK(cocclReleaseBuffer(&recvBuffer, recvStream));
  }

  CUDACHECK(cudaEventRecord(recvEvent, recvStream));
  CUDACHECK(cudaStreamWaitEvent(stream, recvEvent, 0));

  return ncclSuccess;
}
