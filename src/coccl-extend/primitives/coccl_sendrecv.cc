#include "primitives/coccl_primitives_internal.h"

#include "group.h"
#include "pipeline/coccl_frame_exchange.h"
#include "training/coccl_training_assist.h"

#include <limits.h>

// Two thread-local split communicators provide independent forward/backward
// lanes. The runtime contract requires one rank per host thread.
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

__thread cocclCompressorFrameMetadata* hFrameSendMeta = nullptr;
__thread cocclCompressorFrameMetadata* dFrameSendMeta = nullptr;
__thread cocclCompressorFrameMetadata* hFrameBWDSendMeta = nullptr;
__thread cocclCompressorFrameMetadata* dFrameBWDSendMeta = nullptr;
__thread cocclCompressorFrameMetadata* hFrameRecvMeta = nullptr;
__thread cocclCompressorFrameMetadata* dFrameRecvMeta = nullptr;
__thread cocclCompressorFrameMetadata* hFrameBWDRecvMeta = nullptr;
__thread cocclCompressorFrameMetadata* dFrameBWDRecvMeta = nullptr;

namespace {

bool cocclSendRecvBytes(size_t count, ncclDataType_t datatype,
                        size_t* bytes) {
  const int typeSize = ncclTypeSize(datatype);
  if (bytes == nullptr || typeSize <= 0 ||
      count > SIZE_MAX / (size_t)typeSize) {
    return false;
  }
  *bytes = count * (size_t)typeSize;
  return true;
}

ncclResult_t cocclEnsureSendRecvLane(ncclComm_t comm, bool forward) {
  if (comm == nullptr) return ncclInvalidArgument;

  ncclComm_t* laneComm = forward ? &fwdComm : &bwdComm;
  cudaStream_t* laneStream = forward ? &fwdStream : &bwdStream;
  cudaEvent_t* laneEvent = forward ? &fwdEvent : &bwdEvent;
  const int color = forward ? 0 : 1;

  if (*laneComm == nullptr) {
    NCCLCHECK(ncclCommSplit(comm, color, comm->rank, laneComm, NULL));
    cocclTrainingAssistUnregister(*laneComm);
    CUDACHECK(cudaStreamCreateWithFlags(laneStream, cudaStreamNonBlocking));
    CUDACHECK(cudaEventCreateWithFlags(laneEvent, cudaEventDisableTiming));
  }
  return ncclSuccess;
}

ncclResult_t cocclEnsureSendRecvMeta(compMeta_t** hostMeta,
                                     compMeta_t** deviceMeta,
                                     unsigned int hostFlags) {
  if (hostMeta == nullptr || deviceMeta == nullptr) {
    return ncclInvalidArgument;
  }
  if (*hostMeta == nullptr) {
    CUDACHECK(cudaHostAlloc((void**)hostMeta, sizeof(compMeta_t), hostFlags));
    CUDACHECK(cudaMalloc((void**)deviceMeta, sizeof(compMeta_t)));
  }
  return ncclSuccess;
}

ncclResult_t cocclEnsureSendRecvFrameMeta(
    cocclCompressorFrameMetadata** hostMeta,
    cocclCompressorFrameMetadata** deviceMeta) {
  if (hostMeta == nullptr || deviceMeta == nullptr) {
    return ncclInvalidArgument;
  }
  if (*hostMeta == nullptr) {
    CUDACHECK(cudaHostAlloc(
        (void**)hostMeta, sizeof(cocclCompressorFrameMetadata),
        cudaHostAllocDefault));
    CUDACHECK(cudaMalloc(
        (void**)deviceMeta, sizeof(cocclCompressorFrameMetadata)));
  }
  return ncclSuccess;
}

bool cocclValidSendRecvFrame(
    const cocclCompressorFrameMetadata& metadata, size_t slotBytes) {
  return metadata.payloadBytes > 0 && metadata.payloadBytes <= slotBytes &&
      metadata.reserved == 0 &&
      (metadata.encoding == cocclCompressorFrameEncoded ||
       (metadata.encoding == cocclCompressorFrameRaw &&
        metadata.payloadBytes == slotBytes));
}

bool cocclSendRecvForward(const cocclInfo& args) {
  return args.func == ncclFuncSend
      ? args.comm->rank < args.peer
      : args.comm->rank > args.peer;
}

cocclPolicyKey cocclDirectSendRecvPolicy(const cocclInfo& args) {
  return cocclTrainingAssistEnabled()
      ? cocclDirectionalPolicy(cocclOperation::SendRecv,
                               cocclSendRecvForward(args))
      : cocclDefaultPolicy(cocclOperation::SendRecv);
}

ncclResult_t cocclRunFramedSend(
    const cocclPreparedCall* prepared, ncclComm_t sendComm,
    cudaStream_t sendStream, size_t rawBytes,
    cocclCompressorFrameMetadata** hostMetadata,
    cocclCompressorFrameMetadata** deviceMetadata) {
  const cocclInfo& args = prepared->info;
  NCCLCHECK(cocclEnsureSendRecvFrameMeta(hostMetadata, deviceMetadata));

  ncclResult_t ret = ncclSuccess;
  cocclBufferHandle sendBuffer = {};
  NCCLCHECKGOTO(cocclGetBufferForComm(
                    args.comm, sendComm, rawBytes, &sendBuffer),
                ret, exit);
  {
    const cocclCompressorDataView input = {
        args.sendbuff, rawBytes, args.count, 1, args.datatype, nullptr, 0};
    cocclCompressorOutputView output = {
        sendBuffer.ptr, rawBytes, 0, 0, 1, ncclInt8,
        *deviceMetadata, rawBytes};
    NCCLCHECKGOTO(ncclCompress(
                      prepared->compressor, input, &output,
                      args.comm->rank, sendStream),
                  ret, exit);
    NCCLCHECKGOTO(
        ncclSendNaive(*deviceMetadata,
                      sizeof(cocclCompressorFrameMetadata), ncclInt8,
                      args.peer, sendComm, sendStream),
        ret, exit);
    CUDACHECKGOTO(cudaMemcpyAsync(
                      *hostMetadata, *deviceMetadata,
                      sizeof(cocclCompressorFrameMetadata),
                      cudaMemcpyDeviceToHost, sendStream),
                  ret, exit);
    CUDACHECKGOTO(cudaStreamSynchronize(sendStream), ret, exit);
    if (!cocclValidSendRecvFrame(**hostMetadata, rawBytes)) {
      ret = ncclInvalidUsage;
      goto exit;
    }
    const cocclFrameExchange exchange = {
        args.peer, output.data, nullptr,
        (size_t)(*hostMetadata)->payloadBytes, 0, rawBytes};
    NCCLCHECKGOTO(cocclCommitFrameExchange(
                      &exchange, 1, sendComm, sendStream),
                  ret, exit);
  }

exit:
  if (sendBuffer.ptr != nullptr) {
    const ncclResult_t releaseResult =
        cocclReleaseBuffer(&sendBuffer, sendStream);
    if (ret == ncclSuccess) ret = releaseResult;
  }
  return ret;
}

ncclResult_t cocclRunFramedRecv(
    const cocclPreparedCall* prepared, ncclComm_t recvComm,
    cudaStream_t recvStream, size_t rawBytes,
    cocclCompressorFrameMetadata** hostMetadata,
    cocclCompressorFrameMetadata** deviceMetadata) {
  const cocclInfo& args = prepared->info;
  NCCLCHECK(cocclEnsureSendRecvFrameMeta(hostMetadata, deviceMetadata));
  NCCLCHECK(ncclRecvNaive(
      *deviceMetadata, sizeof(cocclCompressorFrameMetadata), ncclInt8,
      args.peer, recvComm, recvStream));
  CUDACHECK(cudaMemcpyAsync(
      *hostMetadata, *deviceMetadata, sizeof(cocclCompressorFrameMetadata),
      cudaMemcpyDeviceToHost, recvStream));
  CUDACHECK(cudaStreamSynchronize(recvStream));
  if (!cocclValidSendRecvFrame(**hostMetadata, rawBytes)) {
    return ncclInvalidUsage;
  }

  ncclResult_t ret = ncclSuccess;
  cocclBufferHandle recvBuffer = {};
  NCCLCHECKGOTO(cocclGetBufferForComm(
                    args.comm, recvComm, rawBytes, &recvBuffer),
                ret, exit);
  {
    const cocclFrameExchange exchange = {
        args.peer, nullptr, recvBuffer.ptr, 0,
        (size_t)(*hostMetadata)->payloadBytes, rawBytes};
    NCCLCHECKGOTO(cocclCommitFrameExchange(
                      &exchange, 1, recvComm, recvStream),
                  ret, exit);
    const cocclCompressorDataView input = {
        recvBuffer.ptr, rawBytes, rawBytes, 1, ncclInt8,
        *deviceMetadata, rawBytes};
    cocclCompressorOutputView output = {
        args.recvbuff, rawBytes, 0, args.count, 1, args.datatype,
        nullptr, 0};
    NCCLCHECKGOTO(ncclDecompress(
                      prepared->compressor, input, &output, recvStream),
                  ret, exit);
  }

exit:
  if (recvBuffer.ptr != nullptr) {
    const ncclResult_t releaseResult =
        cocclReleaseBuffer(&recvBuffer, recvStream);
    if (ret == ncclSuccess) ret = releaseResult;
  }
  return ret;
}

ncclResult_t cocclRunSend(const cocclPreparedCall* prepared) {
  const cocclInfo& args = prepared->info;
  if (ncclGroupDepth > 0) {
    WARN("COCCL compressed Send cannot execute inside an outer NCCL group");
    return ncclGroupErrCheck(ncclInvalidUsage);
  }

  const bool forward = cocclSendRecvForward(args);
  compMeta_t** hMeta = forward ? &hCompSendMeta : &hCompBWDSendMeta;
  compMeta_t** dMeta = forward ? &dCompSendMeta : &dCompBWDSendMeta;

  CUDACHECK(cudaSetDevice(args.comm->cudaDev));
  NCCLCHECK(cocclEnsureSendRecvLane(args.comm, true));
  NCCLCHECK(cocclEnsureSendRecvLane(args.comm, false));
  NCCLCHECK(cocclEnsureSendRecvMeta(hMeta, dMeta,
                                    cudaHostAllocWriteCombined));

  size_t rawBytes = 0;
  if (!cocclSendRecvBytes(args.count, args.datatype, &rawBytes)) {
    return ncclInvalidArgument;
  }
  ncclComm_t sendComm = forward ? fwdComm : bwdComm;
  cudaStream_t sendStream = forward ? fwdStream : bwdStream;

  if (mEvent == nullptr) {
    CUDACHECK(cudaEventCreateWithFlags(&mEvent, cudaEventDisableTiming));
  }
  CUDACHECK(cudaEventRecord(mEvent, args.stream));
  CUDACHECK(cudaStreamWaitEvent(sendStream, mEvent, 0));

  if (cocclCompressorSupports(
          prepared->compressor, cocclCompressorCapabilityFramed)) {
    cocclCompressorFrameMetadata** hFrameMeta = forward
        ? &hFrameSendMeta : &hFrameBWDSendMeta;
    cocclCompressorFrameMetadata** dFrameMeta = forward
        ? &dFrameSendMeta : &dFrameBWDSendMeta;
    return cocclRunFramedSend(
        prepared, sendComm, sendStream, rawBytes, hFrameMeta, dFrameMeta);
  }
  if (rawBytes > SIZE_MAX / 2) return ncclInvalidArgument;
  const size_t capacity = 2 * rawBytes;

  cocclBufferHandle sendBuffer = {};
  NCCLCHECK(cocclGetBufferForComm(args.comm, sendComm, capacity,
                                  &sendBuffer));
  const cocclCompressorDataView input = {
      args.sendbuff, rawBytes, args.count, 1, args.datatype};
  cocclCompressorOutputView output = {
      sendBuffer.ptr, sendBuffer.bytes, 0, 0, 1, ncclInt8};
  NCCLCHECK(ncclCompress(prepared->compressor, input, &output,
                         args.comm->rank, sendStream));
  (*hMeta)->compCount = output.elements;
  (*hMeta)->compDatatype = output.datatype;
  CUDACHECK(cudaMemcpyAsync(*dMeta, *hMeta, sizeof(compMeta_t),
                            cudaMemcpyHostToDevice, sendStream));
  CUDACHECK(cudaStreamSynchronize(sendStream));

  NCCLCHECK(ncclSendNaive(*dMeta, sizeof(compMeta_t), ncclInt8, args.peer,
                          sendComm, sendStream));
  NCCLCHECK(ncclSendNaive(output.data, output.elements, output.datatype,
                          args.peer, sendComm, sendStream));
  return cocclReleaseBuffer(&sendBuffer, sendStream);
}

ncclResult_t cocclRunRecv(const cocclPreparedCall* prepared) {
  const cocclInfo& args = prepared->info;
  if (ncclGroupDepth > 0) {
    WARN("COCCL compressed Recv cannot execute inside an outer NCCL group");
    return ncclGroupErrCheck(ncclInvalidUsage);
  }

  const bool forward = cocclSendRecvForward(args);
  compMeta_t** hMeta = forward ? &hCompRecvMeta : &hCompBWDRecvMeta;
  compMeta_t** dMeta = forward ? &dCompRecvMeta : &dCompBWDRecvMeta;

  CUDACHECK(cudaSetDevice(args.comm->cudaDev));
  NCCLCHECK(cocclEnsureSendRecvMeta(hMeta, dMeta, cudaHostAllocDefault));
  NCCLCHECK(cocclEnsureSendRecvLane(args.comm, true));
  NCCLCHECK(cocclEnsureSendRecvLane(args.comm, false));

  ncclComm_t recvComm = forward ? fwdComm : bwdComm;
  cudaStream_t recvStream = forward ? fwdStream : bwdStream;
  cudaEvent_t recvEvent = forward ? fwdEvent : bwdEvent;

  if (cocclCompressorSupports(
          prepared->compressor, cocclCompressorCapabilityFramed)) {
    size_t rawBytes = 0;
    if (!cocclSendRecvBytes(args.count, args.datatype, &rawBytes)) {
      return ncclInvalidArgument;
    }
    cocclCompressorFrameMetadata** hFrameMeta = forward
        ? &hFrameRecvMeta : &hFrameBWDRecvMeta;
    cocclCompressorFrameMetadata** dFrameMeta = forward
        ? &dFrameRecvMeta : &dFrameBWDRecvMeta;
    NCCLCHECK(cocclRunFramedRecv(
        prepared, recvComm, recvStream, rawBytes, hFrameMeta, dFrameMeta));
    CUDACHECK(cudaEventRecord(recvEvent, recvStream));
    CUDACHECK(cudaStreamWaitEvent(args.stream, recvEvent, 0));
    return ncclSuccess;
  }

  NCCLCHECK(ncclRecvNaive(*dMeta, sizeof(compMeta_t), ncclInt8, args.peer,
                          recvComm, recvStream));
  CUDACHECK(cudaMemcpyAsync(*hMeta, *dMeta, sizeof(compMeta_t),
                            cudaMemcpyDeviceToHost, recvStream));
  CUDACHECK(cudaStreamSynchronize(recvStream));

  if ((*hMeta)->compCount > 0) {
    size_t compressedBytes = 0;
    size_t rawBytes = 0;
    if (!cocclSendRecvBytes((*hMeta)->compCount,
                            (*hMeta)->compDatatype, &compressedBytes) ||
        !cocclSendRecvBytes(args.count, args.datatype, &rawBytes)) {
      return ncclInvalidArgument;
    }
    cocclBufferHandle recvBuffer = {};
    NCCLCHECK(cocclGetBufferForComm(args.comm, recvComm, compressedBytes,
                                    &recvBuffer));
    NCCLCHECK(ncclRecvNaive(recvBuffer.ptr, (*hMeta)->compCount,
                            (*hMeta)->compDatatype, args.peer, recvComm,
                            recvStream));
    const cocclCompressorDataView input = {
        recvBuffer.ptr, compressedBytes, (*hMeta)->compCount, 1,
        (*hMeta)->compDatatype};
    cocclCompressorOutputView output = {
        args.recvbuff, rawBytes, 0, args.count, 1, args.datatype};
    NCCLCHECK(ncclDecompress(prepared->compressor, input, &output,
                             recvStream));
    NCCLCHECK(cocclReleaseBuffer(&recvBuffer, recvStream));
  }

  CUDACHECK(cudaEventRecord(recvEvent, recvStream));
  CUDACHECK(cudaStreamWaitEvent(args.stream, recvEvent, 0));
  return ncclSuccess;
}

cocclPreparedCall cocclDirectSendRecvCall(
    const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, int peer, ncclFunc_t func, ncclComm_t comm,
    cudaStream_t stream) {
  cocclPreparedCall prepared;
  prepared.info.sendbuff = sendbuff;
  prepared.info.recvbuff = recvbuff;
  prepared.info.count = count;
  prepared.info.datatype = datatype;
  prepared.info.peer = peer;
  prepared.info.func = func;
  prepared.info.operation = cocclOperation::SendRecv;
  prepared.info.comm = comm;
  prepared.info.stream = stream;
  prepared.descriptor =
      cocclGetOperationDescriptor(cocclOperation::SendRecv);
  return prepared;
}

}  // namespace

NCCL_API(ncclResult_t, ncclSendComp, const void* sendbuff, size_t count,
  ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclSendComp(const void* sendbuff, size_t count,
  ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream) {
  if (comm == nullptr) return ncclInvalidArgument;
  cocclPreparedCall prepared = cocclDirectSendRecvCall(
      sendbuff, nullptr, count, datatype, peer, ncclFuncSend, comm, stream);
  prepared.policy = cocclDirectSendRecvPolicy(prepared.info);
  NCCLCHECK(cocclCommGetCompressor(
      comm, prepared.policy, &prepared.compressor));
  return cocclExecutePreparedCall(&prepared);
}

NCCL_API(ncclResult_t, ncclRecvDecomp, void* recvbuff, size_t count,
  ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclRecvDecomp(void* recvbuff, size_t count,
  ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream) {
  if (comm == nullptr) return ncclInvalidArgument;
  cocclPreparedCall prepared = cocclDirectSendRecvCall(
      nullptr, recvbuff, count, datatype, peer, ncclFuncRecv, comm, stream);
  prepared.policy = cocclDirectSendRecvPolicy(prepared.info);
  NCCLCHECK(cocclCommGetCompressor(
      comm, prepared.policy, &prepared.compressor));
  return cocclExecutePreparedCall(&prepared);
}

ncclResult_t cocclExecuteSendRecv(const cocclPreparedCall* prepared) {
  if (prepared == nullptr || prepared->info.comm == nullptr ||
      !prepared->compressor) {
    return ncclInvalidArgument;
  }
  if (prepared->info.func == ncclFuncSend) return cocclRunSend(prepared);
  if (prepared->info.func == ncclFuncRecv) return cocclRunRecv(prepared);
  return ncclInvalidArgument;
}
