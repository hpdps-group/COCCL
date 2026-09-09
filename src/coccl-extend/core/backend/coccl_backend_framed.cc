#include "core/backend/coccl_backend_framed.h"

#include "collectives.h"
#include "comm.h"
#include "enqueue.h"

bool cocclBackendUseAllGatherV(
    ncclComm_t comm,
    const cocclFrameExchange* exchanges, size_t exchangeCount) {
  if (ncclParamAllgathervEnable() == 0 ||
      ncclParamEnqueueRearchEnable() != 0 || comm->ccEnable) {
    return false;
  }
  if (comm->nNodes > 1) return true;

  constexpr size_t kSingleNodeP2PFrameBytes = size_t{1} << 30;
  for (size_t i = 0; i < exchangeCount; ++i) {
    if (exchanges[i].recvBytes >= kSingleNodeP2PFrameBytes) return false;
  }
  return true;
}

ncclResult_t cocclCommitAllGatherVFrameExchange(
    const cocclFrameExchange* exchanges, size_t localFrames,
    ncclComm_t comm, cudaStream_t stream) {
  for (size_t frame = 0; frame < localFrames; ++frame) {
    ncclResult_t ret = ncclGroupStart();
    if (ret != ncclSuccess) return ret;
    for (int root = 0; root < comm->nRanks; ++root) {
      const cocclFrameExchange& exchange =
          exchanges[(size_t)root * localFrames + frame];
      ret = ncclBroadcast(
          exchange.sendSlot, exchange.recvSlot, exchange.recvBytes,
          ncclInt8, root, comm, stream);
      if (ret != ncclSuccess) break;
    }
    const ncclResult_t endResult = ncclGroupEnd();
    if (ret != ncclSuccess) return ret;
    if (endResult != ncclSuccess) return endResult;
  }
  return ncclSuccess;
}

ncclResult_t cocclCommitAllToAllRmaFrameExchange(
    const cocclFrameExchange* exchanges, size_t frames,
    size_t frameStrideBytes, int rank, ncclWindow_t window,
    size_t outputWindowOffset, ncclWaitSignalDesc_t* waitDescriptors,
    ncclComm_t comm, cudaStream_t stream) {
  const size_t framesPerPeer = frames / (size_t)comm->nRanks;
  const size_t rankFrame = (size_t)rank * framesPerPeer;

  ncclResult_t ret = ncclGroupStart();
  if (ret != ncclSuccess) return ret;
  for (int peer = 0; peer < comm->nRanks; ++peer) {
    for (size_t frame = 0; frame < framesPerPeer; ++frame) {
      const cocclFrameExchange& exchange =
          exchanges[(size_t)peer * framesPerPeer + frame];
      const size_t remoteOffset = outputWindowOffset +
          (rankFrame + frame) * frameStrideBytes;
      ret = ncclPutSignal(
          exchange.sendSlot, exchange.sendBytes, ncclInt8, peer, window,
          remoteOffset, 0, 0, 0, comm, stream);
      if (ret != ncclSuccess) break;
    }
    if (ret != ncclSuccess) break;
  }
  const ncclResult_t endResult = ncclGroupEnd();
  if (ret != ncclSuccess) return ret;
  if (endResult != ncclSuccess) return endResult;

  for (int peer = 0; peer < comm->nRanks; ++peer) {
    waitDescriptors[peer] = {
        (int)framesPerPeer, peer, 0, 0};
  }
  return ncclWaitSignal(
      comm->nRanks, waitDescriptors, comm, stream);
}
