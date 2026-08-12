#include "pipeline/coccl_frame_exchange.h"

#include "collectives.h"
#include "comm.h"

#include <stdint.h>

namespace {

bool cocclValidSlot(const void* slot, size_t bytes, size_t slotBytes) {
  if (bytes == 0) return slot == nullptr || slotBytes != 0;
  if (slot == nullptr || slotBytes == 0 || bytes > slotBytes) return false;
  return slotBytes <= UINTPTR_MAX - (uintptr_t)slot;
}

bool cocclValidFrameMetadata(const cocclCompressorFrameMetadata& metadata,
                             size_t strideBytes) {
  return metadata.payloadBytes > 0 &&
      metadata.payloadBytes <= strideBytes && metadata.reserved == 0 &&
      (metadata.encoding == cocclCompressorFrameEncoded ||
       (metadata.encoding == cocclCompressorFrameRaw &&
        metadata.payloadBytes == strideBytes));
}

bool cocclFrameSlot(const void* base, size_t frame, size_t strideBytes,
                    const void** slot) {
  if (base == nullptr || slot == nullptr || strideBytes == 0 ||
      frame > SIZE_MAX / strideBytes) {
    return false;
  }
  const size_t offset = frame * strideBytes;
  if (offset > UINTPTR_MAX - (uintptr_t)base) return false;
  *slot = (const char*)base + offset;
  return true;
}

bool cocclFrameSlot(void* base, size_t frame, size_t strideBytes,
                    void** slot) {
  const void* result = nullptr;
  if (!cocclFrameSlot((const void*)base, frame, strideBytes, &result)) {
    return false;
  }
  *slot = const_cast<void*>(result);
  return true;
}

}  // namespace

ncclResult_t cocclBuildAllToAllFrameExchanges(
    const void* sendBase, void* recvBase, size_t frames,
    size_t frameStrideBytes, int nRanks,
    const cocclCompressorFrameMetadata* sendMetadata,
    const cocclCompressorFrameMetadata* recvMetadata,
    cocclFrameExchange* exchanges, size_t exchangeCapacity,
    size_t* exchangeCount) {
  if (sendBase == nullptr || recvBase == nullptr || frames == 0 ||
      frameStrideBytes == 0 || nRanks <= 0 ||
      frames % (size_t)nRanks != 0 || sendMetadata == nullptr ||
      recvMetadata == nullptr || exchanges == nullptr ||
      exchangeCapacity < frames || exchangeCount == nullptr) {
    return ncclInvalidArgument;
  }
  *exchangeCount = 0;
  const size_t framesPerPeer = frames / (size_t)nRanks;
  for (int peer = 0; peer < nRanks; ++peer) {
    for (size_t frame = 0; frame < framesPerPeer; ++frame) {
      const size_t frameIndex = (size_t)peer * framesPerPeer + frame;
      const auto& send = sendMetadata[frameIndex];
      const auto& recv = recvMetadata[frameIndex];
      const void* sendSlot = nullptr;
      void* recvSlot = nullptr;
      if (!cocclValidFrameMetadata(send, frameStrideBytes) ||
          !cocclValidFrameMetadata(recv, frameStrideBytes) ||
          !cocclFrameSlot(sendBase, frameIndex, frameStrideBytes,
                          &sendSlot) ||
          !cocclFrameSlot(recvBase, frameIndex, frameStrideBytes,
                          &recvSlot)) {
        return ncclInvalidUsage;
      }
      exchanges[(*exchangeCount)++] = {
          peer, sendSlot, recvSlot, (size_t)send.payloadBytes,
          (size_t)recv.payloadBytes, frameStrideBytes};
    }
  }
  return ncclSuccess;
}

ncclResult_t cocclBuildAllGatherFrameExchanges(
    const void* sendBase, void* recvBase, size_t localFrames,
    size_t frameStrideBytes, int nRanks,
    const cocclCompressorFrameMetadata* sendMetadata,
    const cocclCompressorFrameMetadata* recvMetadata,
    cocclFrameExchange* exchanges, size_t exchangeCapacity,
    size_t* exchangeCount) {
  size_t totalFrames = 0;
  if (sendBase == nullptr || recvBase == nullptr || localFrames == 0 ||
      frameStrideBytes == 0 || nRanks <= 0 ||
      localFrames > SIZE_MAX / (size_t)nRanks ||
      (totalFrames = localFrames * (size_t)nRanks) == 0 ||
      sendMetadata == nullptr || recvMetadata == nullptr ||
      exchanges == nullptr || exchangeCapacity < totalFrames ||
      exchangeCount == nullptr) {
    return ncclInvalidArgument;
  }
  *exchangeCount = 0;
  for (int peer = 0; peer < nRanks; ++peer) {
    for (size_t frame = 0; frame < localFrames; ++frame) {
      const size_t recvIndex = (size_t)peer * localFrames + frame;
      const auto& send = sendMetadata[frame];
      const auto& recv = recvMetadata[recvIndex];
      const void* sendSlot = nullptr;
      void* recvSlot = nullptr;
      if (!cocclValidFrameMetadata(send, frameStrideBytes) ||
          !cocclValidFrameMetadata(recv, frameStrideBytes) ||
          !cocclFrameSlot(sendBase, frame, frameStrideBytes, &sendSlot) ||
          !cocclFrameSlot(recvBase, recvIndex, frameStrideBytes,
                          &recvSlot)) {
        return ncclInvalidUsage;
      }
      exchanges[(*exchangeCount)++] = {
          peer, sendSlot, recvSlot, (size_t)send.payloadBytes,
          (size_t)recv.payloadBytes, frameStrideBytes};
    }
  }
  return ncclSuccess;
}

ncclResult_t cocclCommitFrameExchange(
    const cocclFrameExchange* exchanges, size_t count,
    ncclComm_t comm, cudaStream_t stream) {
  if (exchanges == nullptr || count == 0 || comm == nullptr) {
    return ncclInvalidArgument;
  }
  for (size_t i = 0; i < count; ++i) {
    const auto& exchange = exchanges[i];
    if (exchange.peer < 0 || exchange.peer >= comm->nRanks ||
        (exchange.sendBytes == 0 && exchange.recvBytes == 0) ||
        !cocclValidSlot(exchange.sendSlot, exchange.sendBytes,
                        exchange.slotBytes) ||
        !cocclValidSlot(exchange.recvSlot, exchange.recvBytes,
                        exchange.slotBytes)) {
      return ncclInvalidArgument;
    }
  }

  ncclResult_t ret = ncclGroupStart();
  if (ret != ncclSuccess) return ret;
  for (size_t i = 0; i < count; ++i) {
    const auto& exchange = exchanges[i];
    if (exchange.recvBytes != 0) {
      ret = ncclRecvNaive(
          exchange.recvSlot, exchange.recvBytes, ncclInt8,
          exchange.peer, comm, stream);
      if (ret != ncclSuccess) break;
    }
    if (exchange.sendBytes != 0) {
      ret = ncclSendNaive(
          exchange.sendSlot, exchange.sendBytes, ncclInt8,
          exchange.peer, comm, stream);
      if (ret != ncclSuccess) break;
    }
  }

  const ncclResult_t endResult = ncclGroupEnd();
  return ret == ncclSuccess ? endResult : ret;
}
