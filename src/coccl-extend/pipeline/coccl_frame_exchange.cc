#include "pipeline/coccl_frame_exchange.h"

#include "collectives.h"
#include "comm.h"
#include "runtime/coccl_group.h"
#include "runtime/coccl_runtime.h"

#include <stdint.h>

namespace {

bool cocclValidSlot(const void* slot, size_t bytes, size_t slotBytes) {
  if (bytes == 0) return slot == nullptr || slotBytes != 0;
  return slot != nullptr && slotBytes != 0 && bytes <= slotBytes;
}

bool cocclFrameSlot(const void* base, size_t frame, size_t strideBytes,
                    const void** slot) {
  if (base == nullptr || slot == nullptr || strideBytes == 0 ||
      frame > SIZE_MAX / strideBytes) {
    return false;
  }
  const size_t offset = frame * strideBytes;
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

bool cocclFrameMetadataValid(
    const cocclCompressorFrameMetadata& metadata, size_t frameStrideBytes) {
  return metadata.payloadBytes > 0 &&
      metadata.payloadBytes <= frameStrideBytes &&
      (metadata.encoding == cocclCompressorFrameEncoded ||
       (metadata.encoding == cocclCompressorFrameRaw &&
        metadata.payloadBytes == frameStrideBytes));
}

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
      if (!cocclFrameMetadataValid(send, frameStrideBytes) ||
          !cocclFrameMetadataValid(recv, frameStrideBytes) ||
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
      if (!cocclFrameMetadataValid(send, frameStrideBytes) ||
          !cocclFrameMetadataValid(recv, frameStrideBytes) ||
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
  if (exchanges == nullptr || count == 0) {
    return ncclInvalidArgument;
  }
  for (size_t i = 0; i < count; ++i) {
    const auto& exchange = exchanges[i];
    ncclComm_t exchangeComm =
        exchange.comm == nullptr ? comm : exchange.comm;
    if (exchangeComm == nullptr || exchange.peer < 0 ||
        exchange.peer >= exchangeComm->nRanks ||
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
    ncclComm_t exchangeComm =
        exchange.comm == nullptr ? comm : exchange.comm;
    cudaStream_t exchangeStream =
        exchange.comm == nullptr ? stream : exchange.stream;
    if (exchange.recvBytes != 0) {
      cocclInfo info;
      info.recvbuff = exchange.recvSlot;
      info.count = exchange.recvBytes;
      info.datatype = ncclInt8;
      info.peer = exchange.peer;
      info.func = ncclFuncRecv;
      info.operation = cocclOperation::SendRecv;
      info.comm = exchangeComm;
      info.stream = exchangeStream;
      ret = cocclReplayNativeCall(info);
      if (ret != ncclSuccess) break;
    }
    if (exchange.sendBytes != 0) {
      cocclInfo info;
      info.sendbuff = exchange.sendSlot;
      info.count = exchange.sendBytes;
      info.datatype = ncclInt8;
      info.peer = exchange.peer;
      info.func = ncclFuncSend;
      info.operation = cocclOperation::SendRecv;
      info.comm = exchangeComm;
      info.stream = exchangeStream;
      ret = cocclReplayNativeCall(info);
      if (ret != ncclSuccess) break;
    }
  }

  const ncclResult_t endResult = ncclGroupEnd();
  return ret == ncclSuccess ? endResult : ret;
}
