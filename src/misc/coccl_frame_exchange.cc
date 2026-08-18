#include "coccl_frame_exchange.h"

#include "collectives.h"
#include "comm.h"
#include "coccl_runtime.h"

namespace {

bool validSlot(const void* slot, size_t bytes, size_t slotBytes) {
  if (bytes == 0) return slot == nullptr || slotBytes != 0;
  return slot != nullptr && bytes <= slotBytes;
}

}  // namespace

bool cocclFrameMetadataValid(
    const cocclCompressorFrameMetadata& metadata,
    size_t frameStrideBytes) {
  return metadata.payloadBytes > 0 &&
      metadata.payloadBytes <= frameStrideBytes &&
      (metadata.encoding == cocclCompressorFrameEncoded ||
       (metadata.encoding == cocclCompressorFrameRaw &&
        metadata.payloadBytes == frameStrideBytes));
}

ncclResult_t cocclCommitFrameExchange(
    const cocclFrameExchange* exchanges, size_t count,
    ncclComm_t comm, cudaStream_t stream) {
  if (exchanges == nullptr || count == 0) return ncclInvalidArgument;
  for (size_t i = 0; i < count; ++i) {
    const cocclFrameExchange& exchange = exchanges[i];
    ncclComm_t exchangeComm =
        exchange.comm == nullptr ? comm : exchange.comm;
    if (exchangeComm == nullptr || exchange.peer < 0 ||
        exchange.peer >= exchangeComm->nRanks ||
        (exchange.sendBytes == 0 && exchange.recvBytes == 0) ||
        !validSlot(exchange.sendSlot, exchange.sendBytes,
                   exchange.slotBytes) ||
        !validSlot(exchange.recvSlot, exchange.recvBytes,
                   exchange.slotBytes)) {
      return ncclInvalidArgument;
    }
  }

  ncclResult_t ret = ncclGroupStart();
  if (ret != ncclSuccess) return ret;
  for (size_t i = 0; i < count; ++i) {
    const cocclFrameExchange& exchange = exchanges[i];
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
