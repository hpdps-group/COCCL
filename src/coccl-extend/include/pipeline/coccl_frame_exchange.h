#ifndef COCCL_FRAME_EXCHANGE_H_
#define COCCL_FRAME_EXCHANGE_H_

#include "nccl.h"
#include "compressor_plugin/detail/coccl_compressor_abi.h"

#include <stddef.h>

// One already-sized payload exchange. A zero byte count disables that
// direction, which also lets compressed Send/Recv use the batch primitive.
struct cocclFrameExchange {
  int peer;
  const void* sendSlot;
  void* recvSlot;
  size_t sendBytes;
  size_t recvBytes;
  size_t slotBytes;
  ncclComm_t comm;
  cudaStream_t stream;
};

bool cocclFrameMetadataValid(
    const cocclCompressorFrameMetadata& metadata,
    size_t frameStrideBytes);

ncclResult_t cocclBuildAllToAllFrameExchanges(
    const void* sendBase, void* recvBase, size_t frames,
    size_t frameStrideBytes, int nRanks,
    const cocclCompressorFrameMetadata* sendMetadata,
    const cocclCompressorFrameMetadata* recvMetadata,
    cocclFrameExchange* exchanges, size_t exchangeCapacity,
    size_t* exchangeCount);

ncclResult_t cocclBuildAllGatherFrameExchanges(
    const void* sendBase, void* recvBase, size_t localFrames,
    size_t frameStrideBytes, int nRanks,
    const cocclCompressorFrameMetadata* sendMetadata,
    const cocclCompressorFrameMetadata* recvMetadata,
    cocclFrameExchange* exchanges, size_t exchangeCapacity,
    size_t* exchangeCount);

// Commits the caller's deterministic descriptor order as one NCCL group.
// An exchange-specific comm selects its matching stream; otherwise the
// defaults supplied to this function are used.
ncclResult_t cocclCommitFrameExchange(
    const cocclFrameExchange* exchanges, size_t count,
    ncclComm_t comm, cudaStream_t stream);

#endif
