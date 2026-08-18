#ifndef COCCL_FRAME_EXCHANGE_H_
#define COCCL_FRAME_EXCHANGE_H_

#include "compressor_plugin/detail/coccl_compressor_abi.h"
#include "nccl.h"

#include <stddef.h>

// One already-sized byte exchange. Zero disables that direction, allowing
// Send and Recv entries to share the same batch representation.
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

ncclResult_t cocclCommitFrameExchange(
    const cocclFrameExchange* exchanges, size_t count,
    ncclComm_t comm, cudaStream_t stream);

#endif
