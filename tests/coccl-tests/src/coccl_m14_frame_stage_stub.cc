#include "coccl_frame_exchange.h"

ncclResult_t cocclBuildAllToAllFrameExchanges(
    const void*, void*, size_t, size_t, int,
    const cocclCompressorFrameMetadata*,
    const cocclCompressorFrameMetadata*, cocclFrameExchange*, size_t,
    size_t*) {
  return ncclInternalError;
}

ncclResult_t cocclBuildAllGatherFrameExchanges(
    const void*, void*, size_t, size_t, int,
    const cocclCompressorFrameMetadata*,
    const cocclCompressorFrameMetadata*, cocclFrameExchange*, size_t,
    size_t*) {
  return ncclInternalError;
}

ncclResult_t cocclCommitFrameExchange(
    const cocclFrameExchange*, size_t, ncclComm_t, cudaStream_t) {
  return ncclInternalError;
}
