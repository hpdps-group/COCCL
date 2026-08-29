#include "core/pipeline/coccl_frame_exchange.h"

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

ncclResult_t ncclReduceScatterConfig(
    const void*, void*, size_t, ncclDataType_t, ncclRedOp_t,
    ncclComm_t, cudaStream_t, const ncclCollConfig_t*) {
  return ncclInternalError;
}
