#ifndef COCCL_BACKEND_FRAMED_H_
#define COCCL_BACKEND_FRAMED_H_

#include "core/pipeline/coccl_frame_exchange.h"

bool cocclBackendUseAllGatherV(
    ncclComm_t comm, const cocclFrameExchange* exchanges, size_t count);

ncclResult_t cocclCommitAllGatherVFrameExchange(
    const cocclFrameExchange* exchanges, size_t localFrames,
    ncclComm_t comm, cudaStream_t stream);

ncclResult_t cocclCommitAllToAllRmaFrameExchange(
    const cocclFrameExchange* exchanges, size_t frames,
    size_t frameStrideBytes, int rank, ncclWindow_t window,
    size_t outputWindowOffset, ncclWaitSignalDesc_t* waitDescriptors,
    ncclComm_t comm, cudaStream_t stream);

#endif
