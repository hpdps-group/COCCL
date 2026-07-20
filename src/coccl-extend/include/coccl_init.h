#ifndef COCCL_INIT_H_
#define COCCL_INIT_H_

#include "nccl.h"

// COCCL lifecycle entry points used by NCCL communicator initialization and
// teardown. The implementation owns plugin loading, compressor-chain setup,
// buffer-pool registration, and init-time autotuning.
ncclResult_t cocclInit(ncclComm_t comm);
ncclResult_t cocclDestroy(ncclComm_t comm);

// Runtime dispatch reuses the same process-wide enable predicate as init so a
// runtime environment change preserves the existing routing behavior.
bool cocclCompressionEnabled();

#endif
