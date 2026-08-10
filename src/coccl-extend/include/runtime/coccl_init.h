#ifndef COCCL_INIT_H_
#define COCCL_INIT_H_

#include "nccl.h"

// COCCL lifecycle entry points used by NCCL communicator initialization and
// teardown. The implementation owns plugin loading, compressor-policy setup,
// buffer-pool registration, and init-time autotuning.
ncclResult_t cocclInit(ncclComm_t comm);
ncclResult_t cocclDestroy(ncclComm_t comm);

// Runtime dispatch reuses the immutable process configuration published by
// the first enabled init. Environment changes after that point are ignored.
bool cocclCompressionEnabled();

#endif
