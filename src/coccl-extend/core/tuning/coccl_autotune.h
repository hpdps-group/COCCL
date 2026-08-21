#ifndef COCCL_AUTOTUNE_H_
#define COCCL_AUTOTUNE_H_

#include "nccl.h"
#include "runtime/coccl_operation.h"

struct cocclCompressorPlugin;
struct cocclPreparedCall;

// Linear latency model: time_us = alpha_us + beta_us_per_byte * bytes.
struct cocclLinearModel {
  double alphaUs = 0.0;
  double betaUsPerByte = 0.0;
  bool valid = false;
};

struct cocclCodecModel {
  cocclLinearModel time;
  double compressionRatio = 1.0;
  bool valid = false;
};

// Profiling is process-wide and lazy. Disabled autotune returns immediately.
ncclResult_t cocclAutotuneRegisterEnabledCompressor(
    void* compressor, cocclPolicyKey policy);
ncclResult_t cocclAutotuneEnsureGlobalModels(ncclComm_t measurementComm);

// Selects and commits the algorithm, policy, and compressor for one call.
ncclResult_t cocclSelectAlgorithm(cocclPreparedCall* prepared);

#endif
