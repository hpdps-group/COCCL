#ifndef COCCL_AUTOTUNE_H_
#define COCCL_AUTOTUNE_H_

#include "nccl.h"
#include "runtime/coccl_operation.h"

#include <array>
#include <cstddef>

struct cocclCompressorPlugin;
struct cocclPreparedCall;

constexpr size_t kCocclAutotuneMaxProfilePoints = 40;

// The fitted line remains available for compact diagnostics. Predictions use
// adjacent profile points when samples are present so small-message latency is
// not dominated by the GiB-scale end of the profile range.
struct cocclLinearModel {
  double alphaUs = 0.0;
  double betaUsPerByte = 0.0;
  bool valid = false;
  size_t sampleCount = 0;
  std::array<double, kCocclAutotuneMaxProfilePoints> sampleBytes = {};
  std::array<double, kCocclAutotuneMaxProfilePoints> sampleTimeUs = {};
};

struct cocclCodecModel {
  cocclLinearModel time;
  double compressionRatio = 1.0;
  bool valid = false;
  cocclLinearModel drcTime;
};

// Profiling is process-wide and lazy. Disabled autotune returns immediately.
ncclResult_t cocclAutotuneRegisterEnabledCompressor(
    void* compressor, cocclPolicyKey policy);
ncclResult_t cocclAutotuneEnsureGlobalModels(ncclComm_t measurementComm);

// Selects and commits the algorithm, policy, and compressor for one call.
ncclResult_t cocclSelectAlgorithm(cocclPreparedCall* prepared);

#endif
