#ifndef COCCL_PREPARED_CALL_H_
#define COCCL_PREPARED_CALL_H_

#include <array>

#include "coccl_runtime.h"
#include "coccl_training_assist.h"

enum cocclAlgorithmKind {
  cocclAlgorithmNone = 0,
  cocclAlgorithmReduceScatterOneShot,
  cocclAlgorithmReduceScatterTwoShot,
  cocclAlgorithmAllReduceOneShot,
  cocclAlgorithmAllReduceTwoShot,
  cocclAlgorithmAllReduceTripleShot,
};

struct cocclPreparedCompressorSet {
  std::array<void*, static_cast<size_t>(cocclCompressionScope::Count)>
      handles = {};
  std::array<bool, static_cast<size_t>(cocclCompressionScope::Count)>
      datatypeSupported = {};
  size_t thresholdBytes = 0;

  void* get(cocclCompressionScope scope) const {
    return handles[static_cast<size_t>(scope)];
  }

  bool supports(cocclCompressionScope scope) const {
    const size_t index = static_cast<size_t>(scope);
    return handles[index] != nullptr && datatypeSupported[index];
  }

  bool anyEnabled() const {
    for (void* handle : handles) {
      if (handle != nullptr) return true;
    }
    return false;
  }
};

// Fully resolved immutable call retained by grouped replay.
struct cocclPreparedCall {
  cocclInfo info;
  const cocclOperationDescriptor* descriptor = nullptr;
  cocclTrainingRole trainingRole = cocclTrainingRoleUnknown;
  cocclPolicyKey policy;
  cocclAlgorithmKind algorithm = cocclAlgorithmNone;
  cocclPreparedCompressorSet compressors;
  double oneShotUs = 0.0;
  double twoShotUs = 0.0;
  double tripleShotUs = 0.0;
  bool usedModel = false;
};

ncclResult_t cocclEnqueuePreparedCall(const cocclPreparedCall* prepared);
ncclResult_t cocclExecutePreparedCall(const cocclPreparedCall* prepared);
bool cocclPreparedAlgorithmHasCompression(
    const cocclPreparedCall* prepared, cocclAlgorithmKind algorithm);
bool cocclPreparedAlgorithmSupported(
    const cocclPreparedCall* prepared, cocclAlgorithmKind algorithm);
ncclResult_t cocclEnqueueExplicitCall(
    const cocclInfo* info, cocclAlgorithmKind algorithm);

#endif
