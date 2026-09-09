#ifndef COCCL_PREPARED_CALL_H_
#define COCCL_PREPARED_CALL_H_

#include <array>

#include "runtime/coccl_runtime.h"

enum cocclAlgorithmKind {
  cocclAlgorithmNone = 0,
  cocclAlgorithmAllGatherOneShot,
  cocclAlgorithmAllGatherTwoShot,
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

  void* get(cocclCompressionScope scope) const {
    return handles[static_cast<size_t>(scope)];
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
  cocclAlgorithmKind algorithm = cocclAlgorithmNone;
  cocclPreparedCompressorSet compressors;
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
