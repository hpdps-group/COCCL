#ifndef COCCL_PREPARED_CALL_H_
#define COCCL_PREPARED_CALL_H_

#include "coccl_runtime.h"

enum cocclAlgorithmKind {
  cocclAlgorithmNone = 0,
  cocclAlgorithmReduceScatterOneShot,
  cocclAlgorithmReduceScatterTwoShot,
  cocclAlgorithmAllReduceOneShot,
  cocclAlgorithmAllReduceTwoShot,
  cocclAlgorithmAllReduceTripleShot,
};

// Fully resolved immutable call retained by grouped replay.
struct cocclPreparedCall {
  cocclInfo info;
  const cocclOperationDescriptor* descriptor = nullptr;
  cocclPolicyKey policy;
  cocclAlgorithmKind algorithm = cocclAlgorithmNone;
  void* compressor = nullptr;
  double oneShotUs = 0.0;
  double twoShotUs = 0.0;
  double tripleShotUs = 0.0;
  bool usedModel = false;
};

ncclResult_t cocclEnqueuePreparedCall(const cocclPreparedCall* prepared);
ncclResult_t cocclExecutePreparedCall(const cocclPreparedCall* prepared);
ncclResult_t cocclEnqueueExplicitCall(
    const cocclInfo* info, cocclAlgorithmKind algorithm);

#endif
