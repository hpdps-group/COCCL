#ifndef COCCL_PREPARED_CALL_H_
#define COCCL_PREPARED_CALL_H_

#include "core/compression/coccl_compressor_runtime.h"
#include "core/training/coccl_training_assist.h"
#include "runtime/coccl_runtime.h"

enum cocclAlgorithmKind {
  cocclAlgorithmNone = 0,
  cocclAlgorithmReduceScatterOneShot,
  cocclAlgorithmReduceScatterTwoShot,
  cocclAlgorithmAllReduceOneShot,
  cocclAlgorithmAllReduceTwoShot,
  cocclAlgorithmAllReduceTripleShot,
};

// Fully resolved immutable execution plan retained by grouped replay.
struct cocclPreparedCall {
  cocclInfo info;
  const cocclOperationDescriptor* descriptor = nullptr;
  cocclTrainingRole trainingRole = cocclTrainingRoleUnknown;
  cocclPolicyKey policy;
  cocclAlgorithmKind algorithm = cocclAlgorithmNone;
  cocclCompressorHandle compressor;
  double oneShotUs = 0.0;
  double twoShotUs = 0.0;
  double tripleShotUs = 0.0;
  bool usedModel = false;
};

ncclResult_t cocclEnqueuePreparedCall(const cocclPreparedCall* prepared);
ncclResult_t cocclExecutePreparedCall(const cocclPreparedCall* prepared);

#endif
