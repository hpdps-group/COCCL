#ifndef COCCL_RUNTIME_H_
#define COCCL_RUNTIME_H_

#include "compression/coccl_compressor.h"
#include "runtime/coccl_init.h"
#include "runtime/coccl_operation.h"
#include "training/coccl_training_assist.h"
#include "nccl.h"
#include "nccl_common.h"

enum cocclAlgorithmKind {
  cocclAlgorithmNone = 0,
  cocclAlgorithmReduceScatterOneShot,
  cocclAlgorithmReduceScatterTwoShot,
  cocclAlgorithmAllReduceOneShot,
  cocclAlgorithmAllReduceTwoShot,
  cocclAlgorithmAllReduceTripleShot,
};

// Complete user call information passed from NCCL's collective wrappers into
// the private COCCL runtime dispatch layer.
struct cocclInfo {
  const void* sendbuff = nullptr;
  void* recvbuff = nullptr;
  size_t count = 0;
  ncclDataType_t datatype = ncclInt8;
  ncclRedOp_t op = ncclSum;
  int peer = 0;
  ncclFunc_t func = ncclNumFuncs;
  cocclOperation operation = cocclOperation::Count;
  ncclComm_t comm = nullptr;
  cudaStream_t stream = nullptr;
};

// Fully resolved immutable execution plan. This is the only object retained by
// group replay and the only source of compressor policy state during execution.
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

// Attempts COCCL routing and reports whether the call was accepted. Native
// fallback remains in the public collective wrapper when isEnqueued is false.
ncclResult_t cocclEnqueueCheck(const cocclInfo* info, bool* isEnqueued);
ncclResult_t cocclEnqueuePreparedCall(const cocclPreparedCall* prepared);

// Immediate execution hook used by the private group replay subsystem.
ncclResult_t cocclExecutePreparedCall(const cocclPreparedCall* prepared);

#endif
