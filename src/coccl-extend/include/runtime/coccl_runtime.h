#ifndef COCCL_RUNTIME_H_
#define COCCL_RUNTIME_H_

#include "runtime/coccl_operation.h"
#include "nccl.h"
#include "nccl_common.h"

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

// Attempts COCCL routing and reports whether the call was accepted. Native
// fallback remains in the public collective wrapper when isEnqueued is false.
ncclResult_t cocclEnqueueCheck(const cocclInfo* info, bool* isEnqueued);
ncclResult_t cocclReplayNativeCall(const cocclInfo& info);

#endif
