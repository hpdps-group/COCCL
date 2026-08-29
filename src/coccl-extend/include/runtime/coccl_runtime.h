#ifndef COCCL_RUNTIME_H_
#define COCCL_RUNTIME_H_

#include "runtime/coccl_operation.h"
#include "device.h"
#include "nccl.h"

// Parameters captured at the public NCCL boundary. Runtime routing and
// grouped replay use this single representation.
struct cocclInfo {
  const void* sendbuff = nullptr;
  void* recvbuff = nullptr;
  size_t count = 0;
  ncclDataType_t datatype = ncclNumTypes;
  ncclRedOp_t op = ncclSum;
  int peer = 0;
  ncclFunc_t func = ncclNumFuncs;
  cocclOperation operation = cocclOperation::Count;
  ncclComm_t comm = nullptr;
  cudaStream_t stream = nullptr;
  uint64_t profilerTag = 0;
};

ncclResult_t cocclEnqueueCheck(const cocclInfo* info, bool* isEnqueued);
ncclResult_t cocclReplayNativeCall(const cocclInfo& info);

#endif
