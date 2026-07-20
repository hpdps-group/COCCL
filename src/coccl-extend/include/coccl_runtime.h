#ifndef COCCL_RUNTIME_H_
#define COCCL_RUNTIME_H_

#include "coccl_init.h"
#include "nccl.h"
#include "nccl_common.h"

struct cocclAlgorithmDecision;

// Argument bundle passed from NCCL's collective wrappers into the private COCCL
// runtime dispatch layer.
struct cocclRuntimeArgs {
  const void* sendbuff;
  void* recvbuff;
  size_t count;
  ncclDataType_t datatype;
  ncclRedOp_t op;
  int peer;
  ncclFunc_t func;
  ncclComm_t comm;
  cudaStream_t stream;
};

// Private runtime entry points used by NCCL's public collective wrappers.
bool cocclAvailable(const cocclRuntimeArgs* args);
ncclResult_t cocclEnqueueCheck(const cocclRuntimeArgs* args);

// Immediate execution hook used by the private group replay subsystem. This
// bypasses group deferral but preserves the recursive COCCL caller guard.
ncclResult_t cocclExecutePrimitive(
    const cocclRuntimeArgs* args, const cocclAlgorithmDecision* decision);

#endif
