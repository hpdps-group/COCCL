#ifndef COCCL_BUFFER_MANAGEMENT_H_
#define COCCL_BUFFER_MANAGEMENT_H_

#include <stddef.h>

#include "nccl.h"
#include "argcheck.h"

// COCCL-private persistent scratch-buffer handle. Primitive code only consumes
// ptr/bytes and returns the handle through cocclReleaseBuffer. The remaining
// fields are opaque metadata used by the buffer manager and its backends.
struct cocclBufferHandle {
  void* ptr = nullptr;
  size_t bytes = 0;
  ncclComm_t ownerComm = nullptr;
  void* block = nullptr;
  void* slice = nullptr;
};

// Create/drop this communicator's view of the per-device pool. Destroy only
// deregisters this comm's handles; idle backing memory stays cached by default.
ncclResult_t cocclBufferCommInit(ncclComm_t comm);
ncclResult_t cocclBufferCommDestroy(ncclComm_t comm);
ncclResult_t cocclBufferDestroyAll();

// Get never waits for GPU work: an unavailable slice causes the pool to grow.
// The stream is therefore needed only by release, where it records completion.
ncclResult_t cocclGetBuffer(ncclComm_t comm, size_t bytes, cocclBufferHandle* buffer);
ncclResult_t cocclGetBufferForComm(ncclComm_t ownerComm, ncclComm_t registeredComm,
                                   size_t bytes, cocclBufferHandle* buffer);
ncclResult_t cocclRegisterBufferForComm(cocclBufferHandle* buffer, ncclComm_t registeredComm);
ncclResult_t cocclReleaseBuffer(cocclBufferHandle* buffer, cudaStream_t stream);

#endif
