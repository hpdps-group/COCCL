#ifndef COCCL_BUFFER_MANAGEMENT_H_
#define COCCL_BUFFER_MANAGEMENT_H_

#include <stddef.h>

#include "nccl.h"

// A workspace lease owned by one NCCL communicator. Primitive code consumes
// ptr/bytes and returns the opaque backend fields through cocclReleaseBuffer.
struct cocclBufferHandle {
  void* ptr = nullptr;
  size_t bytes = 0;
  ncclComm_t ownerComm = nullptr;
  void* block = nullptr;
  void* slice = nullptr;
};

ncclResult_t cocclBufferCommInit(ncclComm_t comm);
ncclResult_t cocclBufferCommDestroy(ncclComm_t comm);

ncclResult_t cocclGetBuffer(ncclComm_t comm, size_t bytes,
                            cudaStream_t stream,
                            cocclBufferHandle* buffer);
ncclResult_t cocclGetBufferForComm(ncclComm_t ownerComm,
                                   ncclComm_t registeredComm, size_t bytes,
                                   cudaStream_t stream,
                                   cocclBufferHandle* buffer);
ncclResult_t cocclGetUnregisteredBuffer(ncclComm_t ownerComm, size_t bytes,
                                        cudaStream_t stream,
                                        cocclBufferHandle* buffer);
ncclResult_t cocclRegisterBufferForComm(cocclBufferHandle* buffer,
                                        ncclComm_t registeredComm);
ncclResult_t cocclReleaseBuffer(cocclBufferHandle* buffer,
                                cudaStream_t stream);

#endif
