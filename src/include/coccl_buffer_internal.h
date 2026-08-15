#ifndef COCCL_BUFFER_INTERNAL_H_
#define COCCL_BUFFER_INTERNAL_H_

#include "coccl_buffer_management.h"

#include "comm.h"

#include <list>
#include <map>
#include <memory>

namespace coccl_buffer {

constexpr size_t kBufferAlignment = 256;

enum class SliceState {
  Free,
  InUse,
  Pending,
};

inline size_t alignUp(size_t value) {
  return (value + kBufferAlignment - 1) / kBufferAlignment *
      kBufferAlignment;
}

struct LegacyBlock;

struct LegacySlice {
  size_t offset = 0;
  size_t bytes = 0;
  SliceState state = SliceState::Free;
  cudaEvent_t doneEvent = nullptr;
  cudaStream_t pendingStream = nullptr;
  LegacyBlock* block = nullptr;
};

struct LegacyRegistration {
  void* handle = nullptr;
};

struct LegacyBlock {
  int cudaDev = -1;
  void* ptr = nullptr;
  size_t capacity = 0;
  std::list<LegacySlice> slices;
  std::map<ncclComm_t, LegacyRegistration> registrations;
};

struct CommBufferPool {
  ncclComm_t ownerComm = nullptr;
  int cudaDev = -1;
  size_t totalBytes = 0;
  std::list<std::unique_ptr<LegacyBlock>> blocks;
};

ncclResult_t legacyAcquire(CommBufferPool* pool, ncclComm_t registeredComm,
                           size_t bytes, cudaStream_t stream,
                           cocclBufferHandle* buffer);
ncclResult_t legacyRegister(cocclBufferHandle* buffer,
                            ncclComm_t registeredComm);
ncclResult_t legacyRelease(cocclBufferHandle* buffer, cudaStream_t stream);
ncclResult_t legacyDeregisterComm(CommBufferPool* pool, ncclComm_t comm);
ncclResult_t legacyDestroy(CommBufferPool* pool);

}  // namespace coccl_buffer

#endif
