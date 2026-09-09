#ifndef COCCL_BUFFER_INTERNAL_H_
#define COCCL_BUFFER_INTERNAL_H_

#include "core/memory/coccl_buffer_management.h"

#include "checks.h"
#include "comm.h"

#include <cuda.h>
#include <iterator>
#include <list>
#include <map>
#include <memory>
#include <vector>

namespace coccl_buffer {

constexpr size_t kBufferAlignment = 256;

enum class SliceState {
  Free,
  InUse,
  Pending,
};

enum class BufferBackend {
  Legacy,
  Vmm,
};

struct BufferBlock {
  explicit BufferBlock(BufferBackend value) : backend(value) {}
  BufferBackend backend;
};

inline size_t alignUp(size_t value) {
  return (value + kBufferAlignment - 1) / kBufferAlignment *
      kBufferAlignment;
}

struct BufferSlice {
  size_t offset = 0;
  size_t bytes = 0;
  SliceState state = SliceState::Free;
  cudaEvent_t doneEvent = nullptr;
  cudaStream_t pendingStream = nullptr;
};

inline bool reusable(BufferSlice* slice) {
  if (slice->state == SliceState::Free) return true;
  if (slice->state != SliceState::Pending) return false;

  const cudaError_t status = cudaEventQuery(slice->doneEvent);
  if (status == cudaSuccess) {
    slice->state = SliceState::Free;
    slice->pendingStream = nullptr;
    return true;
  }
  return false;
}

inline bool reusableOnStream(BufferSlice* slice, cudaStream_t stream,
                              bool allowPendingReuse) {
  if (allowPendingReuse && slice->state == SliceState::Pending &&
      slice->pendingStream == stream) {
    return true;
  }
  return reusable(slice);
}

inline void mergeFreeSlices(std::list<BufferSlice>& slices) {
  for (auto current = slices.begin(); current != slices.end();) {
    auto next = std::next(current);
    if (next == slices.end()) break;
    if (reusable(&*current) && reusable(&*next) &&
        current->offset + current->bytes == next->offset) {
      current->bytes += next->bytes;
      if (next->doneEvent != nullptr) {
        CUDACHECKIGNORE(cudaEventDestroy(next->doneEvent));
      }
      slices.erase(next);
    } else {
      ++current;
    }
  }
}

// Same-stream pending reuse keeps the full slice until its event completes.
inline void splitFreeSlice(std::list<BufferSlice>& slices,
                            std::list<BufferSlice>::iterator slice,
                            size_t bytes) {
  if (slice->bytes > bytes && slice->state != SliceState::Pending) {
    BufferSlice remainder;
    remainder.offset = slice->offset + bytes;
    remainder.bytes = slice->bytes - bytes;
    slice->bytes = bytes;
    slices.insert(std::next(slice), remainder);
  }
}

inline ncclResult_t releaseSlice(BufferSlice* slice, cudaStream_t stream) {
  if (slice->doneEvent == nullptr) {
    CUDACHECK(cudaEventCreateWithFlags(&slice->doneEvent,
                                       cudaEventDisableTiming));
  }
  CUDACHECK(cudaEventRecord(slice->doneEvent, stream));
  slice->state = SliceState::Pending;
  slice->pendingStream = stream;
  return ncclSuccess;
}

struct BufferRegistration {
  ncclComm_t comm = nullptr;
  void* ptr = nullptr;
  size_t bytes = 0;
  void* handle = nullptr;
  ncclWindow_t window = nullptr;
  cocclBufferRegistrationKind kind =
      cocclBufferRegistrationKind::Ordinary;
};

struct LegacyBlock : BufferBlock {
  LegacyBlock() : BufferBlock(BufferBackend::Legacy) {}
  int cudaDev = -1;
  void* ptr = nullptr;
  size_t capacity = 0;
  std::list<BufferSlice> slices;
  std::map<ncclComm_t, BufferRegistration> registrations;
};

#if CUDART_VERSION >= 11030

struct VmmSegment {
  CUmemGenericAllocationHandle handle = 0;
  size_t bytes = 0;
};

struct VmmBlock : BufferBlock {
  VmmBlock() : BufferBlock(BufferBackend::Vmm) {}
  struct VmmPool* pool = nullptr;
  int cudaDev = -1;
  void* ptr = nullptr;
  size_t capacity = 0;
  std::vector<VmmSegment> segments;
  std::list<BufferSlice> slices;
  std::vector<BufferRegistration> registrations;
};

struct VmmPool {
  ncclComm_t ownerComm = nullptr;
  int cudaDev = -1;
  CUdevice cuDev = 0;
  size_t granularity = 0;
  size_t chunkBytes = 0;
  size_t virtualBytes = 0;
  size_t physicalBytes = 0;
  size_t registeredBytes = 0;
  bool gpuDirectRdma = false;
  std::list<std::unique_ptr<VmmBlock>> blocks;
};

#endif

struct CommBufferPool {
  ncclComm_t ownerComm = nullptr;
  int cudaDev = -1;
  BufferBackend backend = BufferBackend::Legacy;
  size_t totalBytes = 0;
  std::list<std::unique_ptr<LegacyBlock>> blocks;
#if CUDART_VERSION >= 11030
  VmmPool vmm;
#endif
};

ncclResult_t registerBuffer(ncclComm_t comm, void* ptr, size_t bytes,
                            cocclBufferRegistrationKind requested,
                            BufferRegistration* registration);
ncclResult_t upgradeRegistration(
    BufferRegistration* registration,
    cocclBufferRegistrationKind requested);
ncclResult_t deregisterBuffer(BufferRegistration* registration);
bool registrationSatisfies(const BufferRegistration& registration,
                           cocclBufferRegistrationKind requested);

ncclResult_t legacyAcquire(CommBufferPool* pool, ncclComm_t registeredComm,
                           cocclBufferRegistrationKind registration,
                           size_t bytes, cudaStream_t stream,
                           cocclBufferHandle* buffer);
ncclResult_t legacyRegister(cocclBufferHandle* buffer,
                            ncclComm_t registeredComm,
                            cocclBufferRegistrationKind registration);
ncclResult_t legacyRelease(cocclBufferHandle* buffer, cudaStream_t stream);
ncclResult_t legacyDeregisterComm(CommBufferPool* pool, ncclComm_t comm);
ncclResult_t legacyDestroy(CommBufferPool* pool);

#if CUDART_VERSION >= 11030

ncclResult_t vmmInit(VmmPool* pool, ncclComm_t ownerComm,
                     bool* available);
ncclResult_t vmmAcquire(VmmPool* pool, ncclComm_t registeredComm,
                        cocclBufferRegistrationKind registration,
                        size_t bytes, cudaStream_t stream,
                        cocclBufferHandle* buffer);
ncclResult_t vmmRegister(cocclBufferHandle* buffer,
                         ncclComm_t registeredComm,
                         cocclBufferRegistrationKind registration);
ncclResult_t vmmRelease(cocclBufferHandle* buffer, cudaStream_t stream);
ncclResult_t vmmDeregisterComm(VmmPool* pool, ncclComm_t comm);
ncclResult_t vmmDestroy(VmmPool* pool);

#endif

}  // namespace coccl_buffer

#endif
