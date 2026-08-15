#ifndef COCCL_BUFFER_INTERNAL_H_
#define COCCL_BUFFER_INTERNAL_H_

#include "coccl_buffer_management.h"

#include "comm.h"

#include <cuda.h>
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

struct LegacyBlock : BufferBlock {
  LegacyBlock() : BufferBlock(BufferBackend::Legacy) {}
  int cudaDev = -1;
  void* ptr = nullptr;
  size_t capacity = 0;
  std::list<LegacySlice> slices;
  std::map<ncclComm_t, LegacyRegistration> registrations;
};

#if CUDART_VERSION >= 11030

struct VmmBlock;

struct VmmSlice {
  size_t offset = 0;
  size_t bytes = 0;
  SliceState state = SliceState::Free;
  cudaEvent_t doneEvent = nullptr;
  cudaStream_t pendingStream = nullptr;
  VmmBlock* block = nullptr;
};

struct VmmRegistration {
  ncclComm_t comm = nullptr;
  void* ptr = nullptr;
  size_t bytes = 0;
  void* handle = nullptr;
};

struct VmmBlock : BufferBlock {
  VmmBlock() : BufferBlock(BufferBackend::Vmm) {}
  struct VmmPool* pool = nullptr;
  int cudaDev = -1;
  void* ptr = nullptr;
  size_t capacity = 0;
  std::vector<CUmemGenericAllocationHandle> handles;
  std::list<VmmSlice> slices;
  std::vector<VmmRegistration> registrations;
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

ncclResult_t legacyAcquire(CommBufferPool* pool, ncclComm_t registeredComm,
                           size_t bytes, cudaStream_t stream,
                           cocclBufferHandle* buffer);
ncclResult_t legacyRegister(cocclBufferHandle* buffer,
                            ncclComm_t registeredComm);
ncclResult_t legacyRelease(cocclBufferHandle* buffer, cudaStream_t stream);
ncclResult_t legacyDeregisterComm(CommBufferPool* pool, ncclComm_t comm);
ncclResult_t legacyDestroy(CommBufferPool* pool);

#if CUDART_VERSION >= 11030

ncclResult_t vmmInit(VmmPool* pool, ncclComm_t ownerComm,
                     bool* available);
ncclResult_t vmmAcquire(VmmPool* pool, ncclComm_t registeredComm,
                        size_t bytes, cudaStream_t stream,
                        cocclBufferHandle* buffer);
ncclResult_t vmmRegister(cocclBufferHandle* buffer,
                         ncclComm_t registeredComm);
ncclResult_t vmmRelease(cocclBufferHandle* buffer, cudaStream_t stream);
ncclResult_t vmmDeregisterComm(VmmPool* pool, ncclComm_t comm);
ncclResult_t vmmDestroy(VmmPool* pool);

#endif

}  // namespace coccl_buffer

#endif
