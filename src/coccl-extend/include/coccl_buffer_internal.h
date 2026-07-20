#ifndef COCCL_BUFFER_INTERNAL_H_
#define COCCL_BUFFER_INTERNAL_H_

#include "coccl_buffer_management.h"

#include "comm.h"
#include "cudawrap.h"
#include "info.h"

#include <list>
#include <map>
#include <memory>
#include <stdint.h>
#include <vector>

int64_t ncclParamCocclDeviceBufferPoolLimit();
int64_t ncclParamCocclDeviceBufferBlockBytes();
int64_t ncclParamCocclDeviceBufferPhysicalChunkBytes();

namespace coccl_buffer {

// Keep slices aligned to a conservative cacheline-ish boundary. The pool hands
// out byte-addressed scratch regions, so all backend accounting uses bytes.
constexpr size_t kCocclBufferAlignment = 256;

// Slice state machine:
//   Free    : immediately reusable.
//   InUse   : held by a caller; no completion event has been recorded yet.
//   Pending : released by the caller, but reusable only after doneEvent fires.
enum class SliceState {
  Free,
  InUse,
  Pending,
};

enum class BufferBackend {
  Legacy,
  Vmm,
};

struct cocclBufferBlockBase {
  explicit cocclBufferBlockBase(BufferBackend backend) : backend(backend) {}
  // Stored in cocclBufferHandle::block so the common API can dispatch release
  // and registration calls back to the owning backend without exposing types.
  BufferBackend backend;
};

inline size_t alignUp(size_t value, size_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

inline bool rangeContains(uintptr_t outerPtr, size_t outerBytes, uintptr_t innerPtr, size_t innerBytes) {
  return innerPtr >= outerPtr && innerPtr + innerBytes <= outerPtr + outerBytes;
}

inline bool rangeOverlaps(uintptr_t aPtr, size_t aBytes, uintptr_t bPtr, size_t bBytes) {
  return aPtr < bPtr + bBytes && bPtr < aPtr + aBytes;
}

struct cocclLegacyBackingBlock;

struct cocclLegacyBufferSlice {
  // Offset and size inside one ncclMemAlloc backing block.
  size_t offset = 0;
  size_t bytes = 0;
  SliceState state = SliceState::Free;
  // The original NCCL communicator that acquired this slice. It is used during
  // cocclBufferCommDestroy to drain and release only that comm's outstanding
  // buffers, even when transport registration was done on a split subcomm.
  ncclComm_t ownerComm = nullptr;
  cudaEvent_t doneEvent = nullptr;
  cocclLegacyBackingBlock* block = nullptr;
};

struct cocclLegacyBufferRegistration {
  // NCCL registration handle for the whole legacy backing block.
  void* handle = nullptr;
  // Comm whose operation caused this registration. If ownerComm is destroyed,
  // registrations it introduced for split/subcomms must also be deregistered.
  ncclComm_t ownerComm = nullptr;
};

struct cocclLegacyBackingBlock : cocclBufferBlockBase {
  cocclLegacyBackingBlock() : cocclBufferBlockBase(BufferBackend::Legacy) {}
  int cudaDev = -1;
  void* ptr = nullptr;
  size_t capacity = 0;
  // list keeps slice addresses stable while splitting/merging free regions.
  std::list<cocclLegacyBufferSlice> slices;
  // Legacy registration is block-wide because ncclMemAlloc gives one physical
  // allocation and older paths cannot remap smaller registered VA ranges.
  std::map<ncclComm_t, cocclLegacyBufferRegistration> registrations;
};

struct cocclLegacyDeviceBufferPool {
  int cudaDev = -1;
  size_t totalBytes = 0;
  std::list<std::unique_ptr<cocclLegacyBackingBlock>> blocks;
};

#if CUDART_VERSION >= 11030

struct cocclVmmBackingBlock;

struct cocclVmmBufferSlice {
  // Offset and size inside one contiguous virtual-address reservation.
  size_t offset = 0;
  size_t bytes = 0;
  SliceState state = SliceState::Free;
  ncclComm_t ownerComm = nullptr;
  cudaEvent_t doneEvent = nullptr;
  cocclVmmBackingBlock* block = nullptr;
};

struct cocclVmmBufferRegistration {
  // Registration is attached to a specific transport comm and a specific VA
  // range. Different NCCL comms may register the same VA range independently.
  ncclComm_t registeredComm = nullptr;
  ncclComm_t ownerComm = nullptr;
  void* ptr = nullptr;
  size_t bytes = 0;
  void* handle = nullptr;
};

struct cocclVmmBackingBlock : cocclBufferBlockBase {
  cocclVmmBackingBlock() : cocclBufferBlockBase(BufferBackend::Vmm) {}
  int cudaDev = -1;
  CUdevice cuDev = 0;
  // Contiguous VA returned to primitives. Physical chunks behind this VA may
  // be non-contiguous; kernels and NCCL still see a regular device pointer.
  void* ptr = nullptr;
  size_t capacity = 0;
  size_t chunkSize = 0;
  // One handle per mapped physical chunk in this VA reservation.
  std::vector<CUmemGenericAllocationHandle> handles;
  std::list<cocclVmmBufferSlice> slices;
  std::vector<cocclVmmBufferRegistration> registrations;
};

struct cocclVmmDeviceBufferPool {
  int cudaDev = -1;
  CUdevice cuDev = 0;
  size_t granularity = 0;
  size_t chunkSize = 0;
  size_t totalPhysicalBytes = 0;
  bool ready = false;
  bool gpuDirectRdmaCapable = false;
  // Physical chunks detached from unused VA ranges. Reusing handles avoids
  // expensive cuMemCreate calls when future requests need a different VA shape.
  std::vector<CUmemGenericAllocationHandle> freeHandles;
  std::list<std::unique_ptr<cocclVmmBackingBlock>> blocks;
};

#endif

struct cocclDeviceBufferPool {
  int cudaDev = -1;
  bool initialized = false;
  // VMM is preferred when available; legacy remains the compatibility backend.
  bool useVmm = false;
  cocclLegacyDeviceBufferPool legacy;
#if CUDART_VERSION >= 11030
  cocclVmmDeviceBufferPool vmm;
#endif
};

ncclResult_t legacyAcquireForComm(cocclLegacyDeviceBufferPool* pool, ncclComm_t ownerComm,
                                  ncclComm_t registeredComm, size_t bytes,
                                  cocclBufferHandle* buffer);
ncclResult_t legacyRegisterBufferForComm(cocclBufferHandle* buffer, ncclComm_t registeredComm);
ncclResult_t legacyReleaseBuffer(cocclBufferHandle* buffer, cudaStream_t stream);
ncclResult_t legacyReleaseCommRegistrationsLocked(cocclLegacyDeviceBufferPool* pool, ncclComm_t comm);
ncclResult_t legacyDestroyPoolLocked(cocclLegacyDeviceBufferPool* pool);

#if CUDART_VERSION >= 11030
ncclResult_t vmmInitPool(cocclVmmDeviceBufferPool* pool, int cudaDev, bool* available);
ncclResult_t vmmAcquireForComm(cocclVmmDeviceBufferPool* pool, ncclComm_t ownerComm,
                               ncclComm_t registeredComm, size_t bytes,
                               cocclBufferHandle* buffer);
ncclResult_t vmmRegisterBufferForComm(cocclBufferHandle* buffer, ncclComm_t registeredComm);
ncclResult_t vmmReleaseBuffer(cocclBufferHandle* buffer, cudaStream_t stream);
ncclResult_t vmmReleaseCommRegistrationsLocked(cocclVmmDeviceBufferPool* pool, ncclComm_t comm);
ncclResult_t vmmDestroyPoolLocked(cocclVmmDeviceBufferPool* pool);
#endif

}  // namespace coccl_buffer

#endif
