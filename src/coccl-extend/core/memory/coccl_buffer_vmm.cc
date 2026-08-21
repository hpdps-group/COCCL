#include "core/memory/coccl_buffer_internal.h"

#if CUDART_VERSION >= 11030

#include "checks.h"
#include "core/config/coccl_config.h"
#include "cudawrap.h"

#include <algorithm>
#include <iterator>
#include <stdint.h>

namespace coccl_buffer {
namespace {

size_t alignUpTo(size_t value, size_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

void buildAllocationProp(VmmPool* pool, CUmemAllocationProp* prop) {
  *prop = {};
  prop->type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop->location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop->location.id = pool->cuDev;
  prop->requestedHandleTypes = ncclCuMemHandleType;
  if (pool->gpuDirectRdma) prop->allocFlags.gpuDirectRDMACapable = 1;
}

bool reusable(VmmSlice* slice) {
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

bool reusableOnStream(VmmSlice* slice, cudaStream_t stream,
                      bool allowPendingReuse) {
  return allowPendingReuse && slice->state == SliceState::Pending &&
          slice->pendingStream == stream
      ? true
      : reusable(slice);
}

void mergeFreeSlices(VmmBlock* block) {
  for (auto current = block->slices.begin(); current != block->slices.end();) {
    auto next = std::next(current);
    if (next == block->slices.end()) break;
    if (reusable(&*current) && reusable(&*next) &&
        current->offset + current->bytes == next->offset) {
      current->bytes += next->bytes;
      if (next->doneEvent != nullptr) {
        CUDACHECKIGNORE(cudaEventDestroy(next->doneEvent));
      }
      block->slices.erase(next);
    } else {
      ++current;
    }
  }
}

bool canGrow(VmmBlock* block) {
  for (VmmSlice& slice : block->slices) {
    if (slice.state == SliceState::InUse) return false;
  }
  return true;
}

ncclResult_t waitForBlock(VmmBlock* block) {
  for (VmmSlice& slice : block->slices) {
    if (slice.state == SliceState::Pending) {
      CUDACHECK(cudaEventSynchronize(slice.doneEvent));
      slice.state = SliceState::Free;
      slice.pendingStream = nullptr;
    }
  }
  return ncclSuccess;
}

ncclResult_t setPeerAccess(VmmPool* pool, CUdeviceptr ptr, size_t bytes) {
  int deviceCount = 0;
  CUDACHECK(cudaGetDeviceCount(&deviceCount));
  for (int dev = 0; dev < deviceCount; ++dev) {
    int canAccess = dev == pool->cudaDev;
    if (!canAccess) {
      CUDACHECK(cudaDeviceCanAccessPeer(&canAccess, dev, pool->cudaDev));
    }
    if (!canAccess) continue;

    CUmemAccessDesc access = {};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = dev;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CUCHECK(cuMemSetAccess(ptr, bytes, &access, 1));
  }
  return ncclSuccess;
}

ncclResult_t createPhysicalHandle(
    VmmPool* pool, CUmemGenericAllocationHandle* handle) {
  CUmemAllocationProp prop = {};
  buildAllocationProp(pool, &prop);
  CUCHECK(cuMemCreate(handle, pool->chunkBytes, &prop, 0));
  pool->physicalBytes += pool->chunkBytes;
  return ncclSuccess;
}

ncclResult_t deregisterAll(VmmBlock* block) {
  for (VmmRegistration& registration : block->registrations) {
    NCCLCHECK(ncclCommDeregister(registration.comm, registration.handle));
    block->pool->registeredBytes -= registration.bytes;
  }
  block->registrations.clear();
  return ncclSuccess;
}

ncclResult_t ensureRegistration(VmmBlock* block, ncclComm_t comm) {
  if (comm == nullptr) return ncclSuccess;
  for (const VmmRegistration& registration : block->registrations) {
    if (registration.comm == comm) return ncclSuccess;
  }

  void* handle = nullptr;
  NCCLCHECK(ncclCommRegister(comm, block->ptr, block->capacity, &handle));
  block->registrations.push_back(
      VmmRegistration{comm, block->ptr, block->capacity, handle});
  block->pool->registeredBytes += block->capacity;
  INFO(NCCL_INIT,
       "COCCL VMM registration comm %p bytes %zu total %zu", comm,
       block->capacity, block->pool->registeredBytes);
  return ncclSuccess;
}

void resetSlices(VmmBlock* block) {
  for (VmmSlice& slice : block->slices) {
    if (slice.doneEvent != nullptr) {
      CUDACHECKIGNORE(cudaEventDestroy(slice.doneEvent));
    }
  }
  block->slices.clear();
  VmmSlice slice;
  slice.bytes = block->capacity;
  slice.block = block;
  block->slices.push_back(slice);
}

ncclResult_t createBlock(VmmPool* pool, size_t requested,
                         VmmBlock** result) {
  ncclResult_t ret = ncclSuccess;
  const size_t capacity = alignUpTo(requested, pool->chunkBytes);
  const size_t chunks = capacity / pool->chunkBytes;
  size_t mapped = 0;
  CUdeviceptr reservation = 0;
  std::unique_ptr<VmmBlock> block(new VmmBlock());
  block->pool = pool;
  block->cudaDev = pool->cudaDev;
  block->capacity = capacity;
  block->handles.reserve(chunks);

  CUCHECKGOTO(cuMemAddressReserve(&reservation, capacity, pool->granularity,
                                   0, 0), ret, fail);
  block->ptr = reinterpret_cast<void*>(reservation);
  for (size_t i = 0; i < chunks; ++i) {
    CUmemGenericAllocationHandle handle = 0;
    NCCLCHECKGOTO(createPhysicalHandle(pool, &handle), ret, fail);
    block->handles.push_back(handle);
    CUCHECKGOTO(cuMemMap(reservation + i * pool->chunkBytes,
                         pool->chunkBytes, 0, handle, 0), ret, fail);
    ++mapped;
  }
  NCCLCHECKGOTO(setPeerAccess(pool, reservation, capacity), ret, fail);

  resetSlices(block.get());
  pool->virtualBytes += capacity;
  *result = block.get();
  pool->blocks.push_back(std::move(block));
  INFO(NCCL_INIT,
       "COCCL VMM allocation comm %p requested %zu reserved %zu virtual %zu physical %zu registered %zu",
       pool->ownerComm, requested, capacity, pool->virtualBytes,
       pool->physicalBytes, pool->registeredBytes);
  return ncclSuccess;

fail:
  if (reservation != 0 && mapped != 0) {
    CUCHECKIGNORE(cuMemUnmap(reservation, mapped * pool->chunkBytes));
  }
  for (CUmemGenericAllocationHandle handle : block->handles) {
    CUCHECKIGNORE(cuMemRelease(handle));
    pool->physicalBytes -= pool->chunkBytes;
  }
  if (reservation != 0) CUCHECKIGNORE(cuMemAddressFree(reservation, capacity));
  return ret;
}

ncclResult_t growBlock(VmmBlock* block, size_t requested) {
  VmmPool* pool = block->pool;
  const size_t oldCapacity = block->capacity;
  const size_t newCapacity = alignUpTo(requested, pool->chunkBytes);
  const size_t oldChunks = block->handles.size();
  const size_t newChunks = newCapacity / pool->chunkBytes;
  CUdeviceptr newReservation = 0;

  CUDACHECK(cudaSetDevice(block->cudaDev));
  NCCLCHECK(waitForBlock(block));
  CUCHECK(cuMemAddressReserve(&newReservation, newCapacity,
                              pool->granularity, 0, 0));

  for (size_t i = oldChunks; i < newChunks; ++i) {
    CUmemGenericAllocationHandle handle = 0;
    ncclResult_t ret = createPhysicalHandle(pool, &handle);
    if (ret != ncclSuccess) {
      CUCHECKIGNORE(cuMemAddressFree(newReservation, newCapacity));
      return ret;
    }
    block->handles.push_back(handle);
  }

  NCCLCHECK(deregisterAll(block));
  CUCHECK(cuMemUnmap(reinterpret_cast<CUdeviceptr>(block->ptr), oldCapacity));
  for (size_t i = 0; i < newChunks; ++i) {
    CUCHECK(cuMemMap(newReservation + i * pool->chunkBytes,
                     pool->chunkBytes, 0, block->handles[i], 0));
  }
  NCCLCHECK(setPeerAccess(pool, newReservation, newCapacity));
  CUCHECK(cuMemAddressFree(reinterpret_cast<CUdeviceptr>(block->ptr),
                           oldCapacity));

  block->ptr = reinterpret_cast<void*>(newReservation);
  block->capacity = newCapacity;
  pool->virtualBytes += newCapacity - oldCapacity;
  resetSlices(block);
  INFO(NCCL_INIT,
       "COCCL VMM growth comm %p requested %zu reserved %zu virtual %zu physical %zu registered %zu",
       pool->ownerComm, requested, newCapacity, pool->virtualBytes,
       pool->physicalBytes, pool->registeredBytes);
  return ncclSuccess;
}

ncclResult_t acquireFromBlock(VmmBlock* block, size_t bytes,
                              ncclComm_t registeredComm,
                              cudaStream_t stream,
                              cocclBufferHandle* buffer) {
  mergeFreeSlices(block);
  const bool registered = registeredComm == nullptr ||
      std::any_of(block->registrations.begin(), block->registrations.end(),
                  [registeredComm](const VmmRegistration& registration) {
                    return registration.comm == registeredComm;
                  });
  for (auto slice = block->slices.begin(); slice != block->slices.end();
       ++slice) {
    if (!reusableOnStream(&*slice, stream, registered) ||
        slice->bytes < bytes) {
      continue;
    }

    const bool pendingReuse = slice->state == SliceState::Pending;
    if (slice->bytes > bytes && !pendingReuse) {
      VmmSlice remainder;
      remainder.offset = slice->offset + bytes;
      remainder.bytes = slice->bytes - bytes;
      remainder.block = block;
      slice->bytes = bytes;
      block->slices.insert(std::next(slice), remainder);
    }

    slice->state = SliceState::InUse;
    slice->pendingStream = nullptr;
    buffer->ptr = static_cast<char*>(block->ptr) + slice->offset;
    buffer->bytes = slice->bytes;
    buffer->ownerComm = block->pool->ownerComm;
    buffer->block = block;
    buffer->slice = &*slice;
    ncclResult_t ret = ensureRegistration(block, registeredComm);
    if (ret == ncclSuccess) return ncclSuccess;
    slice->state = SliceState::Free;
    *buffer = {};
    return ret;
  }
  return ncclInProgress;
}

ncclResult_t releaseBlock(VmmBlock* block) {
  VmmPool* pool = block->pool;
  NCCLCHECK(deregisterAll(block));
  for (VmmSlice& slice : block->slices) {
    if (slice.doneEvent != nullptr) {
      CUDACHECK(cudaEventDestroy(slice.doneEvent));
    }
  }
  CUCHECK(cuMemUnmap(reinterpret_cast<CUdeviceptr>(block->ptr),
                     block->capacity));
  for (CUmemGenericAllocationHandle handle : block->handles) {
    CUCHECK(cuMemRelease(handle));
    pool->physicalBytes -= pool->chunkBytes;
  }
  CUCHECK(cuMemAddressFree(reinterpret_cast<CUdeviceptr>(block->ptr),
                           block->capacity));
  pool->virtualBytes -= block->capacity;
  return ncclSuccess;
}

}  // namespace

ncclResult_t vmmInit(VmmPool* pool, ncclComm_t ownerComm,
                     bool* available) {
  *available = false;
  pool->ownerComm = ownerComm;
  pool->cudaDev = ownerComm->cudaDev;
  if (ncclCudaLibraryInit() != ncclSuccess || !ncclCuMemEnable() ||
      CUPFN(cuDeviceGet) == nullptr ||
      CUPFN(cuDeviceGetAttribute) == nullptr ||
      CUPFN(cuMemAddressReserve) == nullptr ||
      CUPFN(cuMemAddressFree) == nullptr || CUPFN(cuMemCreate) == nullptr ||
      CUPFN(cuMemGetAllocationGranularity) == nullptr ||
      CUPFN(cuMemMap) == nullptr || CUPFN(cuMemRelease) == nullptr ||
      CUPFN(cuMemSetAccess) == nullptr || CUPFN(cuMemUnmap) == nullptr) {
    return ncclSuccess;
  }

  ncclResult_t ret = ncclSuccess;
  int vmmSupported = 0;
  int rdmaVmmSupported = 0;
  int rdmaSupported = 0;
  CUmemAllocationProp prop = {};
  size_t granularity = 0;
  size_t configuredChunkBytes = cocclBufferConfig{}.physicalChunkBytes;
  CUDACHECKGOTO(cudaSetDevice(pool->cudaDev), ret, unavailable);
  CUCHECKGOTO(cuDeviceGet(&pool->cuDev, pool->cudaDev), ret, unavailable);
  CUCHECKGOTO(cuDeviceGetAttribute(
                  &vmmSupported,
                  CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
                  pool->cuDev), ret, unavailable);
  CUCHECKGOTO(cuDeviceGetAttribute(
                  &rdmaVmmSupported,
                  CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED,
                  pool->cuDev), ret, unavailable);
  if (!vmmSupported || !rdmaVmmSupported) goto unavailable;
  CUCHECKGOTO(cuDeviceGetAttribute(
                  &rdmaSupported,
                  CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED, pool->cuDev),
              ret, unavailable);
  pool->gpuDirectRdma = rdmaSupported != 0;
  buildAllocationProp(pool, &prop);
  CUCHECKGOTO(cuMemGetAllocationGranularity(
                  &granularity, &prop,
                  CU_MEM_ALLOC_GRANULARITY_RECOMMENDED),
              ret, unavailable);
  pool->granularity = granularity;
  if (cocclConfigInitialize()) {
    configuredChunkBytes = cocclGetConfig().buffer.physicalChunkBytes;
  }
  pool->chunkBytes = alignUpTo(
      std::max(configuredChunkBytes, granularity), granularity);
  *available = true;
  return ncclSuccess;

unavailable:
  (void)ret;
  *available = false;
  return ncclSuccess;
}

ncclResult_t vmmAcquire(VmmPool* pool, ncclComm_t registeredComm,
                        size_t bytes, cudaStream_t stream,
                        cocclBufferHandle* buffer) {
  for (auto& block : pool->blocks) {
    ncclResult_t ret = acquireFromBlock(block.get(), bytes, registeredComm,
                                        stream, buffer);
    if (ret == ncclSuccess) return ncclSuccess;
    if (ret != ncclInProgress) return ret;
  }

  VmmBlock* grow = nullptr;
  for (auto& block : pool->blocks) {
    if (block->capacity < bytes && canGrow(block.get()) &&
        (grow == nullptr || block->capacity > grow->capacity)) {
      grow = block.get();
    }
  }
  if (grow != nullptr) {
    NCCLCHECK(growBlock(grow, bytes));
    return acquireFromBlock(grow, bytes, registeredComm, stream, buffer);
  }

  VmmBlock* block = nullptr;
  NCCLCHECK(createBlock(pool, bytes, &block));
  return acquireFromBlock(block, bytes, registeredComm, stream, buffer);
}

ncclResult_t vmmRegister(cocclBufferHandle* buffer,
                         ncclComm_t registeredComm) {
  return ensureRegistration(static_cast<VmmBlock*>(buffer->block),
                            registeredComm);
}

ncclResult_t vmmRelease(cocclBufferHandle* buffer, cudaStream_t stream) {
  VmmSlice* slice = static_cast<VmmSlice*>(buffer->slice);
  VmmBlock* block = static_cast<VmmBlock*>(buffer->block);
  CUDACHECK(cudaSetDevice(block->cudaDev));
  if (slice->doneEvent == nullptr) {
    CUDACHECK(cudaEventCreateWithFlags(&slice->doneEvent,
                                       cudaEventDisableTiming));
  }
  CUDACHECK(cudaEventRecord(slice->doneEvent, stream));
  slice->state = SliceState::Pending;
  slice->pendingStream = stream;
  return ncclSuccess;
}

ncclResult_t vmmDeregisterComm(VmmPool* pool, ncclComm_t comm) {
  for (auto& block : pool->blocks) {
    for (auto registration = block->registrations.begin();
         registration != block->registrations.end();) {
      if (registration->comm != comm) {
        ++registration;
        continue;
      }
      NCCLCHECK(ncclCommDeregister(comm, registration->handle));
      pool->registeredBytes -= registration->bytes;
      registration = block->registrations.erase(registration);
    }
  }
  return ncclSuccess;
}

ncclResult_t vmmDestroy(VmmPool* pool) {
  const size_t virtualBytes = pool->virtualBytes;
  const size_t physicalBytes = pool->physicalBytes;
  const size_t registeredBytes = pool->registeredBytes;
  CUDACHECK(cudaSetDevice(pool->cudaDev));
  CUDACHECK(cudaDeviceSynchronize());
  for (auto& block : pool->blocks) NCCLCHECK(releaseBlock(block.get()));
  pool->blocks.clear();
  INFO(NCCL_INIT,
       "COCCL VMM release comm %p virtual %zu physical %zu registered %zu remaining_virtual %zu remaining_physical %zu remaining_registered %zu",
       pool->ownerComm, virtualBytes, physicalBytes, registeredBytes,
       pool->virtualBytes, pool->physicalBytes, pool->registeredBytes);
  return ncclSuccess;
}

}  // namespace coccl_buffer

#endif
