#include "core/memory/coccl_buffer_internal.h"

#if CUDART_VERSION >= 11030

#include "checks.h"
#include "core/config/coccl_config.h"
#include "cudawrap.h"

#include <algorithm>
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

bool canGrow(VmmBlock* block) {
  for (BufferSlice& slice : block->slices) {
    if (slice.state == SliceState::InUse) return false;
  }
  return true;
}

ncclResult_t waitForBlock(VmmBlock* block) {
  for (BufferSlice& slice : block->slices) {
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
    VmmPool* pool, size_t bytes, CUmemGenericAllocationHandle* handle) {
  CUmemAllocationProp prop = {};
  buildAllocationProp(pool, &prop);
  CUCHECK(cuMemCreate(handle, bytes, &prop, 0));
  pool->physicalBytes += bytes;
  return ncclSuccess;
}

ncclResult_t deregisterAll(VmmBlock* block) {
  for (BufferRegistration& registration : block->registrations) {
    NCCLCHECK(deregisterBuffer(&registration));
    block->pool->registeredBytes -= registration.bytes;
  }
  block->registrations.clear();
  return ncclSuccess;
}

ncclResult_t ensureRegistration(
    VmmBlock* block, ncclComm_t comm,
    cocclBufferRegistrationKind requested) {
  if (comm == nullptr) return ncclSuccess;
  for (BufferRegistration& registration : block->registrations) {
    if (registration.comm == comm) {
      return upgradeRegistration(&registration, requested);
    }
  }

  BufferRegistration registration;
  NCCLCHECK(registerBuffer(comm, block->ptr, block->capacity, requested,
                           &registration));
  block->registrations.push_back(registration);
  block->pool->registeredBytes += block->capacity;
  INFO(COCCL_MEMORY,
       "COCCL VMM registration comm %p bytes %zu total %zu", comm,
       block->capacity, block->pool->registeredBytes);
  return ncclSuccess;
}

void resetSlices(VmmBlock* block) {
  for (BufferSlice& slice : block->slices) {
    if (slice.doneEvent != nullptr) {
      CUDACHECKIGNORE(cudaEventDestroy(slice.doneEvent));
    }
  }
  block->slices.clear();
  BufferSlice slice;
  slice.bytes = block->capacity;
  block->slices.push_back(slice);
}

ncclResult_t createBlock(VmmPool* pool, size_t requested,
                         VmmBlock** result) {
  ncclResult_t ret = ncclSuccess;
  const size_t capacity = alignUpTo(requested, pool->chunkBytes);
  bool mapped = false;
  CUdeviceptr reservation = 0;
  std::unique_ptr<VmmBlock> block(new VmmBlock());
  block->pool = pool;
  block->cudaDev = pool->cudaDev;
  block->capacity = capacity;

  CUCHECKGOTO(cuMemAddressReserve(&reservation, capacity, pool->granularity,
                                   0, 0), ret, fail);
  block->ptr = reinterpret_cast<void*>(reservation);
  NCCLCHECKGOTO(createPhysicalHandle(pool, capacity, &block->handle), ret,
                fail);
  CUCHECKGOTO(cuMemMap(reservation, capacity, 0, block->handle, 0), ret,
               fail);
  mapped = true;
  NCCLCHECKGOTO(setPeerAccess(pool, reservation, capacity), ret, fail);

  resetSlices(block.get());
  pool->virtualBytes += capacity;
  *result = block.get();
  pool->blocks.push_back(std::move(block));
  INFO(COCCL_MEMORY,
       "COCCL VMM allocation comm %p requested %zu reserved %zu virtual %zu physical %zu registered %zu",
       pool->ownerComm, requested, capacity, pool->virtualBytes,
       pool->physicalBytes, pool->registeredBytes);
  return ncclSuccess;

fail:
  if (reservation != 0 && mapped) {
    CUCHECKIGNORE(cuMemUnmap(reservation, capacity));
  }
  if (block->handle != 0) {
    CUCHECKIGNORE(cuMemRelease(block->handle));
    pool->physicalBytes -= capacity;
  }
  if (reservation != 0) CUCHECKIGNORE(cuMemAddressFree(reservation, capacity));
  return ret;
}

ncclResult_t growBlock(VmmBlock* block, size_t requested) {
  VmmPool* pool = block->pool;
  const size_t oldCapacity = block->capacity;
  const size_t newCapacity = alignUpTo(requested, pool->chunkBytes);
  ncclResult_t ret = ncclSuccess;
  CUdeviceptr newReservation = 0;
  CUmemGenericAllocationHandle newHandle = 0;
  bool newMapped = false;

  CUDACHECK(cudaSetDevice(block->cudaDev));
  NCCLCHECK(waitForBlock(block));
  CUCHECKGOTO(cuMemAddressReserve(&newReservation, newCapacity,
                                  pool->granularity, 0, 0), ret, fail);
  NCCLCHECKGOTO(createPhysicalHandle(pool, newCapacity, &newHandle), ret,
                fail);
  CUCHECKGOTO(cuMemMap(newReservation, newCapacity, 0, newHandle, 0), ret,
               fail);
  newMapped = true;
  NCCLCHECKGOTO(setPeerAccess(pool, newReservation, newCapacity), ret, fail);

  NCCLCHECKGOTO(deregisterAll(block), ret, fail);
  CUCHECKGOTO(cuMemUnmap(reinterpret_cast<CUdeviceptr>(block->ptr),
                         oldCapacity), ret, fail);
  CUCHECKGOTO(cuMemRelease(block->handle), ret, fail);
  pool->physicalBytes -= oldCapacity;
  CUCHECKGOTO(cuMemAddressFree(reinterpret_cast<CUdeviceptr>(block->ptr),
                               oldCapacity), ret, fail);

  block->ptr = reinterpret_cast<void*>(newReservation);
  block->capacity = newCapacity;
  block->handle = newHandle;
  pool->virtualBytes += newCapacity - oldCapacity;
  resetSlices(block);
  INFO(COCCL_MEMORY,
       "COCCL VMM growth comm %p requested %zu reserved %zu virtual %zu physical %zu registered %zu",
       pool->ownerComm, requested, newCapacity, pool->virtualBytes,
       pool->physicalBytes, pool->registeredBytes);
  return ncclSuccess;

fail:
  if (newMapped) CUCHECKIGNORE(cuMemUnmap(newReservation, newCapacity));
  if (newHandle != 0) {
    CUCHECKIGNORE(cuMemRelease(newHandle));
    pool->physicalBytes -= newCapacity;
  }
  if (newReservation != 0) {
    CUCHECKIGNORE(cuMemAddressFree(newReservation, newCapacity));
  }
  return ret;
}

ncclResult_t acquireFromBlock(VmmBlock* block, size_t bytes,
                              ncclComm_t registeredComm,
                              cocclBufferRegistrationKind requested,
                              cudaStream_t stream,
                              cocclBufferHandle* buffer) {
  mergeFreeSlices(block->slices);
  auto existing = std::find_if(
      block->registrations.begin(), block->registrations.end(),
      [registeredComm](const BufferRegistration& registration) {
        return registration.comm == registeredComm;
      });
  const bool registered = registeredComm == nullptr ||
      existing != block->registrations.end();
  for (auto slice = block->slices.begin(); slice != block->slices.end();
       ++slice) {
    if (!reusableOnStream(&*slice, stream, registered) ||
        slice->bytes < bytes) {
      continue;
    }

    splitFreeSlice(block->slices, slice, bytes);

    slice->state = SliceState::InUse;
    slice->pendingStream = nullptr;
    buffer->ptr = static_cast<char*>(block->ptr) + slice->offset;
    buffer->bytes = slice->bytes;
    buffer->block = block;
    buffer->slice = &*slice;
    ncclResult_t ret = ensureRegistration(block, registeredComm, requested);
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
  for (BufferSlice& slice : block->slices) {
    if (slice.doneEvent != nullptr) {
      CUDACHECK(cudaEventDestroy(slice.doneEvent));
    }
  }
  CUCHECK(cuMemUnmap(reinterpret_cast<CUdeviceptr>(block->ptr),
                     block->capacity));
  CUCHECK(cuMemRelease(block->handle));
  pool->physicalBytes -= block->capacity;
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
                        cocclBufferRegistrationKind registration,
                        size_t bytes, cudaStream_t stream,
                        cocclBufferHandle* buffer) {
  for (auto& block : pool->blocks) {
    ncclResult_t ret = acquireFromBlock(
        block.get(), bytes, registeredComm, registration, stream, buffer);
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
    return acquireFromBlock(
        grow, bytes, registeredComm, registration, stream, buffer);
  }

  VmmBlock* block = nullptr;
  NCCLCHECK(createBlock(pool, bytes, &block));
  return acquireFromBlock(
      block, bytes, registeredComm, registration, stream, buffer);
}

ncclResult_t vmmRegister(cocclBufferHandle* buffer,
                         ncclComm_t registeredComm,
                         cocclBufferRegistrationKind registration) {
  return ensureRegistration(static_cast<VmmBlock*>(buffer->block),
                            registeredComm, registration);
}

ncclResult_t vmmRelease(cocclBufferHandle* buffer, cudaStream_t stream) {
  BufferSlice* slice = static_cast<BufferSlice*>(buffer->slice);
  VmmBlock* block = static_cast<VmmBlock*>(buffer->block);
  CUDACHECK(cudaSetDevice(block->cudaDev));
  return releaseSlice(slice, stream);
}

ncclResult_t vmmDeregisterComm(VmmPool* pool, ncclComm_t comm) {
  for (auto& block : pool->blocks) {
    for (auto registration = block->registrations.begin();
         registration != block->registrations.end();) {
      if (registration->comm != comm) {
        ++registration;
        continue;
      }
      NCCLCHECK(deregisterBuffer(&*registration));
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
  INFO(COCCL_MEMORY,
       "COCCL VMM release comm %p virtual %zu physical %zu registered %zu remaining_virtual %zu remaining_physical %zu remaining_registered %zu",
       pool->ownerComm, virtualBytes, physicalBytes, registeredBytes,
       pool->virtualBytes, pool->physicalBytes, pool->registeredBytes);
  return ncclSuccess;
}

}  // namespace coccl_buffer

#endif
