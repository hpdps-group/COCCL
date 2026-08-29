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

bool blockSupportsRegistration(
    const VmmBlock* block, ncclComm_t comm) {
  return comm == nullptr || block->segments.size() == 1;
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

ncclResult_t createPhysicalSegment(
    VmmPool* pool, size_t bytes, VmmSegment* segment) {
  CUmemAllocationProp prop = {};
  buildAllocationProp(pool, &prop);
  CUCHECK(cuMemCreate(&segment->handle, bytes, &prop, 0));
  segment->bytes = bytes;
  pool->physicalBytes += bytes;
  return ncclSuccess;
}

ncclResult_t mapSegments(CUdeviceptr base,
                         const std::vector<VmmSegment>& segments,
                         size_t* mappedSegments) {
  size_t offset = 0;
  *mappedSegments = 0;
  for (const VmmSegment& segment : segments) {
    CUCHECK(cuMemMap(base + offset, segment.bytes, 0, segment.handle, 0));
    offset += segment.bytes;
    ++*mappedSegments;
  }
  return ncclSuccess;
}

ncclResult_t unmapSegments(CUdeviceptr base,
                           const std::vector<VmmSegment>& segments) {
  size_t offset = 0;
  for (const VmmSegment& segment : segments) {
    CUCHECK(cuMemUnmap(base + offset, segment.bytes));
    offset += segment.bytes;
  }
  return ncclSuccess;
}

void unmapSegmentsIgnore(CUdeviceptr base,
                         const std::vector<VmmSegment>& segments,
                         size_t count) {
  size_t offset = 0;
  for (size_t i = 0; i < count; ++i) {
    CUCHECKIGNORE(cuMemUnmap(base + offset, segments[i].bytes));
    offset += segments[i].bytes;
  }
}

void releaseSegmentsIgnore(VmmPool* pool,
                           const std::vector<VmmSegment>& segments,
                           size_t first) {
  for (size_t i = first; i < segments.size(); ++i) {
    CUCHECKIGNORE(cuMemRelease(segments[i].handle));
    pool->physicalBytes -= segments[i].bytes;
  }
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
  if (block->segments.size() > 1) {
    INFO(COCCL_MEMORY,
         "COCCL VMM registration comm %p base %p bytes %zu segments %zu kind unregistered",
         comm, block->ptr, block->capacity, block->segments.size());
    return ncclSuccess;
  }
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
       "COCCL VMM registration comm %p base %p bytes %zu segments %zu kind %s total %zu",
       comm, block->ptr, block->capacity, block->segments.size(),
       registration.window != nullptr ? "symmetric" : "ordinary",
       block->pool->registeredBytes);
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
  size_t mappedSegments = 0;
  CUdeviceptr reservation = 0;
  VmmSegment segment;
  std::unique_ptr<VmmBlock> block(new VmmBlock());
  block->pool = pool;
  block->cudaDev = pool->cudaDev;
  block->capacity = capacity;

  CUCHECKGOTO(cuMemAddressReserve(&reservation, capacity, pool->granularity,
                                   0, 0), ret, fail);
  block->ptr = reinterpret_cast<void*>(reservation);
  NCCLCHECKGOTO(createPhysicalSegment(pool, capacity, &segment), ret, fail);
  block->segments.push_back(segment);
  NCCLCHECKGOTO(mapSegments(
                    reservation, block->segments, &mappedSegments), ret,
                fail);
  NCCLCHECKGOTO(setPeerAccess(pool, reservation, capacity), ret, fail);

  resetSlices(block.get());
  pool->virtualBytes += capacity;
  *result = block.get();
  pool->blocks.push_back(std::move(block));
  INFO(COCCL_MEMORY,
       "COCCL VMM allocation comm %p base %p requested %zu reserved %zu segments %zu virtual %zu physical %zu registered %zu",
       pool->ownerComm, reinterpret_cast<void*>(reservation), requested,
       capacity, (*result)->segments.size(), pool->virtualBytes,
       pool->physicalBytes, pool->registeredBytes);
  return ncclSuccess;

fail:
  if (reservation != 0) {
    unmapSegmentsIgnore(reservation, block->segments, mappedSegments);
  }
  releaseSegmentsIgnore(pool, block->segments, 0);
  if (reservation != 0) CUCHECKIGNORE(cuMemAddressFree(reservation, capacity));
  return ret;
}

ncclResult_t growBlock(VmmBlock* block, size_t requested,
                       bool singleSegment) {
  VmmPool* pool = block->pool;
  const size_t oldCapacity = block->capacity;
  const size_t newCapacity = alignUpTo(requested, pool->chunkBytes);
  ncclResult_t ret = ncclSuccess;
  CUdeviceptr newReservation = 0;
  size_t mappedSegments = 0;
  std::vector<VmmSegment> newSegments;
  std::vector<VmmSegment> oldSegments;
  size_t firstNewSegment = 0;

  CUDACHECK(cudaSetDevice(block->cudaDev));
  NCCLCHECK(waitForBlock(block));
  CUCHECKGOTO(cuMemAddressReserve(&newReservation, newCapacity,
                                  pool->granularity, 0, 0), ret, fail);
  if (singleSegment) {
    VmmSegment segment;
    NCCLCHECKGOTO(createPhysicalSegment(pool, newCapacity, &segment), ret,
                  fail);
    newSegments.push_back(segment);
  } else {
    newSegments = block->segments;
    firstNewSegment = newSegments.size();
    for (size_t offset = oldCapacity; offset < newCapacity;
         offset += pool->chunkBytes) {
      VmmSegment segment;
      NCCLCHECKGOTO(createPhysicalSegment(
                        pool,
                        std::min(pool->chunkBytes, newCapacity - offset),
                        &segment), ret, fail);
      newSegments.push_back(segment);
    }
  }
  NCCLCHECKGOTO(mapSegments(newReservation, newSegments, &mappedSegments),
                ret, fail);
  NCCLCHECKGOTO(setPeerAccess(pool, newReservation, newCapacity), ret, fail);

  NCCLCHECKGOTO(deregisterAll(block), ret, fail);
  NCCLCHECKGOTO(unmapSegments(
                    reinterpret_cast<CUdeviceptr>(block->ptr),
                    block->segments), ret, fail);
  CUCHECKGOTO(cuMemAddressFree(reinterpret_cast<CUdeviceptr>(block->ptr),
                               oldCapacity), ret, fail);

  if (singleSegment) oldSegments = std::move(block->segments);
  block->ptr = reinterpret_cast<void*>(newReservation);
  block->capacity = newCapacity;
  block->segments = std::move(newSegments);
  pool->virtualBytes += newCapacity - oldCapacity;
  resetSlices(block);
  if (singleSegment) {
    for (const VmmSegment& segment : oldSegments) {
      CUCHECK(cuMemRelease(segment.handle));
      pool->physicalBytes -= segment.bytes;
    }
  }
  INFO(COCCL_MEMORY,
       "COCCL VMM growth comm %p base %p requested %zu reserved %zu segments %zu virtual %zu physical %zu registered %zu",
       pool->ownerComm, block->ptr, requested, newCapacity,
       block->segments.size(), pool->virtualBytes, pool->physicalBytes,
       pool->registeredBytes);
  return ncclSuccess;

fail:
  if (newReservation != 0) {
    unmapSegmentsIgnore(newReservation, newSegments, mappedSegments);
  }
  releaseSegmentsIgnore(pool, newSegments, firstNewSegment);
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
  mergeFreeSlices(block);
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
  for (VmmSlice& slice : block->slices) {
    if (slice.doneEvent != nullptr) {
      CUDACHECK(cudaEventDestroy(slice.doneEvent));
    }
  }
  NCCLCHECK(unmapSegments(reinterpret_cast<CUdeviceptr>(block->ptr),
                          block->segments));
  for (const VmmSegment& segment : block->segments) {
    CUCHECK(cuMemRelease(segment.handle));
    pool->physicalBytes -= segment.bytes;
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
                        cocclBufferRegistrationKind registration,
                        size_t bytes, cudaStream_t stream,
                        cocclBufferHandle* buffer) {
  for (auto& block : pool->blocks) {
    if (!blockSupportsRegistration(block.get(), registeredComm)) {
      continue;
    }
    ncclResult_t ret = acquireFromBlock(
        block.get(), bytes, registeredComm, registration, stream, buffer);
    if (ret == ncclSuccess) return ncclSuccess;
    if (ret != ncclInProgress) return ret;
  }

  VmmBlock* grow = nullptr;
  const bool growableRegistration = registeredComm == nullptr ||
      registration == cocclBufferRegistrationKind::Symmetric;
  if (growableRegistration) {
    for (auto& block : pool->blocks) {
      if (blockSupportsRegistration(block.get(), registeredComm) &&
          block->capacity < bytes && canGrow(block.get()) &&
          (grow == nullptr || block->capacity > grow->capacity)) {
        grow = block.get();
      }
    }
  }
  if (grow != nullptr) {
    NCCLCHECK(growBlock(grow, bytes, registeredComm != nullptr));
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
