#include "buffer/coccl_buffer_internal.h"

#if CUDART_VERSION >= 11030

namespace coccl_buffer {

static ncclResult_t vmmBuildAllocationProp(cocclVmmDeviceBufferPool* pool, CUmemAllocationProp* prop) {
  *prop = {};
  // Use device-local pinned allocations so the resulting VA can participate in
  // GPUDirect-RDMA registration through NCCL when the device supports it.
  prop->type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop->location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop->requestedHandleTypes = ncclCuMemHandleType;
  prop->location.id = pool->cuDev;
  if (pool->gpuDirectRdmaCapable) prop->allocFlags.gpuDirectRDMACapable = 1;
  return ncclSuccess;
}

static ncclResult_t vmmGetMulticastGranularity(cocclVmmDeviceBufferPool* pool, size_t requestedBytes,
                                               size_t* granularityOut) {
  *granularityOut = 0;

#if CUDART_VERSION >= 12010 && CUDA_VERSION >= 12010
  if (CUPFN(cuMulticastCreate) == nullptr || CUPFN(cuMulticastGetGranularity) == nullptr) {
    return ncclSuccess;
  }

  int multicastSupported = 0;
  CUCHECK(cuDeviceGetAttribute(&multicastSupported, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED,
                               pool->cuDev));
  if (!multicastSupported) return ncclSuccess;

  int deviceCount = 0;
  CUDACHECK(cudaGetDeviceCount(&deviceCount));

  CUmulticastObjectProp mcprop = {};
  mcprop.size = requestedBytes;
  // Match ncclMemAlloc: device count is informational today but may affect the
  // recommended multicast granularity on future platforms.
  mcprop.numDevices = deviceCount;
  mcprop.handleTypes = ncclCuMemHandleType;
  mcprop.flags = 0;
  CUCHECK(cuMulticastGetGranularity(granularityOut, &mcprop,
                                    CU_MULTICAST_GRANULARITY_RECOMMENDED));
#endif

  return ncclSuccess;
}

ncclResult_t vmmInitPool(cocclVmmDeviceBufferPool* pool, int cudaDev, bool* available) {
  *available = false;

  if (pool->ready) {
    *available = true;
    return ncclSuccess;
  }

  // VMM is optional. If any CUDA-driver entry point is missing or NCCL disables
  // cuMem, return success with available=false so the caller can use legacy.
  if (ncclCudaLibraryInit() != ncclSuccess || !ncclCuMemEnable() ||
      CUPFN(cuDeviceGet) == nullptr || CUPFN(cuDeviceGetAttribute) == nullptr ||
      CUPFN(cuMemAddressReserve) == nullptr || CUPFN(cuMemAddressFree) == nullptr ||
      CUPFN(cuMemCreate) == nullptr || CUPFN(cuMemGetAllocationGranularity) == nullptr ||
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
  size_t multicastGranularity = 0;

  CUDACHECKGOTO(cudaSetDevice(cudaDev), ret, unavailable);
  CUCHECKGOTO(cuDeviceGet(&pool->cuDev, cudaDev), ret, unavailable);
  CUCHECKGOTO(cuDeviceGetAttribute(&vmmSupported, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
                                   pool->cuDev), ret, unavailable);
  if (!vmmSupported) goto unavailable;
  // COCCL buffers are registered for NCCL transport, so require the VMM path
  // that is compatible with GPUDirect RDMA.
  CUCHECKGOTO(cuDeviceGetAttribute(&rdmaVmmSupported, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED,
                                   pool->cuDev), ret, unavailable);
  if (!rdmaVmmSupported) goto unavailable;
  CUCHECKGOTO(cuDeviceGetAttribute(&rdmaSupported, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED,
                                   pool->cuDev), ret, unavailable);

  pool->cudaDev = cudaDev;
  pool->gpuDirectRdmaCapable = rdmaSupported != 0;
  NCCLCHECKGOTO(vmmBuildAllocationProp(pool, &prop), ret, unavailable);
  CUCHECKGOTO(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED),
              ret, unavailable);

  {
    size_t requestedChunkBytes = cocclGetConfig().buffer.physicalChunkBytes;
    size_t chunkSize = std::max(requestedChunkBytes, granularity);
    NCCLCHECKGOTO(vmmGetMulticastGranularity(pool, requestedChunkBytes, &multicastGranularity),
                  ret, unavailable);
    pool->granularity = granularity;
    // COCCL buffers are large communication workspaces, so use CUDA's
    // recommended allocation granularity to match ncclMemAlloc's communication
    // memory behavior. When multicast is supported, mirror ncclMemAlloc's
    // alignment-only compatibility behavior without creating multicast objects.
    chunkSize = alignUp(chunkSize, granularity);
    if (multicastGranularity != 0) chunkSize = alignUp(chunkSize, multicastGranularity);
    pool->chunkSize = chunkSize;
    pool->ready = true;
    *available = true;
  }
  return ncclSuccess;

unavailable:
  pool->ready = false;
  *available = false;
  return ncclSuccess;
}

static bool vmmSliceReusable(cocclVmmBufferSlice* slice) {
  if (slice->state == SliceState::Free) return true;
  if (slice->state != SliceState::Pending || slice->doneEvent == nullptr) return false;

  // A pending slice becomes reusable only when the stream that released it has
  // passed doneEvent. This prevents overlapping collectives from sharing bytes.
  cudaError_t status = cudaEventQuery(slice->doneEvent);
  if (status == cudaSuccess) {
    slice->state = SliceState::Free;
    slice->ownerComm = nullptr;
    return true;
  }
  if (status == cudaErrorNotReady) return false;
  WARN("COCCL VMM buffer event query failed: %s", cudaGetErrorString(status));
  return false;
}

static bool vmmBlockIsCompletelyFree(cocclVmmBackingBlock* block) {
  for (cocclVmmBufferSlice& slice : block->slices) {
    if (!vmmSliceReusable(&slice)) return false;
  }
  return true;
}

static void vmmMergeFreeSlices(cocclVmmBackingBlock* block) {
  for (auto it = block->slices.begin(); it != block->slices.end();) {
    auto next = std::next(it);
    if (next == block->slices.end()) break;
    if (vmmSliceReusable(&*it) && vmmSliceReusable(&*next) &&
        it->offset + it->bytes == next->offset) {
      it->bytes += next->bytes;
      // The merged slice keeps the first event object. The second one is no
      // longer reachable after erasing that slice.
      if (next->doneEvent != nullptr) CUDACHECKIGNORE(cudaEventDestroy(next->doneEvent));
      block->slices.erase(next);
    } else {
      ++it;
    }
  }
}

static ncclResult_t vmmDeregister(ncclComm_t comm, void* handle) {
  if (handle == nullptr) return ncclSuccess;
  return ncclCommDeregister(comm, handle);
}

static ncclResult_t vmmEnsureRegistrationForRange(cocclVmmBackingBlock* block, ncclComm_t ownerComm,
                                                  ncclComm_t registeredComm, void* ptr, size_t bytes) {
  uintptr_t rangePtr = (uintptr_t)ptr;
  for (const cocclVmmBufferRegistration& registration : block->registrations) {
    if (registration.registeredComm != registeredComm) continue;
    uintptr_t registeredPtr = (uintptr_t)registration.ptr;
    // Exact or larger existing registration can cover the requested range.
    if (rangeContains(registeredPtr, registration.bytes, rangePtr, bytes)) return ncclSuccess;
    // Avoid creating partially overlapping NCCL registrations for the same
    // comm. The caller will try a different free slice or allocate a new block.
    if (rangeOverlaps(registeredPtr, registration.bytes, rangePtr, bytes)) return ncclInProgress;
  }

  // Register the contiguous VA range handed to the primitive, not the entire
  // virtual reservation. This keeps registration cache pressure proportional
  // to active workspace sizes.
  void* handle = nullptr;
  NCCLCHECK(ncclCommRegister(registeredComm, ptr, bytes, &handle));
  block->registrations.push_back({registeredComm, ownerComm, ptr, bytes, handle});
  return ncclSuccess;
}

static ncclResult_t vmmAcquirePhysicalHandle(cocclVmmDeviceBufferPool* pool,
                                             CUmemGenericAllocationHandle* handleOut) {
  if (!pool->freeHandles.empty()) {
    // Reuse a detached physical chunk before asking the driver for a new one.
    *handleOut = pool->freeHandles.back();
    pool->freeHandles.pop_back();
    return ncclSuccess;
  }

  CUmemAllocationProp prop = {};
  NCCLCHECK(vmmBuildAllocationProp(pool, &prop));
  CUCHECK(cuMemCreate(handleOut, pool->chunkSize, &prop, 0));
  pool->totalPhysicalBytes += pool->chunkSize;
  return ncclSuccess;
}

static ncclResult_t vmmReleaseFreeHandlesToLimit(cocclVmmDeviceBufferPool* pool) {
  ncclResult_t ret = ncclSuccess;
  size_t limit = cocclGetConfig().buffer.poolLimitBytes;
  if (limit == 0) return ncclSuccess;

  // Only handles already detached from all VA ranges are eligible here; active
  // blocks are never unmapped merely to satisfy the limit.
  while (pool->totalPhysicalBytes > limit && !pool->freeHandles.empty()) {
    CUmemGenericAllocationHandle handle = pool->freeHandles.back();
    pool->freeHandles.pop_back();
    CUCHECKGOTO(cuMemRelease(handle), ret, fail);
    if (pool->totalPhysicalBytes >= pool->chunkSize) pool->totalPhysicalBytes -= pool->chunkSize;
  }

exit:
  return ret;
fail:
  goto exit;
}

static ncclResult_t vmmSetPeerAccessForBlock(cocclVmmDeviceBufferPool* pool,
                                             CUdeviceptr ptr, size_t bytes) {
  ncclResult_t ret = ncclSuccess;
  int deviceCount = 0;
  CUDACHECKGOTO(cudaGetDeviceCount(&deviceCount), ret, fail);

  // Match NCCL's cuMem allocation behavior: give this VA range read/write
  // access from the local CUDA device and any CUDA device that can P2P-access
  // the allocation device. This is separate from ncclCommRegister, which only
  // installs NCCL transport registration for a particular communicator.
  for (int dev = 0; dev < deviceCount; ++dev) {
    int p2p = 0;
    if (dev == pool->cudaDev ||
        ((cudaDeviceCanAccessPeer(&p2p, pool->cudaDev, dev) == cudaSuccess) && p2p)) {
      CUmemAccessDesc accessDesc = {};
      accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      accessDesc.location.id = dev;
      accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
      CUCHECKGOTO(cuMemSetAccess(ptr, bytes, &accessDesc, 1), ret, fail);
    }
  }

exit:
  return ret;
fail:
  goto exit;
}

static ncclResult_t vmmReleaseBlockMemory(cocclVmmDeviceBufferPool* pool, cocclVmmBackingBlock* block,
                                          bool releasePhysical) {
  ncclResult_t ret = ncclSuccess;
  CUDACHECKGOTO(cudaSetDevice(block->cudaDev), ret, fail);
  // NCCL registrations must be removed before unmapping the VA they refer to.
  for (cocclVmmBufferRegistration& registration : block->registrations) {
    NCCLCHECKGOTO(vmmDeregister(registration.registeredComm, registration.handle), ret, fail);
  }
  block->registrations.clear();

  for (cocclVmmBufferSlice& slice : block->slices) {
    if (slice.doneEvent != nullptr) {
      CUDACHECKGOTO(cudaEventDestroy(slice.doneEvent), ret, fail);
      slice.doneEvent = nullptr;
    }
  }

  if (block->ptr != nullptr && block->capacity != 0) {
    CUCHECKGOTO(cuMemUnmap((CUdeviceptr)block->ptr, block->capacity), ret, fail);
  }

  for (CUmemGenericAllocationHandle handle : block->handles) {
    if (releasePhysical) {
      // Full teardown path: return physical memory to CUDA.
      CUCHECKGOTO(cuMemRelease(handle), ret, fail);
      if (pool->totalPhysicalBytes >= pool->chunkSize) pool->totalPhysicalBytes -= pool->chunkSize;
    } else {
      // Trim/harvest path: keep the physical chunk cached so it can be remapped
      // into a future contiguous VA range without cuMemCreate.
      pool->freeHandles.push_back(handle);
    }
  }
  block->handles.clear();

  if (block->ptr != nullptr && block->capacity != 0) {
    CUCHECKGOTO(cuMemAddressFree((CUdeviceptr)block->ptr, block->capacity), ret, fail);
    block->ptr = nullptr;
    block->capacity = 0;
  }

exit:
  return ret;
fail:
  goto exit;
}

static ncclResult_t vmmCreateBackingBlock(cocclVmmDeviceBufferPool* pool, size_t bytes,
                                          cocclVmmBackingBlock** blockOut) {
  ncclResult_t ret = ncclSuccess;
  // Each block is one contiguous VA reservation sized to full physical chunks.
  size_t blockBytes = alignUp(bytes, pool->chunkSize);
  size_t chunkCount = blockBytes / pool->chunkSize;
  size_t mappedChunks = 0;
  CUdeviceptr reservation = 0;
  std::unique_ptr<cocclVmmBackingBlock> block(new cocclVmmBackingBlock());

  block->cudaDev = pool->cudaDev;
  block->cuDev = pool->cuDev;
  block->capacity = blockBytes;
  block->chunkSize = pool->chunkSize;
  block->handles.reserve(chunkCount);

  CUDACHECKGOTO(cudaSetDevice(pool->cudaDev), ret, fail);
  CUCHECKGOTO(cuMemAddressReserve(&reservation, blockBytes, pool->granularity, 0, 0), ret, fail);
  block->ptr = (void*)reservation;

  for (size_t i = 0; i < chunkCount; ++i) {
    // Physical chunks may come from different addresses internally; after
    // cuMemMap they form one contiguous virtual buffer for callers.
    CUmemGenericAllocationHandle handle;
    NCCLCHECKGOTO(vmmAcquirePhysicalHandle(pool, &handle), ret, fail);
    block->handles.push_back(handle);
    CUCHECKGOTO(cuMemMap(reservation + i * pool->chunkSize, pool->chunkSize, 0, handle, 0), ret, fail);
    mappedChunks++;
  }

  NCCLCHECKGOTO(vmmSetPeerAccessForBlock(pool, reservation, blockBytes), ret, fail);

  block->slices.push_back({0, blockBytes, SliceState::Free, nullptr, nullptr, block.get()});
  *blockOut = block.get();
  pool->blocks.push_back(std::move(block));

exit:
  return ret;
fail:
  if (block != nullptr) {
    // On partial construction failure, recycle handles already acquired in
    // this attempt and unmap only the chunks that were successfully mapped.
    for (CUmemGenericAllocationHandle handle : block->handles) {
      pool->freeHandles.push_back(handle);
    }
    block->handles.clear();
    if (block->ptr != nullptr && mappedChunks != 0) {
      CUCHECKIGNORE(cuMemUnmap((CUdeviceptr)block->ptr, mappedChunks * pool->chunkSize));
    }
    if (block->ptr != nullptr && block->capacity != 0) {
      CUCHECKIGNORE(cuMemAddressFree((CUdeviceptr)block->ptr, block->capacity));
    }
  } else if (reservation != 0) {
    CUCHECKIGNORE(cuMemAddressFree(reservation, blockBytes));
  }
  goto exit;
}

static ncclResult_t vmmTrimPoolLocked(cocclVmmDeviceBufferPool* pool) {
  ncclResult_t ret = ncclSuccess;
  size_t limit = cocclGetConfig().buffer.poolLimitBytes;
  if (limit == 0 || pool->totalPhysicalBytes <= limit) return ncclSuccess;

  for (auto it = pool->blocks.begin(); it != pool->blocks.end() && pool->totalPhysicalBytes > limit;) {
    cocclVmmBackingBlock* block = it->get();
    vmmMergeFreeSlices(block);
    if (block->registrations.empty() && vmmBlockIsCompletelyFree(block)) {
      // Registered blocks are kept because deregistering/remapping a hot range
      // can cost more than the memory saved.
      NCCLCHECKGOTO(vmmReleaseBlockMemory(pool, block, false), ret, fail);
      it = pool->blocks.erase(it);
    } else {
      ++it;
    }
  }
  NCCLCHECKGOTO(vmmReleaseFreeHandlesToLimit(pool), ret, fail);

exit:
  return ret;
fail:
  goto exit;
}

static ncclResult_t vmmHarvestFreeBlocksLocked(cocclVmmDeviceBufferPool* pool) {
  ncclResult_t ret = ncclSuccess;
  // Before growing the pool, detach physical chunks from old VA ranges that no
  // communicator has registered and no stream is still using. The handles stay
  // cached in freeHandles, so growth can remap them without a new cuMemCreate.
  for (auto it = pool->blocks.begin(); it != pool->blocks.end();) {
    cocclVmmBackingBlock* block = it->get();
    vmmMergeFreeSlices(block);
    if (block->registrations.empty() && vmmBlockIsCompletelyFree(block)) {
      NCCLCHECKGOTO(vmmReleaseBlockMemory(pool, block, false), ret, fail);
      it = pool->blocks.erase(it);
    } else {
      ++it;
    }
  }

exit:
  return ret;
fail:
  goto exit;
}

static ncclResult_t vmmAcquireFromBlock(cocclVmmBackingBlock* block, size_t bytes,
                                        ncclComm_t ownerComm, ncclComm_t registeredComm,
                                        cocclBufferHandle* buffer) {
  vmmMergeFreeSlices(block);

  for (auto it = block->slices.begin(); it != block->slices.end(); ++it) {
    if (!vmmSliceReusable(&*it) || it->bytes < bytes) continue;

    if (it->bytes > bytes) {
      // Split the free slice so the returned handle is exactly the requested
      // aligned byte count; the remainder stays available for later requests.
      size_t remainOffset = it->offset + bytes;
      size_t remainBytes = it->bytes - bytes;
      it->bytes = bytes;
      block->slices.insert(std::next(it), {remainOffset, remainBytes, SliceState::Free,
                                           nullptr, nullptr, block});
    }

    it->state = SliceState::InUse;
    it->ownerComm = ownerComm;
    it->block = block;

    buffer->ptr = static_cast<char*>(block->ptr) + it->offset;
    buffer->bytes = bytes;
    buffer->ownerComm = ownerComm;
    buffer->block = block;
    buffer->slice = &*it;

    ncclResult_t ret = registeredComm == nullptr
        ? ncclSuccess
        : vmmEnsureRegistrationForRange(
              block, ownerComm, registeredComm, buffer->ptr, bytes);
    if (ret == ncclSuccess) return ncclSuccess;

    // A partial-overlap registration conflict means this VA range is not usable
    // for registeredComm. Put the slice back and keep scanning other ranges.
    it->state = SliceState::Free;
    it->ownerComm = nullptr;
    *buffer = {};
    if (ret == ncclInProgress) continue;
    return ret;
  }

  return ncclInProgress;
}

ncclResult_t vmmReleaseCommRegistrationsLocked(cocclVmmDeviceBufferPool* pool, ncclComm_t comm) {
  ncclResult_t ret = ncclSuccess;
  CUDACHECKGOTO(cudaSetDevice(pool->cudaDev), ret, fail);
  for (auto& blockPtr : pool->blocks) {
    cocclVmmBackingBlock* block = blockPtr.get();
    for (cocclVmmBufferSlice& slice : block->slices) {
      if (slice.ownerComm != comm) continue;
      // Destroying a comm must not leave its owned slice in flight. Prefer the
      // recorded event when available; fall back to device sync for raw InUse.
      if (slice.state == SliceState::Pending && slice.doneEvent != nullptr) {
        CUDACHECKGOTO(cudaEventSynchronize(slice.doneEvent), ret, fail);
      } else if (slice.state == SliceState::InUse) {
        CUDACHECKGOTO(cudaDeviceSynchronize(), ret, fail);
      }
      slice.state = SliceState::Free;
      slice.ownerComm = nullptr;
    }

    for (auto registration = block->registrations.begin(); registration != block->registrations.end();) {
      if (registration->registeredComm == comm || registration->ownerComm == comm) {
        // Remove both direct registrations for comm and registrations that this
        // comm introduced for split/hierarchical transport comms.
        NCCLCHECKGOTO(vmmDeregister(registration->registeredComm, registration->handle), ret, fail);
        registration = block->registrations.erase(registration);
      } else {
        ++registration;
      }
    }
    vmmMergeFreeSlices(block);
  }
  NCCLCHECKGOTO(vmmTrimPoolLocked(pool), ret, fail);

exit:
  return ret;
fail:
  goto exit;
}

ncclResult_t vmmAcquireForComm(cocclVmmDeviceBufferPool* pool, ncclComm_t ownerComm,
                               ncclComm_t registeredComm, size_t bytes,
                               cocclBufferHandle* buffer) {
  ncclResult_t ret = ncclSuccess;
  for (auto& blockPtr : pool->blocks) {
    cocclVmmBackingBlock* block = blockPtr.get();
    if (block->capacity < bytes) continue;
    ret = vmmAcquireFromBlock(block, bytes, ownerComm, registeredComm, buffer);
    if (ret == ncclSuccess) return ncclSuccess;
    if (ret != ncclInProgress) return ret;
  }

  {
    cocclVmmBackingBlock* block = nullptr;
    NCCLCHECKGOTO(vmmHarvestFreeBlocksLocked(pool), ret, fail);
    NCCLCHECKGOTO(vmmCreateBackingBlock(pool, bytes, &block), ret, fail);
    NCCLCHECKGOTO(vmmAcquireFromBlock(block, bytes, ownerComm, registeredComm, buffer), ret, fail);
  }
  return ncclSuccess;

fail:
  return ret;
}

ncclResult_t vmmRegisterBufferForComm(cocclBufferHandle* buffer, ncclComm_t registeredComm) {
  cocclVmmBackingBlock* block = static_cast<cocclVmmBackingBlock*>(buffer->block);
  return vmmEnsureRegistrationForRange(block, buffer->ownerComm, registeredComm, buffer->ptr, buffer->bytes);
}

ncclResult_t vmmReleaseBuffer(cocclBufferHandle* buffer, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;
  cocclVmmBufferSlice* slice = static_cast<cocclVmmBufferSlice*>(buffer->slice);
  cocclVmmBackingBlock* block = static_cast<cocclVmmBackingBlock*>(buffer->block);

  CUDACHECKGOTO(cudaSetDevice(block->cudaDev), ret, fail);
  if (slice->doneEvent == nullptr) {
    CUDACHECKGOTO(cudaEventCreateWithFlags(&slice->doneEvent, cudaEventDisableTiming), ret, fail);
  }
  // Re-recording the same event is intentional: each release updates the point
  // in stream after which this slice is safe to reuse.
  CUDACHECKGOTO(cudaEventRecord(slice->doneEvent, stream), ret, fail);
  slice->state = SliceState::Pending;
  slice->ownerComm = buffer->ownerComm;
  *buffer = {};

exit:
  return ret;
fail:
  goto exit;
}

ncclResult_t vmmDestroyPoolLocked(cocclVmmDeviceBufferPool* pool) {
  ncclResult_t ret = ncclSuccess;
  if (!pool->ready) return ncclSuccess;

  CUDACHECKGOTO(cudaSetDevice(pool->cudaDev), ret, fail);
  for (auto& blockPtr : pool->blocks) {
    cocclVmmBackingBlock* block = blockPtr.get();
    for (cocclVmmBufferSlice& slice : block->slices) {
      // Full teardown drains all outstanding work before deregistering and
      // unmapping VA ranges.
      if (slice.state == SliceState::Pending && slice.doneEvent != nullptr) {
        CUDACHECKGOTO(cudaEventSynchronize(slice.doneEvent), ret, fail);
      } else if (slice.state == SliceState::InUse) {
        CUDACHECKGOTO(cudaDeviceSynchronize(), ret, fail);
      }
      slice.state = SliceState::Free;
      slice.ownerComm = nullptr;
    }
    NCCLCHECKGOTO(vmmReleaseBlockMemory(pool, block, true), ret, fail);
  }
  pool->blocks.clear();
  while (!pool->freeHandles.empty()) {
    CUmemGenericAllocationHandle handle = pool->freeHandles.back();
    pool->freeHandles.pop_back();
    CUCHECKGOTO(cuMemRelease(handle), ret, fail);
    if (pool->totalPhysicalBytes >= pool->chunkSize) {
      pool->totalPhysicalBytes -= pool->chunkSize;
    }
  }
  pool->ready = false;

exit:
  return ret;
fail:
  goto exit;
}

}  // namespace coccl_buffer

#endif
