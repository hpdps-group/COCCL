#include "coccl_buffer_internal.h"

namespace coccl_buffer {

static bool legacySliceReusable(cocclLegacyBufferSlice* slice) {
  if (slice->state == SliceState::Free) return true;
  if (slice->state != SliceState::Pending || slice->doneEvent == nullptr) return false;

  // Pending slices are released asynchronously. Query the event rather than
  // synchronizing so acquire never serializes independent COCCL operations.
  cudaError_t status = cudaEventQuery(slice->doneEvent);
  if (status == cudaSuccess) {
    slice->state = SliceState::Free;
    slice->ownerComm = nullptr;
    return true;
  }
  if (status == cudaErrorNotReady) return false;
  WARN("COCCL legacy buffer event query failed: %s", cudaGetErrorString(status));
  return false;
}

static bool legacyBlockIsCompletelyFree(cocclLegacyBackingBlock* block) {
  for (cocclLegacyBufferSlice& slice : block->slices) {
    if (!legacySliceReusable(&slice)) return false;
  }
  return true;
}

static void legacyMergeFreeSlices(cocclLegacyBackingBlock* block) {
  for (auto it = block->slices.begin(); it != block->slices.end();) {
    auto next = std::next(it);
    if (next == block->slices.end()) break;
    if (legacySliceReusable(&*it) && legacySliceReusable(&*next) &&
        it->offset + it->bytes == next->offset) {
      it->bytes += next->bytes;
      // After merging, the second slice metadata is destroyed, so its event
      // object must be released as well.
      if (next->doneEvent != nullptr) CUDACHECKIGNORE(cudaEventDestroy(next->doneEvent));
      block->slices.erase(next);
    } else {
      ++it;
    }
  }
}

static ncclResult_t legacyEnsureBlockRegistration(cocclLegacyBackingBlock* block, ncclComm_t ownerComm,
                                                  ncclComm_t registeredComm) {
  if (ownerComm == nullptr || registeredComm == nullptr) return ncclInvalidArgument;
  if (block->registrations.find(registeredComm) != block->registrations.end()) return ncclSuccess;

  // Legacy allocates one physical backing block with ncclMemAlloc, so register
  // the whole block once per transport comm instead of every small slice.
  void* handle = nullptr;
  NCCLCHECK(ncclCommRegister(registeredComm, block->ptr, block->capacity, &handle));
  block->registrations.emplace(registeredComm, cocclLegacyBufferRegistration{handle, ownerComm});
  return ncclSuccess;
}

static ncclResult_t legacyDeregister(ncclComm_t comm, void* handle) {
  if (handle == nullptr) return ncclSuccess;
  return ncclCommDeregister(comm, handle);
}

static ncclResult_t legacyReleaseBlockMemory(cocclLegacyBackingBlock* block) {
  ncclResult_t ret = ncclSuccess;
  if (block == nullptr) return ncclSuccess;

  CUDACHECKGOTO(cudaSetDevice(block->cudaDev), ret, fail);
  // Registration handles reference block->ptr, so they must be removed before
  // the underlying ncclMemAlloc allocation is freed.
  for (auto& registration : block->registrations) {
    NCCLCHECKGOTO(legacyDeregister(registration.first, registration.second.handle), ret, fail);
  }
  block->registrations.clear();

  for (cocclLegacyBufferSlice& slice : block->slices) {
    if (slice.doneEvent != nullptr) {
      CUDACHECKGOTO(cudaEventDestroy(slice.doneEvent), ret, fail);
      slice.doneEvent = nullptr;
    }
  }

  if (block->ptr != nullptr) {
    NCCLCHECKGOTO(ncclMemFree(block->ptr), ret, fail);
    block->ptr = nullptr;
  }

exit:
  return ret;
fail:
  goto exit;
}

static ncclResult_t legacyCreateBackingBlock(cocclLegacyDeviceBufferPool* pool, size_t bytes,
                                             cocclLegacyBackingBlock** blockOut) {
  ncclResult_t ret = ncclSuccess;
  if (pool == nullptr || blockOut == nullptr) return ncclInvalidArgument;

  size_t configuredBlockBytes = (size_t)ncclParamCocclDeviceBufferBlockBytes();
  size_t blockBytes = alignUp(bytes, kCocclBufferAlignment);
  // The optional block size trades memory footprint for fewer future
  // allocations when message sizes fluctuate around a common large size.
  if (configuredBlockBytes > blockBytes) {
    blockBytes = alignUp(configuredBlockBytes, kCocclBufferAlignment);
  }

  std::unique_ptr<cocclLegacyBackingBlock> block(new cocclLegacyBackingBlock());
  block->cudaDev = pool->cudaDev;
  block->capacity = blockBytes;

  CUDACHECKGOTO(cudaSetDevice(pool->cudaDev), ret, fail);
  NCCLCHECKGOTO(ncclMemAlloc(&block->ptr, blockBytes), ret, fail);
  // A fresh block starts as a single free slice. Later acquires split it and
  // releases/queries merge it back.
  block->slices.push_back({0, blockBytes, SliceState::Free, nullptr, nullptr, block.get()});

  *blockOut = block.get();
  pool->totalBytes += blockBytes;
  pool->blocks.push_back(std::move(block));

exit:
  return ret;
fail:
  if (block != nullptr && block->ptr != nullptr) (void)ncclMemFree(block->ptr);
  goto exit;
}

static ncclResult_t legacyTrimPoolLocked(cocclLegacyDeviceBufferPool* pool) {
  ncclResult_t ret = ncclSuccess;
  if (pool == nullptr) return ncclSuccess;

  size_t limit = (size_t)ncclParamCocclDeviceBufferPoolLimit();
  if (limit == 0 || pool->totalBytes <= limit) return ncclSuccess;

  // Only completely free and unregistered blocks are released. This avoids
  // blocking live communication or churning NCCL registration state.
  for (auto it = pool->blocks.begin(); it != pool->blocks.end() && pool->totalBytes > limit;) {
    cocclLegacyBackingBlock* block = it->get();
    legacyMergeFreeSlices(block);
    if (block->registrations.empty() && legacyBlockIsCompletelyFree(block)) {
      size_t capacity = block->capacity;
      NCCLCHECKGOTO(legacyReleaseBlockMemory(block), ret, fail);
      it = pool->blocks.erase(it);
      pool->totalBytes -= capacity;
    } else {
      ++it;
    }
  }

exit:
  return ret;
fail:
  goto exit;
}

static ncclResult_t legacyAcquireFromBlock(cocclLegacyBackingBlock* block, size_t bytes,
                                           ncclComm_t comm, cocclBufferHandle* buffer) {
  if (block == nullptr || buffer == nullptr) return ncclInvalidArgument;
  legacyMergeFreeSlices(block);

  for (auto it = block->slices.begin(); it != block->slices.end(); ++it) {
    if (!legacySliceReusable(&*it) || it->bytes < bytes) continue;

    if (it->bytes > bytes) {
      // Return an exact-size slice and leave the remainder as a neighboring
      // free slice. list iterators remain stable for the acquired slice.
      size_t remainOffset = it->offset + bytes;
      size_t remainBytes = it->bytes - bytes;
      it->bytes = bytes;
      block->slices.insert(std::next(it), {remainOffset, remainBytes, SliceState::Free,
                                           nullptr, nullptr, block});
    }

    it->state = SliceState::InUse;
    it->ownerComm = comm;
    it->block = block;

    buffer->ptr = static_cast<char*>(block->ptr) + it->offset;
    buffer->bytes = bytes;
    buffer->ownerComm = comm;
    buffer->block = block;
    buffer->slice = &*it;
    return ncclSuccess;
  }

  return ncclInProgress;
}

static void legacyRollbackBufferLocked(cocclBufferHandle* buffer) {
  if (buffer == nullptr || buffer->slice == nullptr) return;
  cocclLegacyBufferSlice* slice = static_cast<cocclLegacyBufferSlice*>(buffer->slice);
  // Registration failure happens after a slice has been marked InUse. Roll it
  // back so the failed acquire does not leak capacity.
  slice->state = SliceState::Free;
  slice->ownerComm = nullptr;
  *buffer = {};
}

ncclResult_t legacyReleaseCommRegistrationsLocked(cocclLegacyDeviceBufferPool* pool, ncclComm_t comm) {
  ncclResult_t ret = ncclSuccess;
  if (pool == nullptr || comm == nullptr) return ncclSuccess;

  CUDACHECKGOTO(cudaSetDevice(pool->cudaDev), ret, fail);
  for (auto& blockPtr : pool->blocks) {
    cocclLegacyBackingBlock* block = blockPtr.get();
    for (cocclLegacyBufferSlice& slice : block->slices) {
      if (slice.ownerComm != comm) continue;
      // Comm teardown must wait for any workspace it owns. Pending slices can
      // wait on their event; raw InUse slices require a conservative sync.
      if (slice.state == SliceState::Pending && slice.doneEvent != nullptr) {
        CUDACHECKGOTO(cudaEventSynchronize(slice.doneEvent), ret, fail);
      } else if (slice.state == SliceState::InUse) {
        CUDACHECKGOTO(cudaDeviceSynchronize(), ret, fail);
      }
      slice.state = SliceState::Free;
      slice.ownerComm = nullptr;
    }

    for (auto registration = block->registrations.begin(); registration != block->registrations.end();) {
      if (registration->first == comm || registration->second.ownerComm == comm) {
        // Remove both direct registrations and registrations this comm created
        // for a split/subcommunicator.
        NCCLCHECKGOTO(legacyDeregister(registration->first, registration->second.handle), ret, fail);
        registration = block->registrations.erase(registration);
      } else {
        ++registration;
      }
    }
    legacyMergeFreeSlices(block);
  }
  NCCLCHECKGOTO(legacyTrimPoolLocked(pool), ret, fail);

exit:
  return ret;
fail:
  goto exit;
}

ncclResult_t legacyAcquireForComm(cocclLegacyDeviceBufferPool* pool, ncclComm_t ownerComm,
                                  ncclComm_t registeredComm, size_t bytes,
                                  cocclBufferHandle* buffer) {
  ncclResult_t ret = ncclSuccess;
  if (pool == nullptr || ownerComm == nullptr || registeredComm == nullptr || buffer == nullptr) {
    return ncclInvalidArgument;
  }

  for (auto& blockPtr : pool->blocks) {
    cocclLegacyBackingBlock* block = blockPtr.get();
    if (block->capacity < bytes) continue;
    // First fit keeps acquire cheap. Fragmentation is controlled by merging
    // adjacent free slices whenever blocks are scanned.
    ret = legacyAcquireFromBlock(block, bytes, ownerComm, buffer);
    if (ret == ncclSuccess) {
      NCCLCHECKGOTO(legacyEnsureBlockRegistration(block, ownerComm, registeredComm), ret, rollback);
      return ncclSuccess;
    }
    if (ret != ncclInProgress) return ret;
  }

  {
    cocclLegacyBackingBlock* block = nullptr;
    NCCLCHECKGOTO(legacyCreateBackingBlock(pool, bytes, &block), ret, fail);
    NCCLCHECKGOTO(legacyAcquireFromBlock(block, bytes, ownerComm, buffer), ret, fail);
    NCCLCHECKGOTO(legacyEnsureBlockRegistration(block, ownerComm, registeredComm), ret, rollback);
  }
  return ncclSuccess;

rollback:
  legacyRollbackBufferLocked(buffer);
  return ret;
fail:
  if (buffer != nullptr) *buffer = {};
  return ret;
}

ncclResult_t legacyRegisterBufferForComm(cocclBufferHandle* buffer, ncclComm_t registeredComm) {
  if (buffer == nullptr || buffer->block == nullptr || buffer->ownerComm == nullptr || registeredComm == nullptr) {
    return ncclInvalidArgument;
  }
  cocclLegacyBackingBlock* block = static_cast<cocclLegacyBackingBlock*>(buffer->block);
  // Additional registrations still cover the entire block in the legacy path.
  return legacyEnsureBlockRegistration(block, buffer->ownerComm, registeredComm);
}

ncclResult_t legacyReleaseBuffer(cocclBufferHandle* buffer, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;
  if (buffer == nullptr || buffer->slice == nullptr || buffer->block == nullptr) return ncclSuccess;

  cocclLegacyBufferSlice* slice = static_cast<cocclLegacyBufferSlice*>(buffer->slice);
  cocclLegacyBackingBlock* block = static_cast<cocclLegacyBackingBlock*>(buffer->block);
  if (slice == nullptr || block == nullptr || slice->state != SliceState::InUse) return ncclSuccess;

  CUDACHECKGOTO(cudaSetDevice(block->cudaDev), ret, fail);
  if (slice->doneEvent == nullptr) {
    CUDACHECKGOTO(cudaEventCreateWithFlags(&slice->doneEvent, cudaEventDisableTiming), ret, fail);
  }
  // Reuse one event per slice. Recording it here establishes the point after
  // which this slice can safely be returned by a future acquire.
  CUDACHECKGOTO(cudaEventRecord(slice->doneEvent, stream), ret, fail);
  slice->state = SliceState::Pending;
  slice->ownerComm = buffer->ownerComm;
  *buffer = {};

exit:
  return ret;
fail:
  goto exit;
}

ncclResult_t legacyDestroyPoolLocked(cocclLegacyDeviceBufferPool* pool) {
  ncclResult_t ret = ncclSuccess;
  if (pool == nullptr) return ncclSuccess;
  if (pool->cudaDev >= 0) CUDACHECKGOTO(cudaSetDevice(pool->cudaDev), ret, fail);

  for (auto& blockPtr : pool->blocks) {
    cocclLegacyBackingBlock* block = blockPtr.get();
    for (cocclLegacyBufferSlice& slice : block->slices) {
      // Full pool destruction is allowed to drain all work because it happens
      // only when COCCL runtime tears down its last communicator.
      if (slice.state == SliceState::Pending && slice.doneEvent != nullptr) {
        CUDACHECKGOTO(cudaEventSynchronize(slice.doneEvent), ret, fail);
      } else if (slice.state == SliceState::InUse) {
        CUDACHECKGOTO(cudaDeviceSynchronize(), ret, fail);
      }
      slice.state = SliceState::Free;
      slice.ownerComm = nullptr;
    }
    NCCLCHECKGOTO(legacyReleaseBlockMemory(block), ret, fail);
  }
  pool->blocks.clear();
  pool->totalBytes = 0;

exit:
  return ret;
fail:
  goto exit;
}

}  // namespace coccl_buffer
