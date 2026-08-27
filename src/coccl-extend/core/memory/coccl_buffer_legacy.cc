#include "core/memory/coccl_buffer_internal.h"

#include "checks.h"
#include "core/config/coccl_config.h"

#include <iterator>

namespace coccl_buffer {
namespace {

bool reusable(LegacySlice* slice) {
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

bool reusableOnStream(LegacySlice* slice, cudaStream_t stream,
                      bool allowPendingReuse) {
  if (allowPendingReuse && slice->state == SliceState::Pending &&
      slice->pendingStream == stream) {
    return true;
  }
  return reusable(slice);
}

void mergeFreeSlices(LegacyBlock* block) {
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

ncclResult_t ensureRegistration(LegacyBlock* block,
                                ncclComm_t registeredComm,
                                cocclBufferRegistrationKind requested) {
  if (registeredComm == nullptr) return ncclSuccess;
  auto existing = block->registrations.find(registeredComm);
  if (existing != block->registrations.end()) {
    return upgradeRegistration(&existing->second, requested);
  }

  BufferRegistration registration;
  NCCLCHECK(registerBuffer(registeredComm, block->ptr, block->capacity,
                           requested, &registration));
  block->registrations.emplace(registeredComm, registration);
  return ncclSuccess;
}

ncclResult_t createBlock(CommBufferPool* pool, size_t bytes,
                         LegacyBlock** result) {
  size_t blockBytes = bytes;
  if (cocclGetConfig().buffer.legacyBlockBytes > blockBytes) {
    blockBytes = alignUp(cocclGetConfig().buffer.legacyBlockBytes);
  }

  std::unique_ptr<LegacyBlock> block(new LegacyBlock());
  block->cudaDev = pool->cudaDev;
  block->capacity = blockBytes;
  NCCLCHECK(ncclMemAlloc(&block->ptr, blockBytes));

  LegacySlice initial;
  initial.bytes = blockBytes;
  initial.block = block.get();
  block->slices.push_back(initial);

  *result = block.get();
  pool->totalBytes += blockBytes;
  pool->blocks.push_back(std::move(block));
  INFO(COCCL_MEMORY,
       "COCCL legacy buffer comm %p allocated %zu bytes, total %zu",
       pool->ownerComm, blockBytes, pool->totalBytes);
  return ncclSuccess;
}

ncclResult_t acquireFromBlock(LegacyBlock* block, size_t bytes,
                              ncclComm_t ownerComm, cudaStream_t stream,
                              bool allowPendingReuse,
                              cocclBufferHandle* buffer) {
  mergeFreeSlices(block);
  for (auto slice = block->slices.begin(); slice != block->slices.end();
       ++slice) {
    if (!reusableOnStream(&*slice, stream, allowPendingReuse) ||
        slice->bytes < bytes) {
      continue;
    }

    const bool pendingReuse = slice->state == SliceState::Pending;
    if (slice->bytes > bytes && !pendingReuse) {
      LegacySlice remainder;
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
    buffer->ownerComm = ownerComm;
    buffer->block = block;
    buffer->slice = &*slice;
    return ncclSuccess;
  }
  return ncclInProgress;
}

void rollback(cocclBufferHandle* buffer) {
  LegacySlice* slice = static_cast<LegacySlice*>(buffer->slice);
  slice->state = SliceState::Free;
  *buffer = {};
}

ncclResult_t releaseBlock(LegacyBlock* block) {
  ncclResult_t ret = ncclSuccess;
  for (auto& registration : block->registrations) {
    NCCLCHECKGOTO(deregisterBuffer(&registration.second), ret, exit);
  }
  block->registrations.clear();

  for (LegacySlice& slice : block->slices) {
    if (slice.doneEvent != nullptr) {
      CUDACHECKGOTO(cudaEventDestroy(slice.doneEvent), ret, exit);
      slice.doneEvent = nullptr;
    }
  }
  if (block->ptr != nullptr) {
    NCCLCHECKGOTO(ncclMemFree(block->ptr), ret, exit);
    block->ptr = nullptr;
  }

exit:
  return ret;
}

}  // namespace

ncclResult_t legacyAcquire(CommBufferPool* pool, ncclComm_t registeredComm,
                           cocclBufferRegistrationKind registration,
                           size_t bytes, cudaStream_t stream,
                           cocclBufferHandle* buffer) {
  for (auto& block : pool->blocks) {
    const auto existing = block->registrations.find(registeredComm);
    const bool registered = registeredComm == nullptr ||
        existing != block->registrations.end();
    ncclResult_t ret = acquireFromBlock(block.get(), bytes, pool->ownerComm,
                                        stream, registered, buffer);
    if (ret == ncclSuccess) {
      ret = ensureRegistration(block.get(), registeredComm, registration);
      if (ret != ncclSuccess) rollback(buffer);
      return ret;
    }
  }

  LegacyBlock* block = nullptr;
  NCCLCHECK(createBlock(pool, bytes, &block));
  NCCLCHECK(acquireFromBlock(block, bytes, pool->ownerComm, stream, false,
                             buffer));
  ncclResult_t ret = ensureRegistration(block, registeredComm, registration);
  if (ret != ncclSuccess) rollback(buffer);
  return ret;
}

ncclResult_t legacyRegister(cocclBufferHandle* buffer,
                            ncclComm_t registeredComm,
                            cocclBufferRegistrationKind registration) {
  return ensureRegistration(static_cast<LegacyBlock*>(buffer->block),
                            registeredComm, registration);
}

ncclResult_t legacyRelease(cocclBufferHandle* buffer, cudaStream_t stream) {
  LegacySlice* slice = static_cast<LegacySlice*>(buffer->slice);
  LegacyBlock* block = static_cast<LegacyBlock*>(buffer->block);
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

ncclResult_t legacyDeregisterComm(CommBufferPool* pool, ncclComm_t comm) {
  for (auto& block : pool->blocks) {
    auto registration = block->registrations.find(comm);
    if (registration == block->registrations.end()) continue;
    NCCLCHECK(deregisterBuffer(&registration->second));
    block->registrations.erase(registration);
  }
  return ncclSuccess;
}

ncclResult_t legacyDestroy(CommBufferPool* pool) {
  ncclResult_t ret = ncclSuccess;
  CUDACHECKGOTO(cudaSetDevice(pool->cudaDev), ret, exit);
  CUDACHECKGOTO(cudaDeviceSynchronize(), ret, exit);
  for (auto& block : pool->blocks) {
    NCCLCHECKGOTO(releaseBlock(block.get()), ret, exit);
  }
  pool->blocks.clear();
  pool->totalBytes = 0;

exit:
  return ret;
}

}  // namespace coccl_buffer
