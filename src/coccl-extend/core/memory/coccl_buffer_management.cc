#include "core/memory/coccl_buffer_management.h"

#include "checks.h"
#include "core/memory/coccl_buffer_internal.h"

#include <map>
#include <memory>
#include <pthread.h>

using namespace coccl_buffer;

namespace {

pthread_mutex_t bufferLock = PTHREAD_MUTEX_INITIALIZER;
std::map<ncclComm_t, std::unique_ptr<CommBufferPool>> commPools;

CommBufferPool* findPool(ncclComm_t comm) {
  auto found = commPools.find(comm);
  return found == commPools.end() ? nullptr : found->second.get();
}

ncclResult_t initPool(ncclComm_t comm, CommBufferPool* pool) {
  pool->ownerComm = comm;
  pool->cudaDev = comm->cudaDev;
#if CUDART_VERSION >= 11030
  bool available = false;
  NCCLCHECK(vmmInit(&pool->vmm, comm, &available));
  if (available) pool->backend = BufferBackend::Vmm;
#endif
  INFO(NCCL_INIT, "COCCL buffer comm %p backend %s", comm,
       pool->backend == BufferBackend::Vmm ? "VMM" : "legacy");
  return ncclSuccess;
}

ncclResult_t deregisterComm(CommBufferPool* pool, ncclComm_t comm) {
#if CUDART_VERSION >= 11030
  if (pool->backend == BufferBackend::Vmm) {
    return vmmDeregisterComm(&pool->vmm, comm);
  }
#endif
  return legacyDeregisterComm(pool, comm);
}

ncclResult_t destroyPool(CommBufferPool* pool) {
#if CUDART_VERSION >= 11030
  if (pool->backend == BufferBackend::Vmm) {
    return vmmDestroy(&pool->vmm);
  }
#endif
  const size_t releasedBytes = pool->totalBytes;
  NCCLCHECK(legacyDestroy(pool));
  INFO(NCCL_INIT,
       "COCCL legacy buffer comm %p released %zu bytes", pool->ownerComm,
       releasedBytes);
  return ncclSuccess;
}

ncclResult_t acquire(ncclComm_t ownerComm, ncclComm_t registeredComm,
                     size_t bytes, cudaStream_t stream,
                     cocclBufferHandle* buffer) {
  if (ownerComm == nullptr || buffer == nullptr) return ncclInvalidArgument;
  *buffer = {};

  ncclResult_t ret = ncclSuccess;
  pthread_mutex_lock(&bufferLock);
  CommBufferPool* pool = findPool(ownerComm);
  if (pool == nullptr) {
    ret = ncclInvalidUsage;
    goto exit;
  }
  CUDACHECKGOTO(cudaSetDevice(pool->cudaDev), ret, exit);
  bytes = alignUp(bytes == 0 ? 1 : bytes);
#if CUDART_VERSION >= 11030
  if (pool->backend == BufferBackend::Vmm) {
    NCCLCHECKGOTO(vmmAcquire(&pool->vmm, registeredComm, bytes, stream,
                             buffer), ret, exit);
  } else
#endif
  {
    NCCLCHECKGOTO(legacyAcquire(pool, registeredComm, bytes, stream, buffer),
                  ret, exit);
  }

exit:
  pthread_mutex_unlock(&bufferLock);
  return ret;
}

}  // namespace

ncclResult_t cocclBufferCommInit(ncclComm_t comm) {
  if (comm == nullptr) return ncclInvalidArgument;

  pthread_mutex_lock(&bufferLock);
  if (findPool(comm) == nullptr) {
    std::unique_ptr<CommBufferPool> pool(new CommBufferPool());
    ncclResult_t ret = initPool(comm, pool.get());
    if (ret == ncclSuccess) commPools.emplace(comm, std::move(pool));
    pthread_mutex_unlock(&bufferLock);
    return ret;
  }
  pthread_mutex_unlock(&bufferLock);
  return ncclSuccess;
}

ncclResult_t cocclBufferCommDestroy(ncclComm_t comm) {
  if (comm == nullptr) return ncclSuccess;

  ncclResult_t ret = ncclSuccess;
  pthread_mutex_lock(&bufferLock);

  // A parent-owned block may also be registered with a hierarchical child.
  for (auto& entry : commPools) {
    if (entry.first == comm) continue;
    NCCLCHECKGOTO(deregisterComm(entry.second.get(), comm), ret, exit);
  }

  {
    auto found = commPools.find(comm);
    if (found != commPools.end()) {
      NCCLCHECKGOTO(destroyPool(found->second.get()), ret, exit);
      commPools.erase(found);
    }
  }

exit:
  pthread_mutex_unlock(&bufferLock);
  return ret;
}

ncclResult_t cocclGetBufferForComm(ncclComm_t ownerComm,
                                   ncclComm_t registeredComm, size_t bytes,
                                   cudaStream_t stream,
                                   cocclBufferHandle* buffer) {
  if (registeredComm == nullptr) return ncclInvalidArgument;
  return acquire(ownerComm, registeredComm, bytes, stream, buffer);
}

ncclResult_t cocclGetBuffer(ncclComm_t comm, size_t bytes,
                            cudaStream_t stream,
                            cocclBufferHandle* buffer) {
  return cocclGetBufferForComm(comm, comm, bytes, stream, buffer);
}

ncclResult_t cocclGetUnregisteredBuffer(ncclComm_t ownerComm, size_t bytes,
                                        cudaStream_t stream,
                                        cocclBufferHandle* buffer) {
  return acquire(ownerComm, nullptr, bytes, stream, buffer);
}

ncclResult_t cocclRegisterBufferForComm(cocclBufferHandle* buffer,
                                        ncclComm_t registeredComm) {
  if (buffer == nullptr || buffer->slice == nullptr ||
      registeredComm == nullptr) {
    return ncclInvalidArgument;
  }

  ncclResult_t ret = ncclSuccess;
  pthread_mutex_lock(&bufferLock);
  BufferBlock* block = static_cast<BufferBlock*>(buffer->block);
#if CUDART_VERSION >= 11030
  if (block->backend == BufferBackend::Vmm) {
    NCCLCHECKGOTO(vmmRegister(buffer, registeredComm), ret, exit);
  } else
#endif
  {
    NCCLCHECKGOTO(legacyRegister(buffer, registeredComm), ret, exit);
  }

exit:
  pthread_mutex_unlock(&bufferLock);
  return ret;
}

ncclResult_t cocclReleaseBuffer(cocclBufferHandle* buffer,
                                cudaStream_t stream) {
  if (buffer == nullptr || buffer->slice == nullptr) return ncclSuccess;

  ncclResult_t ret = ncclSuccess;
  pthread_mutex_lock(&bufferLock);
  BufferBlock* block = static_cast<BufferBlock*>(buffer->block);
#if CUDART_VERSION >= 11030
  if (block->backend == BufferBackend::Vmm) {
    NCCLCHECKGOTO(vmmRelease(buffer, stream), ret, exit);
  } else
#endif
  {
    NCCLCHECKGOTO(legacyRelease(buffer, stream), ret, exit);
  }

exit:
  if (ret == ncclSuccess) *buffer = {};
  pthread_mutex_unlock(&bufferLock);
  return ret;
}
