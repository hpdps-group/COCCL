#include "buffer/coccl_buffer_management.h"

#include "buffer/coccl_buffer_internal.h"
#include "config/coccl_config.h"

#include <map>
#include <pthread.h>

using namespace coccl_buffer;

namespace {

// One global lock protects devicePools and all backend metadata. Actual CUDA
// work is limited to allocation/registration/event management; primitive CUDA
// kernels and NCCL operations run after the lock is released.
pthread_mutex_t cocclBufferLock = PTHREAD_MUTEX_INITIALIZER;
// Device pools are keyed by ncclComm::cudaDev so multiple cocclComm sidecars on
// the same GPU share cached scratch memory.
std::map<int, cocclDeviceBufferPool> devicePools;

static ncclResult_t initDevicePoolLocked(cocclDeviceBufferPool* pool, int cudaDev) {
  if (pool == nullptr) return ncclInvalidArgument;
  if (pool->initialized) return ncclSuccess;

  pool->cudaDev = cudaDev;
  pool->legacy.cudaDev = cudaDev;
#if CUDART_VERSION >= 11030
  // Prefer VMM when NCCL's CUDA-driver wrapper and platform capabilities allow
  // it. vmmInitPool reports "not available" as success so COCCL can fall back.
  bool vmmAvailable = false;
  NCCLCHECK(vmmInitPool(&pool->vmm, cudaDev, &vmmAvailable));
  pool->useVmm = vmmAvailable;
#else
  pool->useVmm = false;
#endif
  pool->initialized = true;
  INFO(NCCL_INIT, "COCCL buffer manager cudaDev %d backend %s", cudaDev,
       pool->useVmm ? "VMM" : "legacy");
  return ncclSuccess;
}

static ncclResult_t releaseDevicePoolCommLocked(cocclDeviceBufferPool* pool, ncclComm_t comm) {
  ncclResult_t ret = ncclSuccess;
  if (pool == nullptr) return ncclSuccess;
  // A comm may have registered memory in either backend if the process created
  // pools before/after VMM availability changed or while cleaning up failures.
#if CUDART_VERSION >= 11030
  if (pool->useVmm) {
    NCCLCHECKGOTO(vmmReleaseCommRegistrationsLocked(&pool->vmm, comm), ret, fail);
  }
#endif
  NCCLCHECKGOTO(legacyReleaseCommRegistrationsLocked(&pool->legacy, comm), ret, fail);

exit:
  return ret;
fail:
  goto exit;
}

static ncclResult_t destroyDevicePoolLocked(cocclDeviceBufferPool* pool) {
  ncclResult_t ret = ncclSuccess;
  if (pool == nullptr) return ncclSuccess;

#if CUDART_VERSION >= 11030
  NCCLCHECKGOTO(vmmDestroyPoolLocked(&pool->vmm), ret, fail);
#endif
  NCCLCHECKGOTO(legacyDestroyPoolLocked(&pool->legacy), ret, fail);
  pool->initialized = false;
  pool->useVmm = false;

exit:
  return ret;
fail:
  goto exit;
}

}  // namespace

ncclResult_t cocclBufferCommInit(ncclComm_t comm) {
  if (comm == nullptr) return ncclInvalidArgument;

  ncclResult_t ret = ncclSuccess;
  pthread_mutex_lock(&cocclBufferLock);
  {
    // Init only ensures the owning device pool exists. No backing memory is
    // allocated until the first primitive actually requests a buffer.
    cocclDeviceBufferPool& pool = devicePools[comm->cudaDev];
    NCCLCHECKGOTO(initDevicePoolLocked(&pool, comm->cudaDev), ret, exit);
  }

exit:
  pthread_mutex_unlock(&cocclBufferLock);
  return ret;
}

ncclResult_t cocclBufferCommDestroy(ncclComm_t comm) {
  ncclResult_t ret = ncclSuccess;
  pthread_mutex_lock(&cocclBufferLock);
  // Split/hierarchical communicators can register memory owned by the original
  // comm's device pool. Scan every device pool so both owner and registered
  // comm handles are detached before NCCL destroys the comm.
  for (auto& poolEntry : devicePools) {
    NCCLCHECKGOTO(releaseDevicePoolCommLocked(&poolEntry.second, comm), ret, exit);
  }

exit:
  pthread_mutex_unlock(&cocclBufferLock);
  return ret;
}

ncclResult_t cocclBufferDestroyAll() {
  ncclResult_t ret = ncclSuccess;

  pthread_mutex_lock(&cocclBufferLock);
  // Runtime calls this only when the last COCCL comm is gone. Unlike normal
  // release, this is the path that really frees cached backing memory.
  for (auto& poolEntry : devicePools) {
    NCCLCHECKGOTO(destroyDevicePoolLocked(&poolEntry.second), ret, fail);
  }
  devicePools.clear();

exit:
  pthread_mutex_unlock(&cocclBufferLock);
  return ret;
fail:
  goto exit;
}

static ncclResult_t cocclAcquireBuffer(
    ncclComm_t ownerComm, ncclComm_t registeredComm, size_t bytes,
    cocclBufferHandle* buffer) {
  ncclResult_t ret = ncclSuccess;
  if (ownerComm == nullptr || buffer == nullptr) return ncclInvalidArgument;

  *buffer = {};
  bytes = alignUp(bytes == 0 ? 1 : bytes, kCocclBufferAlignment);

  pthread_mutex_lock(&cocclBufferLock);
  {
    // ownerComm chooses the device pool and owns lifetime accounting.
    // registeredComm is the NCCL transport comm that must see a registered
    // pointer before the workspace is used in send/recv/collective calls.
    cocclDeviceBufferPool& pool = devicePools[ownerComm->cudaDev];
    NCCLCHECKGOTO(initDevicePoolLocked(&pool, ownerComm->cudaDev), ret, fail);
    CUDACHECKGOTO(cudaSetDevice(ownerComm->cudaDev), ret, fail);

#if CUDART_VERSION >= 11030
    if (pool.useVmm) {
      NCCLCHECKGOTO(vmmAcquireForComm(&pool.vmm, ownerComm, registeredComm, bytes, buffer), ret, fail);
      goto exit;
    }
#endif
    NCCLCHECKGOTO(legacyAcquireForComm(&pool.legacy, ownerComm, registeredComm, bytes, buffer), ret, fail);
  }

exit:
  pthread_mutex_unlock(&cocclBufferLock);
  return ret;
fail:
  if (buffer != nullptr) *buffer = {};
  goto exit;
}

ncclResult_t cocclGetBufferForComm(ncclComm_t ownerComm,
                                   ncclComm_t registeredComm, size_t bytes,
                                   cocclBufferHandle* buffer) {
  if (registeredComm == nullptr) return ncclInvalidArgument;
  return cocclAcquireBuffer(ownerComm, registeredComm, bytes, buffer);
}

ncclResult_t cocclGetBuffer(ncclComm_t comm, size_t bytes, cocclBufferHandle* buffer) {
  return cocclGetBufferForComm(comm, comm, bytes, buffer);
}

ncclResult_t cocclGetUnregisteredBuffer(
    ncclComm_t ownerComm, size_t bytes, cocclBufferHandle* buffer) {
  return cocclAcquireBuffer(ownerComm, nullptr, bytes, buffer);
}

ncclResult_t cocclRegisterBufferForComm(cocclBufferHandle* buffer, ncclComm_t registeredComm) {
  ncclResult_t ret = ncclSuccess;
  if (buffer == nullptr || buffer->block == nullptr || buffer->ownerComm == nullptr || registeredComm == nullptr) {
    return ncclInvalidArgument;
  }

  pthread_mutex_lock(&cocclBufferLock);
  {
    // Some hierarchical primitives reuse the same workspace across intra/inter
    // subcommunicators. This extends an existing buffer with another NCCL
    // registration without changing slice ownership.
    cocclBufferBlockBase* base = static_cast<cocclBufferBlockBase*>(buffer->block);
#if CUDART_VERSION >= 11030
    if (base->backend == BufferBackend::Vmm) {
      NCCLCHECKGOTO(vmmRegisterBufferForComm(buffer, registeredComm), ret, fail);
      goto exit;
    }
#endif
    NCCLCHECKGOTO(legacyRegisterBufferForComm(buffer, registeredComm), ret, fail);
  }

exit:
  pthread_mutex_unlock(&cocclBufferLock);
  return ret;
fail:
  goto exit;
}

ncclResult_t cocclReleaseBuffer(cocclBufferHandle* buffer, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;
  if (buffer == nullptr) return ncclSuccess;
  if (buffer->slice == nullptr || buffer->block == nullptr) {
    *buffer = {};
    return ncclSuccess;
  }

  pthread_mutex_lock(&cocclBufferLock);
  {
    // Release is asynchronous with respect to GPU work: the backend records an
    // event on stream and marks the slice Pending. A future acquire promotes it
    // to Free only after cudaEventQuery reports completion.
    cocclBufferBlockBase* base = static_cast<cocclBufferBlockBase*>(buffer->block);
#if CUDART_VERSION >= 11030
    if (base->backend == BufferBackend::Vmm) {
      NCCLCHECKGOTO(vmmReleaseBuffer(buffer, stream), ret, fail);
      goto exit;
    }
#endif
    NCCLCHECKGOTO(legacyReleaseBuffer(buffer, stream), ret, fail);
  }

exit:
  if (ret == ncclSuccess) *buffer = {};
  pthread_mutex_unlock(&cocclBufferLock);
  return ret;
fail:
  goto exit;
}
