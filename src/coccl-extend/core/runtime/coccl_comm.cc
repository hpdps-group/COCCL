#include "core/runtime/coccl_comm.h"

#include "checks.h"
#include "comm.h"

#include <map>
#include <memory>
#include <pthread.h>
#include <utility>

namespace {

struct cocclCompressorPolicyKey {
  cocclTrainingRole role = cocclTrainingRoleUnknown;
  cocclPolicyKey policy;
};

bool operator<(const cocclCompressorPolicyKey& lhs,
               const cocclCompressorPolicyKey& rhs) {
  if (lhs.role != rhs.role) return lhs.role < rhs.role;
  return lhs.policy < rhs.policy;
}

// A configured policy owns one immutable plugin context and its routing
// threshold. A missing handle is the single source of truth for
// "compression disabled"; no parallel enable flag or threshold map is kept.
struct cocclCompressorPolicyContext {
  cocclCompressorHandle compressor;
  size_t thresholdBytes = 0;
};

// Private subcommunicators used by hierarchical primitives. They are created on
// first use because many jobs enable COCCL but never enter these paths.
struct cocclHierarchicalCommState {
  ncclComm_t intraComm = nullptr;
  ncclComm_t interComm = nullptr;
};

// COCCL's communicator extension. It intentionally stays private to this
// translation unit so NCCL core code only depends on lifecycle/query APIs.
struct cocclComm {
  // Lifecycle readiness only. Per-operation compression availability is
  // represented exclusively by the configured compressor policies below.
  bool committed = false;

  std::map<cocclCompressorPolicyKey, cocclCompressorPolicyContext> compressors;

  // Hierarchical ReduceScatter/AllReduce primitives use these private
  // subcommunicators for intra-node and inter-node exchanges.
  cocclHierarchicalCommState hierarchicalComms;
};

pthread_mutex_t cocclCommLock = PTHREAD_MUTEX_INITIALIZER;
// NCCL core owns ncclComm; COCCL keeps side-car state in this registry so
// struct ncclComm stays untouched.
std::map<ncclComm_t, std::unique_ptr<cocclComm>> cocclCommRegistry;

static cocclComm* findCocclCommLocked(ncclComm_t comm) {
  auto it = cocclCommRegistry.find(comm);
  return it == cocclCommRegistry.end() ? nullptr : it->second.get();
}

static const cocclCompressorPolicyContext* configuredCompressor(
    cocclComm* coccl, cocclTrainingRole role, cocclPolicyKey key) {
  if (coccl == nullptr) return nullptr;
  auto configured = coccl->compressors.find({role, key});
  return configured == coccl->compressors.end() ||
                 !configured->second.compressor
      ? nullptr
      : &configured->second;
}

static bool validPolicyKey(cocclPolicyKey key) {
  return cocclOperationSupportsPolicy(
      cocclGetOperationDescriptor(key.operation), key.variant);
}

static ncclResult_t destroyHierarchicalComms(cocclHierarchicalCommState* hierarchy);

static void recordNcclCleanup(ncclResult_t status, ncclResult_t* ret) {
  if (status == ncclSuccess) return;
  if (ret != nullptr && *ret == ncclSuccess) *ret = status;
}

static ncclResult_t createHierarchicalComms(ncclComm_t ownerComm,
                                            cocclHierarchicalCommState* hierarchy) {
  // rank/localRanks groups same-node ranks; rank%localRanks groups matching
  // local ranks across nodes for the inter-node phase.
  if (ownerComm == nullptr || hierarchy == nullptr) return ncclInvalidArgument;

  ncclResult_t ret = ncclSuccess;
  int localRanks = ownerComm->localRanks;
  CUDACHECKGOTO(cudaSetDevice(ownerComm->cudaDev), ret, fail);
  NCCLCHECKGOTO(ncclCommSplit(ownerComm, ownerComm->rank / localRanks, ownerComm->rank,
                              &hierarchy->intraComm, NULL),
                ret, fail);
  cocclTrainingAssistUnregister(hierarchy->intraComm);
  NCCLCHECKGOTO(ncclCommSplit(ownerComm, ownerComm->rank % localRanks, ownerComm->rank,
                              &hierarchy->interComm, NULL),
                ret, fail);
  cocclTrainingAssistUnregister(hierarchy->interComm);

exit:
  return ret;
fail:
  (void)destroyHierarchicalComms(hierarchy);
  goto exit;
}

static ncclResult_t destroyHierarchicalComms(cocclHierarchicalCommState* hierarchy) {
  ncclResult_t ret = ncclSuccess;
  if (hierarchy == nullptr) return ncclSuccess;

  if (hierarchy->intraComm != nullptr) {
    recordNcclCleanup(ncclCommDestroy(hierarchy->intraComm), &ret);
    hierarchy->intraComm = nullptr;
  }
  if (hierarchy->interComm != nullptr) {
    recordNcclCleanup(ncclCommDestroy(hierarchy->interComm), &ret);
    hierarchy->interComm = nullptr;
  }

  return ret;
}

}

struct cocclCommDetachedResources {
  // Owns the removed registry entry until runtime can safely destroy CUDA/NCCL
  // resources outside cocclCommLock.
  std::unique_ptr<cocclComm> state;
};

ncclResult_t cocclCommCreate(ncclComm_t comm) {
  if (comm == nullptr) return ncclInvalidArgument;

  pthread_mutex_lock(&cocclCommLock);
  if (findCocclCommLocked(comm) == nullptr) {
    std::unique_ptr<cocclComm> state(new cocclComm());
    cocclCommRegistry.emplace(comm, std::move(state));
  }
  pthread_mutex_unlock(&cocclCommLock);
  return ncclSuccess;
}

ncclResult_t cocclCommDestroy(ncclComm_t comm, bool* registryEmpty,
                              cocclCommDetachedResources** detachedResources) {
  // Remove the side-car state quickly under lock. Actual split-comm and CUDA
  // cleanup happens later through cocclCommDestroyDetachedResources().
  std::unique_ptr<cocclComm> detachedState;

  pthread_mutex_lock(&cocclCommLock);
  auto it = cocclCommRegistry.find(comm);
  if (it != cocclCommRegistry.end()) {
    detachedState = std::move(it->second);
    cocclCommRegistry.erase(it);
  }
  bool empty = cocclCommRegistry.empty();
  pthread_mutex_unlock(&cocclCommLock);

  if (registryEmpty != nullptr) *registryEmpty = empty;
  if (detachedResources != nullptr) {
    if (detachedState != nullptr) {
      *detachedResources = new cocclCommDetachedResources{std::move(detachedState)};
    } else {
      *detachedResources = nullptr;
    }
  }
  return ncclSuccess;
}

ncclResult_t cocclCommDestroyDetachedResources(cocclCommDetachedResources* detachedResources) {
  ncclResult_t ret = ncclSuccess;
  if (detachedResources == nullptr) return ncclSuccess;

  cocclComm* coccl = detachedResources->state.get();
  if (coccl != nullptr) {
    // Compressor instances may own persistent pool slices. Drop their handles
    // before destroying split communicators, because teardown of the final
    // split comm may release the shared device pool.
    coccl->compressors.clear();
    recordNcclCleanup(destroyHierarchicalComms(&coccl->hierarchicalComms), &ret);
  }

  delete detachedResources;
  return ret;
}

ncclResult_t cocclCommCommit(ncclComm_t comm) {
  ncclResult_t ret = ncclSuccess;
  pthread_mutex_lock(&cocclCommLock);
  cocclComm* coccl = findCocclCommLocked(comm);
  if (coccl == nullptr) {
    ret = ncclInvalidArgument;
    goto exit;
  }
  coccl->committed = true;

exit:
  pthread_mutex_unlock(&cocclCommLock);
  return ret;
}

bool cocclCommCommitted(ncclComm_t comm) {
  pthread_mutex_lock(&cocclCommLock);
  cocclComm* coccl = findCocclCommLocked(comm);
  bool committed = coccl != nullptr && coccl->committed;
  pthread_mutex_unlock(&cocclCommLock);
  return committed;
}

ncclResult_t cocclCommSetCompressorPolicy(
    ncclComm_t comm, cocclTrainingRole role, cocclPolicyKey key,
    size_t thresholdBytes, const cocclCompressorHandle& compressor) {
  if (!compressor || !validPolicyKey(key)) return ncclInvalidArgument;
  ncclResult_t ret = ncclSuccess;
  pthread_mutex_lock(&cocclCommLock);
  cocclComm* coccl = findCocclCommLocked(comm);
  if (coccl == nullptr) {
    ret = ncclInvalidArgument;
    goto exit;
  }
  coccl->compressors.insert_or_assign(
      cocclCompressorPolicyKey{role, key},
      cocclCompressorPolicyContext{compressor, thresholdBytes});

exit:
  pthread_mutex_unlock(&cocclCommLock);
  return ret;
}

ncclResult_t cocclCommCopyCompressorPolicy(
    ncclComm_t comm, cocclTrainingRole role, cocclPolicyKey destination,
    cocclPolicyKey source) {
  ncclResult_t ret = ncclSuccess;
  pthread_mutex_lock(&cocclCommLock);
  cocclComm* coccl = findCocclCommLocked(comm);
  if (coccl == nullptr) {
    ret = ncclInvalidArgument;
    goto exit;
  }
  {
    const cocclCompressorPolicyContext* configured =
        configuredCompressor(coccl, role, source);
    if (configured == nullptr) {
      ret = ncclInvalidUsage;
      goto exit;
    }
    coccl->compressors.insert_or_assign({role, destination}, *configured);
  }

exit:
  pthread_mutex_unlock(&cocclCommLock);
  return ret;
}

ncclResult_t cocclCommResolveCompressorPolicy(
    ncclComm_t comm, cocclTrainingRole role, cocclPolicyKey key,
    cocclResolvedCompressorPolicy* resolved) {
  if (comm == nullptr || resolved == nullptr) {
    return ncclInvalidArgument;
  }
  *resolved = {};

  ncclResult_t ret = ncclSuccess;
  pthread_mutex_lock(&cocclCommLock);
  cocclComm* coccl = findCocclCommLocked(comm);
  if (coccl == nullptr) {
    ret = ncclInvalidArgument;
  } else if (!coccl->committed) {
    ret = ncclInvalidUsage;
  } else {
    const cocclCompressorPolicyContext* configured =
        configuredCompressor(coccl, role, key);
    if (configured == nullptr) {
      ret = ncclInvalidUsage;
    } else {
      resolved->compressor = configured->compressor;
      resolved->thresholdBytes = configured->thresholdBytes;
    }
  }
  pthread_mutex_unlock(&cocclCommLock);
  return ret;
}

ncclResult_t cocclCommGetHierarchicalComms(ncclComm_t comm, cocclHierarchicalComms* resource) {
  // Keep ncclCommSplit outside cocclCommLock; split creation can trigger NCCL
  // init/destroy callbacks that need to re-enter the COCCL registry.
  if (comm == nullptr || resource == nullptr) return ncclInvalidArgument;

  ncclResult_t ret = ncclSuccess;
  *resource = {};

  pthread_mutex_lock(&cocclCommLock);
  cocclComm* coccl = findCocclCommLocked(comm);
  if (coccl == nullptr) {
    ret = ncclInvalidArgument;
  } else if (coccl->hierarchicalComms.intraComm != nullptr) {
    resource->ownerComm = comm;
    resource->intraComm = coccl->hierarchicalComms.intraComm;
    resource->interComm = coccl->hierarchicalComms.interComm;
  }
  pthread_mutex_unlock(&cocclCommLock);
  if (ret != ncclSuccess) return ret;
  if (resource->intraComm != nullptr) return ncclSuccess;

  cocclHierarchicalCommState hierarchy = {};
  NCCLCHECK(createHierarchicalComms(comm, &hierarchy));

  pthread_mutex_lock(&cocclCommLock);
  coccl->hierarchicalComms = hierarchy;
  resource->ownerComm = comm;
  resource->intraComm = hierarchy.intraComm;
  resource->interComm = hierarchy.interComm;
  pthread_mutex_unlock(&cocclCommLock);
  return ncclSuccess;
}

ncclResult_t cocclCommGetCompressor(
    ncclComm_t comm, cocclPolicyKey key,
    cocclCompressorHandle* compressor) {
  if (comm == nullptr || compressor == nullptr) return ncclInvalidArgument;
  *compressor = {};

  // training_assist remains the sole authority for role selection. Unknown
  // training communicators cannot accidentally use a normal-mode policy.
  cocclTrainingRole role = cocclTrainingRoleUnknown;
  if (cocclTrainingAssistEnabled()) {
    cocclTrainingClassification classification;
    if (!cocclTrainingAssistQuery(comm, &classification) ||
        classification.role == cocclTrainingRoleUnknown) {
      return ncclInvalidUsage;
    }
    role = classification.role;
  }

  ncclResult_t ret = ncclSuccess;
  cocclResolvedCompressorPolicy resolved;
  ret = cocclCommResolveCompressorPolicy(comm, role, key, &resolved);
  if (ret == ncclSuccess) *compressor = std::move(resolved.compressor);
  return ret;
}
