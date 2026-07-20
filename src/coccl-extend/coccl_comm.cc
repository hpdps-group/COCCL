#include "coccl_comm.h"

#include "checks.h"
#include "comm.h"

#include <map>
#include <memory>
#include <pthread.h>
#include <stdlib.h>
#include <utility>
#include <vector>

namespace {

struct cocclCompressorContext {
  ncclCompressor_t* compressor = nullptr;
  // Config ownership is shared so op fallback/copy chains can reuse parsed
  // plugin config without double-freeing it during comm teardown.
  std::shared_ptr<void> config;
};

using cocclCompressorChain = std::vector<cocclCompressorContext>;
using cocclTrainingCompressorKey =
    std::pair<cocclTrainingRole, ncclCommOp_t>;

// A configured chain owns both its callbacks and the threshold that controls
// routing to it. A missing or empty chain is the single source of truth for
// "compression disabled"; no parallel enable flag or threshold map is kept.
struct cocclCompressorChainConfig {
  cocclCompressorChain chain;
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
  // represented exclusively by the configured chains below.
  bool committed = false;

  // The default chain comes from NCCL_COMPRESSORS and is reserved for
  // pre-commit profiling. Committed communication always requires an explicit
  // normal op or training role/op chain.
  cocclCompressorChain defaultCompressorChain;
  std::map<ncclCommOp_t, cocclCompressorChainConfig> compressorChains;
  std::map<cocclTrainingCompressorKey, cocclCompressorChainConfig>
      trainingCompressorChains;

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

static std::shared_ptr<void> ownCompressorConfig(void* config) {
  if (config == nullptr) return std::shared_ptr<void>();
  // Compressor parseConfig allocates with malloc-compatible ownership today.
  return std::shared_ptr<void>(config, [](void* ptr) { free(ptr); });
}

static const cocclCompressorChainConfig* configuredCompressorChainForOp(
    cocclComm* coccl, ncclCommOp_t op) {
  if (coccl == nullptr) return nullptr;
  auto chain = coccl->compressorChains.find(op);
  return chain == coccl->compressorChains.end() || chain->second.chain.empty()
      ? nullptr : &chain->second;
}

static const cocclCompressorChainConfig* trainingCompressorChainForRoleAndOp(
    cocclComm* coccl, cocclTrainingRole role, ncclCommOp_t op) {
  if (coccl == nullptr || role == cocclTrainingRoleUnknown) return nullptr;
  auto chain = coccl->trainingCompressorChains.find({role, op});
  return chain == coccl->trainingCompressorChains.end() ||
                 chain->second.chain.empty()
             ? nullptr
             : &chain->second;
}

// This NCCL ABI has no FP8 datatype. Future FP8 enum values belong in this
// centralized predicate without changing operation-specific routing code.
static bool compressionDatatypeSupported(ncclDataType_t datatype) {
  if (datatype == ncclFloat16 || datatype == ncclFloat32) return true;
#if defined(__CUDA_BF16_TYPES_EXIST__)
  return datatype == ncclBfloat16;
#else
  return false;
#endif
}

static bool operationRequiresReduction(ncclCommOp_t op) {
  return op == AllReduce || op == AllReduce_Inter ||
         op == ReduceScatter || op == ReduceScatter_Inter;
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

ncclResult_t cocclCommAppendDefaultCompressor(ncclComm_t comm, ncclCompressor_t* compressor, void* config) {
  if (compressor == nullptr) return ncclInvalidArgument;

  ncclResult_t ret = ncclSuccess;
  pthread_mutex_lock(&cocclCommLock);
  cocclComm* coccl = findCocclCommLocked(comm);
  if (coccl == nullptr) {
    ret = ncclInvalidArgument;
    goto exit;
  }
  coccl->defaultCompressorChain.push_back({compressor, ownCompressorConfig(config)});

exit:
  pthread_mutex_unlock(&cocclCommLock);
  return ret;
}

ncclResult_t cocclCommResetOpChain(
    ncclComm_t comm, ncclCommOp_t op, size_t thresholdBytes) {
  ncclResult_t ret = ncclSuccess;
  pthread_mutex_lock(&cocclCommLock);
  cocclComm* coccl = findCocclCommLocked(comm);
  if (coccl == nullptr) {
    ret = ncclInvalidArgument;
    goto exit;
  }
  {
    cocclCompressorChainConfig& chain = coccl->compressorChains[op];
    chain.chain.clear();
    chain.thresholdBytes = thresholdBytes;
  }

exit:
  pthread_mutex_unlock(&cocclCommLock);
  return ret;
}

ncclResult_t cocclCommAppendOpCompressor(ncclComm_t comm, ncclCommOp_t op, ncclCompressor_t* compressor, void* config) {
  if (compressor == nullptr) return ncclInvalidArgument;

  ncclResult_t ret = ncclSuccess;
  pthread_mutex_lock(&cocclCommLock);
  cocclComm* coccl = findCocclCommLocked(comm);
  if (coccl == nullptr) {
    ret = ncclInvalidArgument;
    goto exit;
  }
  coccl->compressorChains[op].chain.push_back(
      {compressor, ownCompressorConfig(config)});

exit:
  pthread_mutex_unlock(&cocclCommLock);
  return ret;
}

ncclResult_t cocclCommCopyOpChain(ncclComm_t comm, ncclCommOp_t dstOp, ncclCommOp_t srcOp) {
  ncclResult_t ret = ncclSuccess;
  pthread_mutex_lock(&cocclCommLock);
  cocclComm* coccl = findCocclCommLocked(comm);
  if (coccl == nullptr) {
    ret = ncclInvalidArgument;
    goto exit;
  }
  {
    const cocclCompressorChainConfig* source =
        configuredCompressorChainForOp(coccl, srcOp);
    if (source == nullptr) {
      ret = ncclInvalidUsage;
      goto exit;
    }
    coccl->compressorChains.insert_or_assign(dstOp, *source);
  }

exit:
  pthread_mutex_unlock(&cocclCommLock);
  return ret;
}

ncclResult_t cocclCommResetTrainingRoleChain(
    ncclComm_t comm, cocclTrainingRole role, ncclCommOp_t op,
    size_t thresholdBytes) {
  if (role == cocclTrainingRoleUnknown) return ncclInvalidArgument;

  ncclResult_t ret = ncclSuccess;
  pthread_mutex_lock(&cocclCommLock);
  cocclComm* coccl = findCocclCommLocked(comm);
  if (coccl == nullptr) {
    ret = ncclInvalidArgument;
    goto exit;
  }
  {
    cocclCompressorChainConfig& chain =
        coccl->trainingCompressorChains[{role, op}];
    chain.chain.clear();
    chain.thresholdBytes = thresholdBytes;
  }

exit:
  pthread_mutex_unlock(&cocclCommLock);
  return ret;
}

ncclResult_t cocclCommAppendTrainingRoleCompressor(
    ncclComm_t comm, cocclTrainingRole role, ncclCommOp_t op,
    ncclCompressor_t* compressor, void* config) {
  if (role == cocclTrainingRoleUnknown || compressor == nullptr) {
    return ncclInvalidArgument;
  }

  ncclResult_t ret = ncclSuccess;
  pthread_mutex_lock(&cocclCommLock);
  cocclComm* coccl = findCocclCommLocked(comm);
  if (coccl == nullptr) {
    ret = ncclInvalidArgument;
    goto exit;
  }
  coccl->trainingCompressorChains[{role, op}].chain.push_back(
      {compressor, ownCompressorConfig(config)});

exit:
  pthread_mutex_unlock(&cocclCommLock);
  return ret;
}

bool cocclCommAvailable(ncclComm_t comm) {
  bool available = false;
  pthread_mutex_lock(&cocclCommLock);
  cocclComm* coccl = findCocclCommLocked(comm);
  available = coccl != nullptr && coccl->committed;
  pthread_mutex_unlock(&cocclCommLock);
  return available;
}

bool cocclCommShouldCompress(
    ncclComm_t comm, ncclCommOp_t compressorOp, cocclTrainingRole role,
    size_t totalBytes, ncclDataType_t datatype, ncclRedOp_t reductionOp) {
  if (comm == nullptr || comm->nRanks <= 1 ||
      !compressionDatatypeSupported(datatype)) {
    return false;
  }

  if (operationRequiresReduction(compressorOp) && reductionOp != ncclSum) {
    return false;
  }

  bool chainConfigured = false;
  size_t thresholdBytes = 0;
  pthread_mutex_lock(&cocclCommLock);
  cocclComm* coccl = findCocclCommLocked(comm);
  if (coccl != nullptr && coccl->committed) {
    const cocclCompressorChainConfig* chain = role == cocclTrainingRoleUnknown
        ? configuredCompressorChainForOp(coccl, compressorOp)
        : trainingCompressorChainForRoleAndOp(coccl, role, compressorOp);
    if (chain != nullptr) {
      chainConfigured = true;
      thresholdBytes = chain->thresholdBytes;
    }
  }
  pthread_mutex_unlock(&cocclCommLock);

  return chainConfigured && totalBytes > thresholdBytes;
}

ncclResult_t cocclCommGetHierarchicalComms(ncclComm_t comm, cocclHierarchicalComms* resource) {
  // Keep ncclCommSplit outside cocclCommLock; split creation can trigger NCCL
  // init/destroy callbacks that need to re-enter the COCCL registry.
  if (comm == nullptr || resource == nullptr) return ncclInvalidArgument;

  ncclResult_t ret = ncclSuccess;
  cocclHierarchicalCommState newHierarchy = {};
  bool createdHierarchy = false;
  bool needHierarchy = false;

  *resource = {};

  pthread_mutex_lock(&cocclCommLock);
  {
    cocclComm* coccl = findCocclCommLocked(comm);
    if (coccl == nullptr) {
      ret = ncclInvalidArgument;
    } else {
      needHierarchy = coccl->hierarchicalComms.intraComm == nullptr ||
                      coccl->hierarchicalComms.interComm == nullptr;
    }
  }
  pthread_mutex_unlock(&cocclCommLock);
  if (ret != ncclSuccess) return ret;

  if (needHierarchy) {
    NCCLCHECKGOTO(createHierarchicalComms(comm, &newHierarchy), ret, fail);
    createdHierarchy = true;
  }

  pthread_mutex_lock(&cocclCommLock);
  {
    cocclComm* coccl = findCocclCommLocked(comm);
    if (coccl == nullptr) {
      ret = ncclInvalidArgument;
    } else {
      if ((coccl->hierarchicalComms.intraComm == nullptr ||
           coccl->hierarchicalComms.interComm == nullptr) && createdHierarchy) {
        coccl->hierarchicalComms = newHierarchy;
        newHierarchy = {};
        createdHierarchy = false;
      }
      resource->ownerComm = comm;
      resource->intraComm = coccl->hierarchicalComms.intraComm;
      resource->interComm = coccl->hierarchicalComms.interComm;
    }
  }
  pthread_mutex_unlock(&cocclCommLock);

exit:
  if (createdHierarchy) {
    ncclResult_t cleanupRet = destroyHierarchicalComms(&newHierarchy);
    if (ret == ncclSuccess) ret = cleanupRet;
  }
  return ret;
fail:
  goto exit;
}

ncclResult_t cocclVisitCompressorChain(ncclComm_t comm, ncclCommOp_t op, bool reverse,
                                       cocclCompressorVisitor visitor, void* context) {
  if (visitor == nullptr) return ncclInvalidArgument;

  bool found = false;
  bool runtimeCommitted = false;
  pthread_mutex_lock(&cocclCommLock);
  cocclComm* coccl = findCocclCommLocked(comm);
  if (coccl != nullptr) {
    found = true;
    runtimeCommitted = coccl->committed;
  }
  pthread_mutex_unlock(&cocclCommLock);
  if (!found) {
    WARN("COCCL compressor chain requested for an unregistered communicator %p", comm);
    return ncclInvalidArgument;
  }

  // Keep the training decision authoritative in training_assist. Direct private
  // primitive calls must not reach a default chain while a role is unknown.
  // The exact role/op chain checked below is the sole compression-enable state.
  // Before runtime commit, autotune profiling intentionally uses the default
  // chain because no training role can exist yet.
  cocclTrainingRole role = cocclTrainingRoleUnknown;
  if (runtimeCommitted && cocclTrainingAssistEnabled()) {
    cocclTrainingClassification classification;
    if (!cocclTrainingAssistQuery(comm, &classification) ||
        classification.role == cocclTrainingRoleUnknown) {
      return ncclInvalidUsage;
    }
    role = classification.role;
  }

  // Copy the vector under the lock, then invoke callbacks after releasing it.
  // Compressor callbacks can enqueue CUDA/NCCL work and must not run while the
  // global COCCL comm registry mutex is held.
  cocclCompressorChain chainSnapshot;
  bool snapshotFound = false;
  bool configuredChainFound = true;
  pthread_mutex_lock(&cocclCommLock);
  coccl = findCocclCommLocked(comm);
  if (coccl != nullptr) {
    snapshotFound = true;
    if (role == cocclTrainingRoleUnknown) {
      if (runtimeCommitted) {
        const cocclCompressorChainConfig* normalChain =
            configuredCompressorChainForOp(coccl, op);
        configuredChainFound = normalChain != nullptr;
        if (normalChain != nullptr) chainSnapshot = normalChain->chain;
      } else {
        chainSnapshot = coccl->defaultCompressorChain;
      }
    } else {
      const cocclCompressorChainConfig* trainingChain =
          trainingCompressorChainForRoleAndOp(coccl, role, op);
      configuredChainFound = trainingChain != nullptr;
      if (trainingChain != nullptr) chainSnapshot = trainingChain->chain;
    }
  }
  pthread_mutex_unlock(&cocclCommLock);

  if (!snapshotFound) {
    WARN("COCCL compressor chain requested for an unregistered communicator %p", comm);
    return ncclInvalidArgument;
  }
  if (!configuredChainFound) {
    if (role == cocclTrainingRoleUnknown) {
      WARN("COCCL has no explicitly configured compressor chain for op %d",
           (int)op);
    } else {
      WARN("COCCL training role %s has no configured compressor chain for op %d",
           cocclTrainingRoleName(role), (int)op);
    }
    return ncclInvalidUsage;
  }

  if (reverse) {
    for (auto it = chainSnapshot.rbegin(); it != chainSnapshot.rend(); ++it) {
      NCCLCHECK(visitor(it->compressor, it->config.get(), context));
    }
  } else {
    for (const cocclCompressorContext& compressorContext : chainSnapshot) {
      NCCLCHECK(visitor(compressorContext.compressor, compressorContext.config.get(), context));
    }
  }
  return ncclSuccess;
}
