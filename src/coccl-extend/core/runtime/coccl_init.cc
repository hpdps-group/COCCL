#include "runtime/coccl_init.h"

#include "core/tuning/coccl_autotune.h"
#include "core/memory/coccl_buffer_management.h"
#include "core/runtime/coccl_comm.h"
#include "core/config/coccl_config.h"
#include "core/training/coccl_training_assist.h"
#include "comm.h"
#include "debug.h"

#include <atomic>
#include <dlfcn.h>
#include <pthread.h>
#include <stddef.h>

#include <map>
#include <string>
#include <utility>
#include <vector>

namespace {

pthread_mutex_t cocclInitLock = PTHREAD_MUTEX_INITIALIZER;
std::atomic<bool> cocclRuntimeDisabled{false};
bool compressorHandlesLoaded = false;
std::map<std::string, const cocclCompressorPlugin*> compressorHandles;
std::vector<void*> compressorLibraryHandles;

std::string compressorLibraryPath(const cocclConfig& config,
                                  const std::string& name) {
  std::string path = config.plugins.libraryPath;
  if (!path.empty() && path.back() != '/') path.push_back('/');
  return path + "lib" + name + ".so";
}

const cocclCompressorPlugin* findCompressor(const std::string& name) {
  auto it = compressorHandles.find(name);
  return it == compressorHandles.end() ? nullptr : it->second;
}

ncclResult_t loadCompressorHandlesLocked(const cocclConfig& config) {
  if (compressorHandlesLoaded) return ncclSuccess;

  for (const std::string& name : config.plugins.compressors) {
    const std::string path = compressorLibraryPath(config, name);
    void* library = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr) {
      WARN("COCCL failed to open compressor library %s: %s", path.c_str(),
           dlerror());
      return ncclSystemError;
    }
    dlerror();
    cocclGetCompressorPluginFn getPlugin =
        reinterpret_cast<cocclGetCompressorPluginFn>(
            dlsym(library, COCCL_COMPRESSOR_ENTRY_SYMBOL));
    const char* symbolError = dlerror();
    const cocclCompressorPlugin* compressor =
        symbolError == nullptr && getPlugin != nullptr ? getPlugin() : nullptr;
    char descriptorError[192] = {};
    if (!cocclValidateCompressorPlugin(
            name.c_str(), compressor, descriptorError,
            sizeof(descriptorError))) {
      WARN("COCCL compressor %s has an invalid %s descriptor: %s",
           name.c_str(), COCCL_COMPRESSOR_ENTRY_SYMBOL, descriptorError);
      compressorLibraryHandles.push_back(library);
      return ncclInvalidArgument;
    }
    compressorLibraryHandles.push_back(library);
    compressorHandles.emplace(name, compressor);
    NCCLCHECK(cocclAutotuneRegisterEnabledCompressor(compressor));
  }
  compressorHandlesLoaded = true;
  return ncclSuccess;
}

class cocclConfigViewStorage {
 public:
  explicit cocclConfigViewStorage(const cocclConfigValues& values) {
    pairs_.reserve(values.size());
    for (const auto& value : values) {
      pairs_.push_back({value.first.c_str(), value.second.c_str()});
    }
  }

  cocclConfigView view() const {
    return {pairs_.empty() ? nullptr : pairs_.data(), pairs_.size()};
  }

 private:
  std::vector<cocclConfigPair> pairs_;
};

ncclResult_t parseCompressorConfig(
    ncclComm_t comm, const cocclCompressorPlugin* compressor,
    const cocclCompressorPolicyEntry& configured,
    cocclCompressorConfigVariant variant, void** parsedConfig) {
  const cocclConfigValues& values =
      variant == cocclCompressorConfigHierarchical
          ? configured.hierarchicalValues
          : configured.defaultValues;
  cocclConfigViewStorage storage(values);
  const cocclConfigView view = storage.view();
  const cocclCompressorConfigContext context = {
      variant, comm->nNodes, comm->localRanks};
  char error[256] = {};
  ncclResult_t result = compressor->parseConfig(
      &view, &context, parsedConfig, error, sizeof(error));
  if (result != ncclSuccess && error[0] != '\0') {
    WARN("COCCL compressor %s configuration error: %s",
         compressor->name, error);
  }
  return result;
}

ncclResult_t createConfiguredCompressor(
    ncclComm_t comm, cocclCompressorConfigVariant variant,
    const cocclCompressorPolicyEntry& configured,
    cocclCompressorHandle* handle) {
  const cocclCompressorPlugin* compressor = findCompressor(configured.name);
  if (compressor == nullptr) return ncclInvalidArgument;
  void* parsedConfig = nullptr;
  ncclResult_t result =
      parseCompressorConfig(comm, compressor, configured, variant,
                            &parsedConfig);
  if (result != ncclSuccess) {
    if (parsedConfig != nullptr) compressor->destroyConfig(parsedConfig);
    WARN("COCCL compressor %s rejected its %s configuration",
         configured.name.c_str(),
         variant == cocclCompressorConfigHierarchical
             ? "hierarchical" : "default");
    return result;
  }
  result =
      cocclCreateCompressorHandle(comm, compressor, parsedConfig, handle);
  if (result != ncclSuccess && parsedConfig != nullptr) {
    compressor->destroyConfig(parsedConfig);
  }
  return result;
}

bool trainingRoleForScope(cocclPolicyScope scope, cocclTrainingRole* role) {
  if (role == nullptr) return false;
  switch (scope) {
    case cocclPolicyScope::Normal:
      *role = cocclTrainingRoleUnknown;
      return true;
    case cocclPolicyScope::DataParallel:
      *role = cocclTrainingRoleDataParallel;
      return true;
    case cocclPolicyScope::TensorParallel:
      *role = cocclTrainingRoleTensorParallel;
      return true;
    case cocclPolicyScope::PipelineParallel:
      *role = cocclTrainingRolePipelineParallel;
      return true;
    default:
      return false;
  }
}

ncclResult_t loadPolicy(
    ncclComm_t comm, cocclTrainingRole role, cocclPolicyKey key,
    const cocclPrimitivePolicy& policy,
    cocclCompressorConfigVariant variant) {
  if (policy.compressor.name.empty()) return ncclSuccess;
  cocclCompressorHandle handle;
  NCCLCHECK(createConfiguredCompressor(
      comm, variant, policy.compressor, &handle));
  return cocclCommSetCompressorPolicy(
      comm, role, key, policy.thresholdBytes, handle);
}

ncclResult_t loadConfiguredPolicy(
    ncclComm_t comm, const cocclConfigPolicyView& configured) {
  if (configured.policy == nullptr) return ncclInvalidArgument;
  cocclTrainingRole role = cocclTrainingRoleUnknown;
  if (!trainingRoleForScope(configured.scope, &role)) {
    return ncclInvalidArgument;
  }
  const cocclPrimitivePolicy& policy = *configured.policy;
  NCCLCHECK(loadPolicy(comm, role, configured.key, policy,
                       cocclCompressorConfigDefault));
  if (!configured.usesHierarchicalConfig ||
      policy.compressor.name.empty()) {
    return ncclSuccess;
  }

  const cocclPolicyKey hierarchical =
      cocclHierarchicalPolicy(configured.key.operation);
  return policy.compressor.hasHierarchicalConfig
      ? loadPolicy(comm, role, hierarchical, policy,
                   cocclCompressorConfigHierarchical)
      : cocclCommCopyCompressorPolicy(
            comm, role, hierarchical, configured.key);
}

ncclResult_t loadCompressorsForComm(ncclComm_t comm,
                                    const cocclConfig& config) {
  for (const cocclConfigPolicyView& configured :
       cocclEnumeratePolicies(config)) {
    if (configured.mode != config.runtime.mode) continue;
    NCCLCHECK(loadConfiguredPolicy(comm, configured));
  }
  return ncclSuccess;
}

void detachFailedCommLocked(
    ncclComm_t comm, bool* registryEmpty,
    cocclCommDetachedResources** detachedResources) {
  cocclTrainingAssistUnregister(comm);
  (void)cocclCommDestroy(comm, registryEmpty, detachedResources);
}

}  // namespace

bool cocclCompressionEnabled() {
  return !cocclRuntimeDisabled.load(std::memory_order_acquire);
}

ncclResult_t cocclInit(ncclComm_t comm) {
  if (cocclRuntimeDisabled.load(std::memory_order_acquire)) {
    return ncclSuccess;
  }
  if (!cocclConfigInitialize()) {
    cocclRuntimeDisabled.store(true, std::memory_order_release);
    return ncclSuccess;
  }
  if (comm == nullptr) return ncclInvalidArgument;

  const cocclConfig& config = cocclGetConfig();
  ncclResult_t ret = ncclSuccess;
  bool registryEmpty = false;
  cocclCommDetachedResources* detachedResources = nullptr;

  pthread_mutex_lock(&cocclInitLock);
  if (cocclCommCommitted(comm)) {
    pthread_mutex_unlock(&cocclInitLock);
    return ncclSuccess;
  }
  NCCLCHECKGOTO(loadCompressorHandlesLocked(config), ret, fail_locked);
  NCCLCHECKGOTO(cocclCommCreate(comm), ret, fail_locked);
  cocclTrainingAssistRegister(comm);
  NCCLCHECKGOTO(cocclBufferCommInit(comm), ret, fail_locked);
  NCCLCHECKGOTO(loadCompressorsForComm(comm, config), ret, fail_locked);
  pthread_mutex_unlock(&cocclInitLock);

  // Profiling is collective and must not run while the process init lock is
  // held. Completed global model categories are O(1) no-ops on later comms.
  ret = cocclAutotuneEnsureGlobalModels(comm);
  if (ret != ncclSuccess && comm->rank == 0) {
    WARN("COCCL autotune initialization failed with %d; using heuristics", ret);
  }

  pthread_mutex_lock(&cocclInitLock);
  ret = cocclCommCommit(comm);
  if (ret == ncclSuccess) {
    pthread_mutex_unlock(&cocclInitLock);
    return ncclSuccess;
  }

fail_locked:
  detachFailedCommLocked(comm, &registryEmpty, &detachedResources);
  pthread_mutex_unlock(&cocclInitLock);
  (void)cocclCommDestroyDetachedResources(detachedResources);
  (void)cocclBufferCommDestroy(comm);
  if (registryEmpty) (void)cocclBufferDestroyAll();
  cocclRuntimeDisabled.store(true, std::memory_order_release);
  if (comm->rank == 0) {
    WARN("COCCL initialization failed with %d; falling back to native NCCL",
         ret);
  }
  return ncclSuccess;
}

ncclResult_t cocclDestroy(ncclComm_t comm) {
  ncclResult_t ret = ncclSuccess;
  bool registryEmpty = false;
  cocclCommDetachedResources* detachedResources = nullptr;

  cocclTrainingAssistUnregister(comm);
  pthread_mutex_lock(&cocclInitLock);
  ret = cocclCommDestroy(comm, &registryEmpty, &detachedResources);
  pthread_mutex_unlock(&cocclInitLock);

  ncclResult_t destroyRet =
      cocclCommDestroyDetachedResources(detachedResources);
  if (ret == ncclSuccess) ret = destroyRet;

  pthread_mutex_lock(&cocclInitLock);
  ncclResult_t bufferRet = cocclBufferCommDestroy(comm);
  if (ret == ncclSuccess) ret = bufferRet;
  if (registryEmpty) {
    bufferRet = cocclBufferDestroyAll();
    if (ret == ncclSuccess) ret = bufferRet;
  }
  pthread_mutex_unlock(&cocclInitLock);
  return ret;
}
