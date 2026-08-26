#include "core/compression/coccl_compressor_runtime.h"

#include "core/tuning/coccl_autotune.h"
#include "core/config/coccl_config.h"
#include "core/training/coccl_training_assist.h"
#include "comm.h"
#include "compressor_plugin/detail/coccl_compressor_abi.h"
#include "debug.h"

#include <cuda_runtime.h>
#include <dlfcn.h>
#include <pthread.h>

#include <array>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace {

struct PersistentBuffer {
  void* data = nullptr;
  size_t bytes = 0;
};

struct StateEntry {
  void* data = nullptr;
  cocclCompressorDestroyStateFn destroy = nullptr;
};

struct DeviceResources {
  std::map<size_t, PersistentBuffer> persistent;
  std::map<const void*, StateEntry> states;
  size_t scratchPeakBytes = 0;
};

struct CompressorPolicy {
  const cocclCompressorPlugin* plugin = nullptr;
  void* config = nullptr;
  size_t thresholdBytes = 0;
  std::mutex resourceLock;
  std::map<int, DeviceResources> resources;
};

struct LoadedPlugin {
  void* library = nullptr;
  const cocclCompressorPlugin* descriptor = nullptr;
};

struct ExecutionResources {
  CompressorPolicy* policy = nullptr;
  int cudaDev = 0;
  cudaStream_t stream = nullptr;
  std::vector<void*> scratch;
  size_t scratchBytes = 0;
};

constexpr size_t kOperationCount =
    static_cast<size_t>(cocclOperation::Count);
constexpr size_t kPolicyVariantCount = 3;
constexpr size_t kTrainingRoleCount =
    static_cast<size_t>(cocclTrainingRoleCount);
constexpr size_t kCompressionScopeCount =
    static_cast<size_t>(cocclCompressionScope::Count);

pthread_mutex_t compressorLock = PTHREAD_MUTEX_INITIALIZER;
bool runtimeInitialized = false;
ncclResult_t runtimeInitResult = ncclSuccess;
bool runtimeHasPolicies = false;
int runtimeRanks = 1;
int runtimeNodes = 1;
int runtimeDevicesPerNode = 1;
std::map<int, int> rankByDevice;
std::map<int, size_t> communicatorsByDevice;
std::map<std::string, LoadedPlugin> loadedPlugins;
std::vector<std::unique_ptr<CompressorPolicy>> ownedPolicies;
CompressorPolicy* policies[kTrainingRoleCount][kPolicyVariantCount]
                              [kOperationCount][kCompressionScopeCount] = {};

class ConfigViewStorage {
 public:
  explicit ConfigViewStorage(const cocclConfigValues& values) {
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

ncclResult_t allocateScratch(void* opaque, size_t bytes,
                             cocclCompressorBufferView* buffer) {
  ExecutionResources* execution = static_cast<ExecutionResources*>(opaque);
  void* data = nullptr;
  cudaError_t result = cudaMallocAsync(&data, bytes, execution->stream);
  if (result != cudaSuccess) return ncclUnhandledCudaError;
  execution->scratch.push_back(data);
  execution->scratchBytes += bytes;
  *buffer = {data, bytes};
  return ncclSuccess;
}

ncclResult_t acquirePersistent(void* opaque, size_t slot, size_t bytes,
                               cocclCompressorBufferView* buffer) {
  ExecutionResources* execution = static_cast<ExecutionResources*>(opaque);
  CompressorPolicy* policy = execution->policy;
  std::lock_guard<std::mutex> guard(policy->resourceLock);
  PersistentBuffer& persistent =
      policy->resources[execution->cudaDev].persistent[slot];
  if (bytes > persistent.bytes) {
    if (persistent.data != nullptr) {
      cudaError_t result = cudaFree(persistent.data);
      if (result != cudaSuccess) return ncclUnhandledCudaError;
    }
    cudaError_t result = cudaMalloc(&persistent.data, bytes);
    if (result != cudaSuccess) return ncclUnhandledCudaError;
    persistent.bytes = bytes;
    INFO(COCCL_COMPRESS,
         "COCCL compressor %s persistent device %d slot %zu bytes %zu",
         policy->plugin->name, execution->cudaDev, slot, bytes);
  }
  *buffer = {persistent.data, persistent.bytes};
  return ncclSuccess;
}

ncclResult_t getOrCreateState(void* opaque, const void* typeKey,
                              cocclCompressorCreateStateFn createState,
                              cocclCompressorDestroyStateFn destroyState,
                              void** state) {
  ExecutionResources* execution = static_cast<ExecutionResources*>(opaque);
  CompressorPolicy* policy = execution->policy;
  std::lock_guard<std::mutex> guard(policy->resourceLock);
  StateEntry& entry = policy->resources[execution->cudaDev].states[typeKey];
  if (entry.data == nullptr) {
    NCCLCHECK(createState(&entry.data));
    entry.destroy = destroyState;
  }
  *state = entry.data;
  return ncclSuccess;
}

const cocclCompressorHostApi kHostApi = {
    COCCL_COMPRESSOR_HOST_API_VERSION,
    sizeof(cocclCompressorHostApi),
    allocateScratch,
    acquirePersistent,
    getOrCreateState,
};

std::string pluginPath(const cocclConfig& config, const std::string& name) {
  std::string path = config.plugins.libraryPath;
  if (!path.empty() && path.back() != '/') path.push_back('/');
  return path + "lib" + name + ".so";
}

ncclResult_t loadPlugins(const cocclConfig& config) {
  for (const std::string& name : config.plugins.compressors) {
    const std::string path = pluginPath(config, name);
    void* library = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr) {
      WARN("COCCL failed to open compressor %s: %s", path.c_str(), dlerror());
      return ncclSystemError;
    }
    auto entry = reinterpret_cast<cocclGetCompressorPluginFn>(
        dlsym(library, COCCL_COMPRESSOR_ENTRY_SYMBOL));
    const cocclCompressorPlugin* plugin = entry == nullptr ? nullptr : entry();
    char error[192] = {};
    if (!cocclValidateCompressorPlugin(name.c_str(), plugin, error,
                                       sizeof(error))) {
      WARN("COCCL compressor %s rejected: %s", name.c_str(), error);
      dlclose(library);
      return ncclInvalidArgument;
    }
    loadedPlugins.emplace(name, LoadedPlugin{library, plugin});
  }
  return ncclSuccess;
}

ncclResult_t createPolicy(const cocclCompressorScopeEntry& configured,
                          CompressorPolicy** policy) {
  auto plugin = loadedPlugins.find(configured.name);
  if (plugin == loadedPlugins.end()) return ncclInvalidArgument;

  ConfigViewStorage storage(configured.values);
  const cocclConfigView view = storage.view();
  const cocclCompressorConfigContext context = {
      cocclCompressorConfigDefault, runtimeNodes, runtimeDevicesPerNode};
  char error[256] = {};
  void* parsedConfig = nullptr;
  ncclResult_t result = plugin->second.descriptor->parseConfig(
      &view, &context, &parsedConfig, error, sizeof(error));
  if (result != ncclSuccess) {
    WARN("COCCL compressor %s configuration rejected: %s",
         configured.name.c_str(), error);
    return result;
  }

  auto created = std::make_unique<CompressorPolicy>();
  created->plugin = plugin->second.descriptor;
  created->config = parsedConfig;
  *policy = created.get();
  ownedPolicies.push_back(std::move(created));
  return ncclSuccess;
}

ncclResult_t installPolicy(cocclTrainingRole trainingRole,
                           cocclOperation operation,
                           const cocclPrimitivePolicy& configured,
                           cocclPolicyVariant variant) {
  const size_t index = static_cast<size_t>(operation);
  const size_t role = static_cast<size_t>(trainingRole);
  const size_t policyVariant = static_cast<size_t>(variant);
  for (cocclCompressionScope scope : {
           cocclCompressionScope::Default,
           cocclCompressionScope::Intra,
           cocclCompressionScope::Inter}) {
    const size_t scopeIndex = static_cast<size_t>(scope);
    const cocclEffectiveCompressorScope effective =
        cocclEffectiveCompressorScopeFor(configured, scope);
    if (!effective.enabled()) continue;
    runtimeHasPolicies = true;
    if (scope != cocclCompressionScope::Default &&
        effective.source == cocclCompressionScope::Default) {
      policies[role][policyVariant][index][scopeIndex] =
          policies[role][policyVariant][index][static_cast<size_t>(
              cocclCompressionScope::Default)];
      continue;
    }
    NCCLCHECK(createPolicy(
        *effective.entry,
        &policies[role][policyVariant][index][scopeIndex]));
    policies[role][policyVariant][index][scopeIndex]->thresholdBytes =
        configured.thresholdBytes;
  }
  return ncclSuccess;
}

ncclResult_t installCollectivePolicies(
    cocclTrainingRole role, const cocclCollectivePolicies& configured) {
  NCCLCHECK(installPolicy(role, cocclOperation::AllGather,
                          configured.allGather,
                          cocclPolicyVariant::Default));
  NCCLCHECK(installPolicy(role, cocclOperation::ReduceScatter,
                          configured.reduceScatter,
                          cocclPolicyVariant::Default));
  return installPolicy(role, cocclOperation::AllReduce,
                       configured.allReduce,
                       cocclPolicyVariant::Default);
}

ncclResult_t initializeRuntime(const ncclComm_t comm,
                               const cocclConfig& config) {
  runtimeRanks = comm->nRanks;
  runtimeNodes = comm->nNodes;
  runtimeDevicesPerNode = comm->localRanks;
  NCCLCHECK(loadPlugins(config));
  if (config.runtime.mode == cocclRuntimeMode::Normal) {
    NCCLCHECK(installCollectivePolicies(
        cocclTrainingRoleUnknown, config.normal));
    NCCLCHECK(installPolicy(cocclTrainingRoleUnknown,
                            cocclOperation::AllToAll,
                            config.normal.allToAll,
                            cocclPolicyVariant::Default));
    NCCLCHECK(installPolicy(cocclTrainingRoleUnknown,
                            cocclOperation::SendRecv,
                            config.normal.sendRecv,
                            cocclPolicyVariant::Default));
  } else {
    NCCLCHECK(installCollectivePolicies(
        cocclTrainingRoleDataParallel,
        config.trainingPolicies.dataParallel));
    NCCLCHECK(installCollectivePolicies(
        cocclTrainingRoleTensorParallel,
        config.trainingPolicies.tensorParallel));
    NCCLCHECK(installPolicy(
        cocclTrainingRolePipelineParallel,
        cocclOperation::SendRecv,
        config.trainingPolicies.pipelineSendRecvForward,
        cocclPolicyVariant::Forward));
    NCCLCHECK(installPolicy(
        cocclTrainingRolePipelineParallel,
        cocclOperation::SendRecv,
        config.trainingPolicies.pipelineSendRecvBackward,
        cocclPolicyVariant::Backward));
  }

  for (size_t role = 0; role < kTrainingRoleCount; ++role) {
    for (cocclOperation operation : {
             cocclOperation::ReduceScatter, cocclOperation::AllReduce}) {
      CompressorPolicy* previous = nullptr;
      for (cocclCompressionScope scope : {
               cocclCompressionScope::Default,
               cocclCompressionScope::Intra,
               cocclCompressionScope::Inter}) {
        CompressorPolicy* policy =
            policies[role][static_cast<size_t>(cocclPolicyVariant::Default)]
                    [static_cast<size_t>(operation)]
                    [static_cast<size_t>(scope)];
        if (policy != nullptr && policy != previous) {
          NCCLCHECK(cocclAutotuneRegisterEnabledCompressor(
              policy, cocclDefaultPolicy(operation, scope)));
        }
        previous = policy;
      }
    }
  }
  return ncclSuccess;
}


int rankForDevice(int cudaDev) {
  pthread_mutex_lock(&compressorLock);
  auto rank = rankByDevice.find(cudaDev);
  const int value = rank == rankByDevice.end() ? 0 : rank->second;
  pthread_mutex_unlock(&compressorLock);
  return value;
}

ncclResult_t execute(CompressorPolicy* policy,
                     const CompressorPolicy* inputPolicy,
                     cocclCompressorCall* call, int rank,
                     cudaStream_t stream) {
  int cudaDev = 0;
  CUDACHECK(cudaGetDevice(&cudaDev));
  ExecutionResources resources = {policy, cudaDev, stream};
  cocclCompressorExecutionContext execution = {
      sizeof(cocclCompressorExecutionContext), &kHostApi, &resources,
      stream, cudaDev, rank, runtimeRanks, runtimeNodes,
      runtimeDevicesPerNode};
  call->config = policy->config;
  call->execution = &execution;
  call->inputConfig = inputPolicy->config;
  ncclResult_t result = policy->plugin->execute(call);
  if (resources.scratchBytes != 0) {
    std::lock_guard<std::mutex> guard(policy->resourceLock);
    DeviceResources& device = policy->resources[cudaDev];
    if (resources.scratchBytes > device.scratchPeakBytes) {
      device.scratchPeakBytes = resources.scratchBytes;
      INFO(COCCL_COMPRESS, "COCCL compressor %s scratch device %d peak %zu",
           policy->plugin->name, cudaDev, device.scratchPeakBytes);
    }
  }
  for (void* scratch : resources.scratch) {
    const cudaError_t freeResult = cudaFreeAsync(scratch, stream);
    if (result == ncclSuccess && freeResult != cudaSuccess) {
      result = ncclUnhandledCudaError;
    }
  }
  return result;
}


ncclResult_t validateEncodedOutput(const cocclCompressorView& output,
                                   size_t expectedChunks) {
  const size_t typeBytes = ncclTypeSize(output.datatype);
  if (output.data == nullptr || output.chunks != expectedChunks ||
      output.chunks == 0 || output.elements % output.chunks != 0 ||
      (output.datatype != ncclInt8 &&
       output.datatype != COCCL_COMPRESSOR_RAW_PASSTHROUGH) ||
      typeBytes == 0 || output.elements * typeBytes != output.bytes ||
      output.bytes > output.capacityBytes) {
    return ncclInvalidUsage;
  }
  return ncclSuccess;
}


}  // namespace

bool cocclCompressionEnabled() {
  return runtimeInitialized && runtimeInitResult == ncclSuccess &&
      runtimeHasPolicies;
}

ncclResult_t cocclResolveCompressorPolicy(
    cocclTrainingRole trainingRole, cocclPolicyKey key,
    cocclResolvedCompressorPolicy* resolved) {
  const size_t index = static_cast<size_t>(key.operation);
  const size_t role = static_cast<size_t>(trainingRole);
  const size_t variant = static_cast<size_t>(key.variant);
  const size_t scope = static_cast<size_t>(key.scope);
  if (index >= kOperationCount || role >= kTrainingRoleCount ||
      variant >= kPolicyVariantCount ||
      scope >= kCompressionScopeCount) {
    return ncclInvalidUsage;
  }

  CompressorPolicy* policy = policies[role][variant][index][scope];
  if (policy == nullptr) return ncclInvalidUsage;
  resolved->compressor = policy;
  resolved->thresholdBytes = policy->thresholdBytes;
  return ncclSuccess;
}


ncclResult_t cocclGetCompressorEncodedSizeBound(
    void* compressor, cocclCompressorOperation operation,
    size_t elements, size_t chunks, ncclDataType_t datatype,
    size_t* encodedBytes) {
  CompressorPolicy* policy = static_cast<CompressorPolicy*>(compressor);
  if (policy == nullptr) return ncclInvalidUsage;
  return cocclQueryCompressorEncodedSizeBound(
      policy->plugin, policy->config, operation, elements, chunks,
      datatype, encodedBytes);
}


bool cocclCompressorSupports(
    void* compressor, cocclCompressorCapability capability) {
  const CompressorPolicy* policy = static_cast<CompressorPolicy*>(compressor);
  return policy != nullptr &&
      (policy->plugin->capabilities & (uint64_t)capability) != 0;
}

const cocclCompressorPlugin* cocclCompressorDescriptor(void* compressor) {
  return static_cast<CompressorPolicy*>(compressor)->plugin;
}

ncclResult_t cocclExecuteCompressor(
    void* compressor, void* inputCompressor,
    cocclCompressorOperation operation,
    const cocclCompressorView& input, cocclCompressorView* output, int rank,
    size_t reduceChunks, ncclDataType_t originalDatatype,
    size_t originalElements, cudaStream_t stream) {
  CompressorPolicy* policy = static_cast<CompressorPolicy*>(compressor);
  CompressorPolicy* inputPolicy =
      static_cast<CompressorPolicy*>(inputCompressor);
  if (policy == nullptr || inputPolicy == nullptr || output == nullptr) {
    return ncclInvalidArgument;
  }
  if (rank < 0) {
    int cudaDev = 0;
    CUDACHECK(cudaGetDevice(&cudaDev));
    rank = rankForDevice(cudaDev);
  }
  cocclCompressorCall call = {
      sizeof(cocclCompressorCall), operation, input, output, rank,
      reduceChunks, originalDatatype, originalElements, nullptr, nullptr,
      nullptr};
  NCCLCHECK(execute(policy, inputPolicy, &call, rank, stream));
  if (operation == cocclCompressorOperationCompress) {
    return validateEncodedOutput(*output, input.chunks);
  }
  if (operation == cocclCompressorOperationDecompressReduceCompress) {
    return validateEncodedOutput(*output, input.chunks / reduceChunks);
  }
  return ncclSuccess;
}

ncclResult_t cocclCompressorRuntimeInit(const ncclComm_t comm) {
  const bool configReady = cocclConfigInitialize();
  pthread_mutex_lock(&compressorLock);
  if (!runtimeInitialized) {
    runtimeInitialized = true;
    runtimeInitResult = configReady
        ? initializeRuntime(comm, cocclGetConfig()) : ncclSuccess;
  }
  const ncclResult_t result = runtimeInitResult;
  if (result == ncclSuccess && configReady) {
    rankByDevice[comm->cudaDev] = comm->rank;
    ++communicatorsByDevice[comm->cudaDev];
  }
  pthread_mutex_unlock(&compressorLock);
  if (result != ncclSuccess) return result;
  if (!configReady) return ncclSuccess;

  cocclTrainingAssistRegister(comm);

  const ncclResult_t autotuneResult =
      cocclAutotuneEnsureGlobalModels(comm);
  if (autotuneResult != ncclSuccess && comm->rank == 0) {
    WARN("COCCL autotune profiling failed with %d; using heuristics",
         autotuneResult);
  }
  return ncclSuccess;
}

ncclResult_t cocclCompressorRuntimeDestroy(const ncclComm_t comm) {
  cocclTrainingAssistUnregister(comm);
  pthread_mutex_lock(&compressorLock);
  auto count = communicatorsByDevice.find(comm->cudaDev);
  if (count == communicatorsByDevice.end()) {
    pthread_mutex_unlock(&compressorLock);
    return ncclSuccess;
  }
  if (--count->second != 0) {
    pthread_mutex_unlock(&compressorLock);
    return ncclSuccess;
  }

  communicatorsByDevice.erase(count);
  rankByDevice.erase(comm->cudaDev);
  ncclResult_t result = ncclSuccess;
  for (const std::unique_ptr<CompressorPolicy>& policy : ownedPolicies) {
    std::lock_guard<std::mutex> guard(policy->resourceLock);
    auto resources = policy->resources.find(comm->cudaDev);
    if (resources == policy->resources.end()) continue;
    size_t persistentBytes = 0;
    for (const auto& state : resources->second.states) {
      state.second.destroy(state.second.data);
    }
    for (const auto& persistent : resources->second.persistent) {
      persistentBytes += persistent.second.bytes;
      if (cudaFree(persistent.second.data) != cudaSuccess &&
          result == ncclSuccess) {
        result = ncclUnhandledCudaError;
      }
    }
    INFO(COCCL_COMPRESS,
         "COCCL compressor %s release device %d persistent %zu scratch_peak %zu states %zu",
         policy->plugin->name, comm->cudaDev, persistentBytes,
         resources->second.scratchPeakBytes, resources->second.states.size());
    policy->resources.erase(resources);
  }
  pthread_mutex_unlock(&compressorLock);
  return result;
}
