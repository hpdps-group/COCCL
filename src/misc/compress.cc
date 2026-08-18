#include "compress.h"
#include "coccl_buffer_management.h"

#include "coccl_config.h"
#include "comm.h"
#include "compressor_plugin/detail/coccl_compressor_abi.h"
#include "debug.h"
#include "reduce_extend.h"

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

bool enableAllToAllComp = false;
bool enableAllReduceComp = false;
bool enableAllGatherComp = false;
bool enableReduceScatterComp = false;
bool enableSendRecvComp = false;
bool enableCheck = false;
bool enableTimer = false;
size_t CompEnableThreshold = 0;

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

pthread_mutex_t compressorLock = PTHREAD_MUTEX_INITIALIZER;
bool runtimeInitialized = false;
ncclResult_t runtimeInitResult = ncclSuccess;
int runtimeRanks = 1;
int runtimeNodes = 1;
int runtimeDevicesPerNode = 1;
std::map<int, int> rankByDevice;
std::map<int, size_t> communicatorsByDevice;
std::map<std::string, LoadedPlugin> loadedPlugins;
std::vector<std::unique_ptr<CompressorPolicy>> ownedPolicies;
std::array<CompressorPolicy*, kOperationCount> defaultPolicies = {};
std::array<CompressorPolicy*, kOperationCount> hierarchicalPolicies = {};

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

ncclResult_t cudaResult(cudaError_t result) {
  return result == cudaSuccess ? ncclSuccess : ncclUnhandledCudaError;
}

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
    INFO(NCCL_INIT,
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

ncclResult_t createPolicy(const cocclCompressorPolicyEntry& configured,
                          cocclCompressorConfigVariant variant,
                          CompressorPolicy** policy) {
  auto plugin = loadedPlugins.find(configured.name);
  if (plugin == loadedPlugins.end()) return ncclInvalidArgument;

  const cocclConfigValues& values =
      variant == cocclCompressorConfigHierarchical
          ? configured.hierarchicalValues
          : configured.defaultValues;
  ConfigViewStorage storage(values);
  const cocclConfigView view = storage.view();
  const cocclCompressorConfigContext context = {
      variant, runtimeNodes, runtimeDevicesPerNode};
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

ncclResult_t installPolicy(cocclOperation operation,
                           const cocclPrimitivePolicy& configured,
                           bool hierarchical) {
  if (configured.compressor.name.empty()) return ncclSuccess;
  const size_t index = static_cast<size_t>(operation);
  NCCLCHECK(createPolicy(configured.compressor, cocclCompressorConfigDefault,
                         &defaultPolicies[index]));
  defaultPolicies[index]->thresholdBytes = configured.thresholdBytes;
  hierarchicalPolicies[index] = defaultPolicies[index];
  if (hierarchical && configured.compressor.hasHierarchicalConfig) {
    NCCLCHECK(createPolicy(configured.compressor,
                           cocclCompressorConfigHierarchical,
                           &hierarchicalPolicies[index]));
    hierarchicalPolicies[index]->thresholdBytes = configured.thresholdBytes;
  }
  return ncclSuccess;
}

ncclResult_t initializeRuntime(const ncclComm_t comm,
                               const cocclConfig& config) {
  runtimeRanks = comm->nRanks;
  runtimeNodes = comm->nNodes;
  runtimeDevicesPerNode = comm->localRanks;
  NCCLCHECK(loadPlugins(config));
  if (config.runtime.mode != cocclRuntimeMode::Normal) return ncclSuccess;

  NCCLCHECK(installPolicy(cocclOperation::AllToAll,
                          config.normal.allToAll, false));
  NCCLCHECK(installPolicy(cocclOperation::AllGather,
                          config.normal.allGather, false));
  NCCLCHECK(installPolicy(cocclOperation::AllReduce,
                          config.normal.allReduce, true));
  NCCLCHECK(installPolicy(cocclOperation::ReduceScatter,
                          config.normal.reduceScatter, true));
  NCCLCHECK(installPolicy(cocclOperation::SendRecv,
                          config.normal.sendRecv, false));

  CompEnableThreshold = config.runtime.compressionThresholdBytes;
  enableAllToAllComp = defaultPolicies[static_cast<size_t>(
      cocclOperation::AllToAll)] != nullptr;
  enableAllGatherComp = defaultPolicies[static_cast<size_t>(
      cocclOperation::AllGather)] != nullptr;
  enableAllReduceComp = defaultPolicies[static_cast<size_t>(
      cocclOperation::AllReduce)] != nullptr;
  enableReduceScatterComp = defaultPolicies[static_cast<size_t>(
      cocclOperation::ReduceScatter)] != nullptr;
  enableSendRecvComp = defaultPolicies[static_cast<size_t>(
      cocclOperation::SendRecv)] != nullptr;
  return ncclSuccess;
}

CompressorPolicy* policyFor(ncclCommOp_t operation) {
  cocclOperation mapped = cocclOperation::Count;
  bool hierarchical = false;
  switch (operation) {
    case AlltoAll: mapped = cocclOperation::AllToAll; break;
    case AlltoAll_Inter:
      mapped = cocclOperation::AllToAll;
      hierarchical = true;
      break;
    case AllReduce: mapped = cocclOperation::AllReduce; break;
    case AllReduce_Inter:
      mapped = cocclOperation::AllReduce;
      hierarchical = true;
      break;
    case AllGather: mapped = cocclOperation::AllGather; break;
    case AllGather_Inter:
      mapped = cocclOperation::AllGather;
      hierarchical = true;
      break;
    case ReduceScatter: mapped = cocclOperation::ReduceScatter; break;
    case ReduceScatter_Inter:
      mapped = cocclOperation::ReduceScatter;
      hierarchical = true;
      break;
    case SendRecv:
    case SendRecv_BWD: mapped = cocclOperation::SendRecv; break;
  }
  if (mapped == cocclOperation::Count) return nullptr;
  const size_t index = static_cast<size_t>(mapped);
  return hierarchical ? hierarchicalPolicies[index] : defaultPolicies[index];
}

int rankForDevice(int cudaDev) {
  pthread_mutex_lock(&compressorLock);
  auto rank = rankByDevice.find(cudaDev);
  const int value = rank == rankByDevice.end() ? 0 : rank->second;
  pthread_mutex_unlock(&compressorLock);
  return value;
}

ncclResult_t execute(CompressorPolicy* policy, cocclCompressorCall* call,
                     int rank, cudaStream_t stream) {
  int cudaDev = 0;
  CUDACHECK(cudaGetDevice(&cudaDev));
  ExecutionResources resources = {policy, cudaDev, stream};
  cocclCompressorExecutionContext execution = {
      sizeof(cocclCompressorExecutionContext), &kHostApi, &resources,
      stream, cudaDev, rank, runtimeRanks, runtimeNodes,
      runtimeDevicesPerNode};
  call->config = policy->config;
  call->execution = &execution;
  ncclResult_t result = policy->plugin->execute(call);
  if (resources.scratchBytes != 0) {
    std::lock_guard<std::mutex> guard(policy->resourceLock);
    DeviceResources& device = policy->resources[cudaDev];
    if (resources.scratchBytes > device.scratchPeakBytes) {
      device.scratchPeakBytes = resources.scratchBytes;
      INFO(NCCL_INIT, "COCCL compressor %s scratch device %d peak %zu",
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

ncclResult_t allocateEncodedOutput(void** output, size_t capacity,
                                   cudaStream_t stream, bool* allocated) {
  *allocated = false;
  if (*output != nullptr) return ncclSuccess;
  CUDACHECK(cudaMallocAsync(output, capacity, stream));
  *allocated = true;
  return ncclSuccess;
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

ncclResult_t executeCompress(CompressorPolicy* policy, const void* input,
                             void** output, size_t chunkElements,
                             ncclDataType_t datatype, size_t chunks, int rank,
                             size_t* encodedChunkElements,
                             ncclDataType_t* encodedDatatype,
                             cudaStream_t stream) {
  const size_t elements = chunkElements * chunks;
  const size_t bytes = elements * ncclTypeSize(datatype);
  bool allocated = false;
  NCCLCHECK(allocateEncodedOutput(output, bytes, stream, &allocated));

  const cocclCompressorView inputView = {
      const_cast<void*>(input), bytes, bytes, elements, chunks, datatype,
      nullptr, 0};
  cocclCompressorView outputView = {
      *output, bytes, 0, 0, chunks, ncclInt8, nullptr, 0};
  cocclCompressorCall call = {
      sizeof(cocclCompressorCall), cocclCompressorOperationCompress,
      inputView, &outputView, rank, 0, datatype, elements, nullptr, nullptr};
  ncclResult_t result = execute(policy, &call, rank, stream);
  if (result == ncclSuccess) {
    result = validateEncodedOutput(outputView, chunks);
  }
  if (result != ncclSuccess) {
    if (allocated) {
      (void)cudaFreeAsync(*output, stream);
      *output = nullptr;
    }
    return result;
  }
  *encodedChunkElements = outputView.elements / outputView.chunks;
  *encodedDatatype = outputView.datatype;
  return ncclSuccess;
}

ncclResult_t executeDecompress(CompressorPolicy* policy, void* output,
                               const void* input, size_t outputChunkElements,
                               ncclDataType_t outputDatatype,
                               size_t inputChunkElements,
                               ncclDataType_t inputDatatype, size_t chunks,
                               cudaStream_t stream) {
  const size_t outputElements = outputChunkElements * chunks;
  const size_t outputBytes = outputElements * ncclTypeSize(outputDatatype);
  const size_t inputElements = inputChunkElements * chunks;
  const size_t inputBytes = inputElements * ncclTypeSize(inputDatatype);
  if (inputDatatype == COCCL_COMPRESSOR_RAW_PASSTHROUGH) {
    CUDACHECK(cudaMemcpyAsync(output, input, outputBytes,
                              cudaMemcpyDeviceToDevice, stream));
    return ncclSuccess;
  }

  const cocclCompressorView inputView = {
      const_cast<void*>(input), inputBytes, inputBytes, inputElements, chunks,
      inputDatatype, nullptr, 0};
  cocclCompressorView outputView = {
      output, outputBytes, 0, outputElements, chunks, outputDatatype,
      nullptr, 0};
  int cudaDev = 0;
  CUDACHECK(cudaGetDevice(&cudaDev));
  const int rank = rankForDevice(cudaDev);
  cocclCompressorCall call = {
      sizeof(cocclCompressorCall), cocclCompressorOperationDecompress,
      inputView, &outputView, rank, 0, outputDatatype, outputElements,
      nullptr, nullptr};
  return execute(policy, &call, rank, stream);
}

ncclResult_t genericDecompressReduce(
    CompressorPolicy* policy, void* output, const void* input,
    size_t inputChunkElements, ncclDataType_t inputDatatype,
    size_t outputElements, ncclDataType_t outputDatatype,
    size_t inputChunks, size_t reduceChunks, cudaStream_t stream,
    ncclComm_t ownerComm) {
  const size_t outputChunks = inputChunks / reduceChunks;
  const size_t rawChunkElements = outputElements / outputChunks;
  const size_t bytes = rawChunkElements * inputChunks *
      ncclTypeSize(outputDatatype);
  cocclBufferHandle temporary;
  ncclResult_t ret = cocclGetUnregisteredBuffer(ownerComm, bytes, stream,
                                                 &temporary);
  if (ret == ncclSuccess) {
    ret = executeDecompress(policy, temporary.ptr, input,
                            rawChunkElements, outputDatatype,
                            inputChunkElements, inputDatatype, inputChunks,
                            stream);
  }
  if (ret == ncclSuccess) {
    ret = ncclReduceChunk(temporary.ptr, outputElements, output,
                          outputDatatype, reduceChunks, stream);
  }
  ncclResult_t releaseRet = cocclReleaseBuffer(&temporary, stream);
  return ret == ncclSuccess ? releaseRet : ret;
}

}  // namespace

bool cocclCompressionEnabled() {
  return runtimeInitialized && runtimeInitResult == ncclSuccess &&
      (enableAllToAllComp || enableAllReduceComp || enableAllGatherComp ||
       enableReduceScatterComp || enableSendRecvComp);
}

ncclResult_t cocclResolveCompressorPolicy(
    cocclPolicyKey key, cocclResolvedCompressorPolicy* resolved) {
  const size_t index = static_cast<size_t>(key.operation);
  if (index >= kOperationCount) return ncclInvalidUsage;

  CompressorPolicy* policy = nullptr;
  if (key.variant == cocclPolicyVariant::Default) {
    policy = defaultPolicies[index];
  } else if (key.variant == cocclPolicyVariant::Hierarchical) {
    policy = hierarchicalPolicies[index];
  }
  if (policy == nullptr) return ncclInvalidUsage;
  resolved->compressor = policy;
  resolved->thresholdBytes = policy->thresholdBytes;
  return ncclSuccess;
}

ncclResult_t ncclCompressInit(const ncclComm_t comm) {
  const bool configReady = cocclConfigInitialize();
  pthread_mutex_lock(&compressorLock);
  if (!runtimeInitialized) {
    runtimeInitialized = true;
    runtimeInitResult = configReady
        ? initializeRuntime(comm, cocclGetConfig()) : ncclSuccess;
  }
  const ncclResult_t result = runtimeInitResult;
  if (result == ncclSuccess) {
    rankByDevice[comm->cudaDev] = comm->rank;
    ++communicatorsByDevice[comm->cudaDev];
  }
  pthread_mutex_unlock(&compressorLock);
  return result;
}

ncclResult_t ncclCompressDestroy(const ncclComm_t comm) {
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
    INFO(NCCL_INIT,
         "COCCL compressor %s release device %d persistent %zu scratch_peak %zu states %zu",
         policy->plugin->name, comm->cudaDev, persistentBytes,
         resources->second.scratchPeakBytes, resources->second.states.size());
    policy->resources.erase(resources);
  }
  pthread_mutex_unlock(&compressorLock);
  return result;
}

ncclResult_t ncclCompress(
    const void* orgbuff, void** compbuff, const size_t orgChunkCount,
    ncclDataType_t orgDatatype, size_t* compChunkCount,
    ncclDataType_t* compDatatype, const size_t numChunks, const int rank,
    ncclCommOp_t commOp, cudaStream_t stream) {
  CompressorPolicy* policy = policyFor(commOp);
  if (policy == nullptr) return ncclInvalidUsage;
  return executeCompress(policy, orgbuff, compbuff, orgChunkCount,
                         orgDatatype, numChunks, rank, compChunkCount,
                         compDatatype, stream);
}

ncclResult_t ncclDecompress(
    void* decompbuff, const void* compbuff, const size_t decompChunkCount,
    ncclDataType_t decompDatatype, const size_t compChunkCount,
    ncclDataType_t compDatatype, const size_t numChunks,
    ncclCommOp_t commOp, cudaStream_t stream) {
  CompressorPolicy* policy = policyFor(commOp);
  if (policy == nullptr) return ncclInvalidUsage;
  return executeDecompress(policy, decompbuff, compbuff, decompChunkCount,
                           decompDatatype, compChunkCount, compDatatype,
                           numChunks, stream);
}

ncclResult_t ncclDecompressReduce(
    void* reducebuff, const void* compbuff, const size_t compChunkCount,
    ncclDataType_t compDatatype, const size_t outputElements,
    ncclDataType_t reduceDatatype, const size_t inputChunks,
    const size_t reduceChunks,
    ncclCommOp_t commOp, cudaStream_t stream, ncclComm_t ownerComm) {
  CompressorPolicy* policy = policyFor(commOp);
  if (policy == nullptr) return ncclInvalidUsage;
  if (reduceChunks == 0 || inputChunks % reduceChunks != 0) {
    return ncclInvalidArgument;
  }
  const size_t outputChunks = inputChunks / reduceChunks;
  if (outputElements % outputChunks != 0) return ncclInvalidArgument;
  if (compDatatype == COCCL_COMPRESSOR_RAW_PASSTHROUGH ||
      (policy->plugin->capabilities &
       cocclCompressorCapabilityDecompressReduce) == 0) {
    return genericDecompressReduce(
        policy, reducebuff, compbuff, compChunkCount, compDatatype,
        outputElements, reduceDatatype, inputChunks, reduceChunks, stream,
        ownerComm);
  }

  const size_t inputElements = compChunkCount * inputChunks;
  const size_t inputBytes = inputElements * ncclTypeSize(compDatatype);
  const size_t outputBytes = outputElements * ncclTypeSize(reduceDatatype);
  const cocclCompressorView input = {
      const_cast<void*>(compbuff), inputBytes, inputBytes, inputElements,
      inputChunks, compDatatype, nullptr, 0};
  cocclCompressorView output = {
      reducebuff, outputBytes, 0, outputElements, outputChunks, reduceDatatype,
      nullptr, 0};
  int cudaDev = 0;
  CUDACHECK(cudaGetDevice(&cudaDev));
  const int rank = rankForDevice(cudaDev);
  cocclCompressorCall call = {
      sizeof(cocclCompressorCall),
      cocclCompressorOperationDecompressReduce,
      input,
      &output,
      rank,
      reduceChunks,
      reduceDatatype,
      outputElements,
      nullptr,
      nullptr};
  return execute(policy, &call, rank, stream);
}

ncclResult_t ncclDecompReduceComp(
    const void* compbuff, void** recompbuff, const size_t originalElements,
    ncclDataType_t orgDatatype, const size_t compChunkCount,
    ncclDataType_t compDatatype, size_t* reCompChunkCount,
    ncclDataType_t* reCompDatatype, const size_t inputChunks,
    const size_t reduceChunks,
    ncclCommOp_t commOp, cudaStream_t stream, ncclComm_t ownerComm) {
  CompressorPolicy* policy = policyFor(commOp);
  if (policy == nullptr) return ncclInvalidUsage;
  if (reduceChunks == 0 || inputChunks % reduceChunks != 0) {
    return ncclInvalidArgument;
  }
  const size_t outputChunks = inputChunks / reduceChunks;
  if (originalElements % outputChunks != 0) return ncclInvalidArgument;
  const size_t rawChunkElements = originalElements / outputChunks;
  if (outputChunks != 1 ||
      compDatatype == COCCL_COMPRESSOR_RAW_PASSTHROUGH ||
      (policy->plugin->capabilities &
       cocclCompressorCapabilityDecompressReduceCompress) == 0) {
    const size_t bytes = rawChunkElements * inputChunks *
        ncclTypeSize(orgDatatype);
    cocclBufferHandle temporary;
    ncclResult_t ret = cocclGetUnregisteredBuffer(ownerComm, bytes, stream,
                                                   &temporary);
    if (ret == ncclSuccess) {
      ret = executeDecompress(policy, temporary.ptr, compbuff,
                              rawChunkElements, orgDatatype, compChunkCount,
                              compDatatype, inputChunks, stream);
    }
    if (ret == ncclSuccess) {
      ret = ncclReduceChunk(temporary.ptr, originalElements, temporary.ptr,
                            orgDatatype, reduceChunks, stream);
    }
    if (ret == ncclSuccess) {
      ret = executeCompress(policy, temporary.ptr, recompbuff,
                            rawChunkElements, orgDatatype, outputChunks, 0,
                            reCompChunkCount, reCompDatatype, stream);
    }
    ncclResult_t releaseRet = cocclReleaseBuffer(&temporary, stream);
    return ret == ncclSuccess ? releaseRet : ret;
  }

  const size_t inputElements = compChunkCount * inputChunks;
  const size_t inputBytes = inputElements * ncclTypeSize(compDatatype);
  const size_t rawOutputBytes = originalElements * ncclTypeSize(orgDatatype);
  bool allocated = false;
  NCCLCHECK(allocateEncodedOutput(recompbuff, rawOutputBytes, stream,
                                 &allocated));
  const cocclCompressorView input = {
      const_cast<void*>(compbuff), inputBytes, inputBytes, inputElements,
      inputChunks, compDatatype, nullptr, 0};
  cocclCompressorView output = {
      *recompbuff, rawOutputBytes, 0, 0, outputChunks, ncclInt8, nullptr, 0};
  int cudaDev = 0;
  CUDACHECK(cudaGetDevice(&cudaDev));
  const int rank = rankForDevice(cudaDev);
  cocclCompressorCall call = {
      sizeof(cocclCompressorCall),
      cocclCompressorOperationDecompressReduceCompress,
      input,
      &output,
      rank,
      reduceChunks,
      orgDatatype,
      originalElements,
      nullptr,
      nullptr};
  ncclResult_t result = execute(policy, &call, rank, stream);
  if (result == ncclSuccess) {
    result = validateEncodedOutput(output, outputChunks);
  }
  if (result != ncclSuccess) {
    if (allocated) {
      (void)cudaFreeAsync(*recompbuff, stream);
      *recompbuff = nullptr;
    }
    return result;
  }
  *reCompChunkCount = output.elements / output.chunks;
  *reCompDatatype = output.datatype;
  return ncclSuccess;
}
