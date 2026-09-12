#include "core/compression/coccl_compressor_runtime.h"
#include "core/compression/coccl_compressor_internal.h"

#include "core/config/coccl_config.h"
#include "core/training/coccl_training_assist.h"
#include "comm.h"
#include "compressor_plugin/detail/coccl_compressor_abi.h"
#include "debug.h"

#include <cuda_runtime.h>
#include <pthread.h>

#include <array>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

using namespace cocclCompressorInternal;

namespace {

struct ExecutionResources {
  CompressorPolicy* policy = nullptr;
  int cudaDev = 0;
  cudaStream_t stream = nullptr;
  std::vector<void*> scratch;
  size_t scratchBytes = 0;
  StateScopeKey stateScope = {};
  StatefulResources* history = nullptr;
  size_t historySlot = 0;
  size_t historySlots = 0;
  cocclCompressorOperation operation = cocclCompressorOperationCompress;
};

pthread_mutex_t compressorLock = PTHREAD_MUTEX_INITIALIZER;
bool runtimeInitialized = false;
ncclResult_t runtimeInitResult = ncclSuccess;
int runtimeRanks = 1;
int runtimeNodes = 1;
int runtimeDevicesPerNode = 1;
std::map<int, int> rankByDevice;
std::map<int, size_t> communicatorsByDevice;
Registry registry;

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
      policy->resources[execution->cudaDev].scopes[execution->stateScope]
          .persistent[slot];
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
  StatefulResources& scope = policy->resources[execution->cudaDev]
      .scopes[execution->stateScope];
  if (execution->historySlots != 0) {
    execution->history = &scope;
    // Different ranks need not use the same sequence of user streams.
    // Only serialize a history slot against its preceding decoder write.
    if (scope.activeSlices != execution->historySlots) {
      // Repartitioning reuses the same history allocations after old writes.
      for (cudaEvent_t ready : scope.historyReady) {
        if (ready != nullptr) {
          CUDACHECK(cudaStreamWaitEvent(execution->stream, ready, 0));
        }
      }
      scope.activeSlices = execution->historySlots;
      if (scope.historyReady.size() < scope.activeSlices) {
        scope.historyReady.resize(scope.activeSlices, nullptr);
      }
    } else {
      cudaEvent_t ready = scope.historyReady[execution->historySlot];
      if (execution->operation == cocclCompressorOperationCompress && ready) {
        CUDACHECK(cudaStreamWaitEvent(execution->stream, ready, 0));
      }
    }
  }
  StateEntry& entry = scope.states[typeKey];
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

ncclResult_t initializeRuntime(const ncclComm_t comm,
                               const cocclConfig& config) {
  runtimeRanks = comm->nRanks;
  runtimeNodes = comm->nNodes;
  runtimeDevicesPerNode = comm->localRanks;
  const cocclCompressorConfigContext context = {
      cocclCompressorConfigDefault, runtimeNodes, runtimeDevicesPerNode};
  return initializeRegistry(&registry, config, context);
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
                     cudaStream_t stream, const cocclCompressorScope* scope) {
  int cudaDev = 0;
  CUDACHECK(cudaGetDevice(&cudaDev));
  ExecutionResources resources = {policy, cudaDev, stream};
  cocclCompressorExecutionContext execution = {
      COCCL_COMPRESSOR_EXECUTION_BASE_SIZE, &kHostApi, &resources,
      stream, cudaDev, rank, runtimeRanks, runtimeNodes,
      runtimeDevicesPerNode};
  if (policy->plugin->capabilities & cocclCompressorCapabilityPipelineState) {
    execution.structSize = sizeof(execution);
    if (scope != nullptr && scope->slices != 0) {
      resources.stateScope = {scope->comm, scope->layout};
      resources.historySlot = scope->slice;
      resources.historySlots = scope->slices;
      resources.operation = call->operation;
      execution.pipelineSlice = scope->slice;
      execution.pipelineSlices = scope->slices;
      execution.localChunkIndex = scope->localChunkIndex;
    }
  }
  call->config = policy->config;
  call->execution = &execution;
  call->inputConfig = inputPolicy->config;
  ncclResult_t result = policy->plugin->execute(call);
  if (result == ncclSuccess && resources.history != nullptr &&
      call->operation == cocclCompressorOperationDecompress) {
    cudaEvent_t& ready = resources.history->historyReady[resources.historySlot];
    cudaError_t status = cudaSuccess;
    if (ready == nullptr) {
      status = cudaEventCreateWithFlags(&ready, cudaEventDisableTiming);
    }
    if (status == cudaSuccess) status = cudaEventRecord(ready, stream);
    if (status != cudaSuccess) result = ncclUnhandledCudaError;
  }
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
      registry.hasPolicies;
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

  CompressorPolicy* policy = registry.policies[role][variant][index][scope];
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
    size_t originalElements, cudaStream_t stream,
    const cocclCompressorScope* scope) {
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
  NCCLCHECK(execute(policy, inputPolicy, &call, rank, stream, scope));
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
  const bool lastDeviceComm = --count->second == 0;
  if (lastDeviceComm) {
    communicatorsByDevice.erase(count);
    rankByDevice.erase(comm->cudaDev);
  }
  ncclResult_t result = ncclSuccess;
  for (const std::unique_ptr<CompressorPolicy>& policy : registry.ownedPolicies) {
    std::lock_guard<std::mutex> guard(policy->resourceLock);
    auto resources = policy->resources.find(comm->cudaDev);
    if (resources == policy->resources.end()) continue;
    size_t persistentBytes = 0;
    size_t stateCount = 0;
    auto& scopes = resources->second.scopes;
    for (auto scope = scopes.begin(); scope != scopes.end();) {
      if (!lastDeviceComm && std::get<0>(scope->first) != comm) {
        ++scope;
        continue;
      }
      stateCount += scope->second.states.size();
      for (cudaEvent_t ready : scope->second.historyReady) {
        if (ready != nullptr && cudaEventDestroy(ready) != cudaSuccess &&
            result == ncclSuccess) {
          result = ncclUnhandledCudaError;
        }
      }
      for (const auto& state : scope->second.states) {
        state.second.destroy(state.second.data);
      }
      for (const auto& persistent : scope->second.persistent) {
        persistentBytes += persistent.second.bytes;
        if (cudaFree(persistent.second.data) != cudaSuccess &&
            result == ncclSuccess) {
          result = ncclUnhandledCudaError;
        }
      }
      scope = scopes.erase(scope);
    }
    INFO(COCCL_COMPRESS,
         "COCCL compressor %s release device %d persistent %zu scratch_peak %zu states %zu",
         policy->plugin->name, comm->cudaDev, persistentBytes,
         resources->second.scratchPeakBytes, stateCount);
    if (lastDeviceComm) policy->resources.erase(resources);
  }
  pthread_mutex_unlock(&compressorLock);
  return result;
}
