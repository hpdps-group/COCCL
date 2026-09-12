#ifndef COCCL_COMPRESSOR_INTERNAL_H_
#define COCCL_COMPRESSOR_INTERNAL_H_

#include "core/compression/coccl_compressor_runtime.h"
#include "core/config/coccl_config.h"

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

namespace cocclCompressorInternal {

struct PersistentBuffer {
  void* data = nullptr;
  size_t bytes = 0;
};

struct StateEntry {
  void* data = nullptr;
  cocclCompressorDestroyStateFn destroy = nullptr;
};

struct StatefulResources {
  std::map<size_t, PersistentBuffer> persistent;
  std::map<const void*, StateEntry> states;
  std::vector<cudaEvent_t> historyReady;
  size_t activeSlices = 0;
};

using StateScopeKey = std::tuple<ncclComm_t, int>;

struct DeviceResources {
  std::map<StateScopeKey, StatefulResources> scopes;
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

constexpr size_t kOperationCount =
    static_cast<size_t>(cocclOperation::Count);
constexpr size_t kPolicyVariantCount = 3;
constexpr size_t kTrainingRoleCount =
    static_cast<size_t>(cocclTrainingRoleCount);
constexpr size_t kCompressionScopeCount =
    static_cast<size_t>(cocclCompressionScope::Count);

// The same policy objects are shared by inherited default/intra/inter bindings.
struct Registry {
  bool hasPolicies = false;
  std::map<std::string, LoadedPlugin> loadedPlugins;
  std::vector<std::unique_ptr<CompressorPolicy>> ownedPolicies;
  CompressorPolicy* policies[kTrainingRoleCount][kPolicyVariantCount]
                            [kOperationCount][kCompressionScopeCount] = {};
};

ncclResult_t initializeRegistry(
    Registry* registry, const cocclConfig& config,
    const cocclCompressorConfigContext& context);

}  // namespace cocclCompressorInternal

#endif
