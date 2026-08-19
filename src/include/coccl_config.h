#ifndef COCCL_CONFIG_H_
#define COCCL_CONFIG_H_

#include <stddef.h>
#include <stdint.h>

#include <array>
#include <map>
#include <string>
#include <vector>

#include "coccl_operation.h"

enum class cocclRuntimeMode {
  Normal,
  Training,
};

enum class cocclReduceScatterAlgorithmPolicy {
  Auto,
  OneShot,
  TwoShot,
};

enum class cocclAllReduceAlgorithmPolicy {
  Auto,
  OneShot,
  TwoShot,
  TripleShot,
};

using cocclConfigValues = std::map<std::string, std::string>;

struct cocclCompressorScopeEntry {
  bool configured = false;
  bool enabled = false;
  std::string name;
  cocclConfigValues values;
};

struct cocclPrimitivePolicy {
  std::array<cocclCompressorScopeEntry,
             static_cast<size_t>(cocclCompressionScope::Count)> scopes;
  size_t thresholdBytes = 8ULL * 1024 * 1024;
};

struct cocclEffectiveCompressorScope {
  const cocclCompressorScopeEntry* entry = nullptr;
  cocclCompressionScope source = cocclCompressionScope::Default;

  bool enabled() const { return entry != nullptr && entry->enabled; }
};

const cocclCompressorScopeEntry& cocclConfiguredCompressorScope(
    const cocclPrimitivePolicy& policy, cocclCompressionScope scope);
cocclEffectiveCompressorScope cocclEffectiveCompressorScopeFor(
    const cocclPrimitivePolicy& policy, cocclCompressionScope scope);

struct cocclCollectivePolicies {
  cocclPrimitivePolicy allGather;
  cocclPrimitivePolicy reduceScatter;
  cocclPrimitivePolicy allReduce;
};

struct cocclNormalPolicies : cocclCollectivePolicies {
  cocclPrimitivePolicy allToAll;
  // Normal mode deliberately uses one policy in both P2P directions.
  cocclPrimitivePolicy sendRecv;
};

struct cocclTrainingPolicies {
  cocclCollectivePolicies dataParallel;
  cocclCollectivePolicies tensorParallel;
  cocclPrimitivePolicy pipelineSendRecvForward;
  cocclPrimitivePolicy pipelineSendRecvBackward;
};

struct cocclRuntimeConfig {
  cocclRuntimeMode mode = cocclRuntimeMode::Normal;
  size_t compressionThresholdBytes = 8ULL * 1024 * 1024;
};

struct cocclPluginCatalog {
  std::vector<std::string> compressors;
  std::string libraryPath;
};

struct cocclPipelineConfig {
  int depth = 1;
};

struct cocclBufferConfig {
  size_t poolLimitBytes = 0;
  size_t legacyBlockBytes = 0;
  size_t physicalChunkBytes = 8ULL * 1024 * 1024;
};

struct cocclAutotuneConfig {
  bool enabled = true;
  size_t profileMinBytes = 256ULL * 1024;
  size_t profileMaxBytes = 8ULL * 1024 * 1024 * 1024;
  int warmup = 3;
  int iterations = 10;
  cocclReduceScatterAlgorithmPolicy reduceScatterAlgorithm =
      cocclReduceScatterAlgorithmPolicy::Auto;
  cocclAllReduceAlgorithmPolicy allReduceAlgorithm =
      cocclAllReduceAlgorithmPolicy::Auto;
};

struct cocclTrainingConfig {
  int observationIterations = 10;
  size_t maxEvents = 65536;
};

struct cocclConfig {
  cocclRuntimeConfig runtime;
  cocclPluginCatalog plugins;
  cocclPipelineConfig pipeline;
  cocclBufferConfig buffer;
  cocclAutotuneConfig autotune;
  cocclTrainingConfig training;
  cocclNormalPolicies normal;
  cocclTrainingPolicies trainingPolicies;
};

constexpr int kCocclConfigSchemaVersion = 3;
constexpr int kCocclMinPipelineDepth = 1;
constexpr int kCocclMaxPipelineDepth = 16;

enum class cocclConfigPolicyId {
  NormalAllGather,
  NormalReduceScatter,
  NormalAllReduce,
  NormalAllToAll,
  NormalSendRecv,
  TrainingDpAllGather,
  TrainingDpReduceScatter,
  TrainingDpAllReduce,
  TrainingTpAllGather,
  TrainingTpReduceScatter,
  TrainingTpAllReduce,
  TrainingPpSendRecvForward,
  TrainingPpSendRecvBackward,
  Count,
};

enum class cocclPolicyScope {
  Normal,
  DataParallel,
  TensorParallel,
  PipelineParallel,
};

constexpr size_t kCocclConfigPolicyCount =
    static_cast<size_t>(cocclConfigPolicyId::Count);

struct cocclConfigPolicyView {
  cocclConfigPolicyId id;
  const char* path;
  cocclRuntimeMode mode;
  cocclPolicyScope scope;
  cocclPolicyKey key;
  const cocclPrimitivePolicy* policy;
};

using cocclConfigPolicyList =
    std::array<cocclConfigPolicyView, kCocclConfigPolicyCount>;

// Pure configuration helpers shared by the runtime and the standalone
// checker. They do not read environment variables or emit NCCL logs.
bool cocclLoadConfigFile(const std::string& path, cocclConfig* config,
                         std::string* error);
cocclConfigPolicyList cocclEnumeratePolicies(const cocclConfig& config);

// These are the only functions allowed to read COCCL_ENABLE and
// COCCL_CONFIG_FILE. Initialization is process-local, thread-safe, and runs at
// most once. Invalid configuration permanently disables COCCL for the process.
bool cocclConfigInitialize();
// Requires a successful cocclConfigInitialize(). The returned process config
// is immutable and remains valid until process exit.
const cocclConfig& cocclGetConfig();

#endif
