#include "coccl_config.h"

#ifndef COCCL_CONFIG_PARSER_ONLY
#include "coccl_config_debug.h"
#include "debug.h"
#endif
#include "third_party/tomlplusplus/toml.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <limits>
#include <optional>
#include <set>
#include <sstream>
#include <string_view>
#include <utility>

#ifndef COCCL_CONFIG_PARSER_ONLY
#include <atomic>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <mutex>
#endif

namespace {

bool fail(std::string* error, const std::string& message) {
  if (error != nullptr) *error = message;
  return false;
}

bool allowedKey(std::string_view key,
                std::initializer_list<std::string_view> allowed) {
  return std::find(allowed.begin(), allowed.end(), key) != allowed.end();
}

bool validateKeys(const toml::table& table,
                  std::initializer_list<std::string_view> allowed,
                  const std::string& path, std::string* error) {
  for (const auto& [key, value] : table) {
    (void)value;
    if (!allowedKey(key.str(), allowed)) {
      return fail(error, path + ": unknown key '" + std::string(key.str()) + "'");
    }
  }
  return true;
}

const toml::table* optionalTable(const toml::table& parent, const char* key,
                                 const std::string& path,
                                 std::string* error) {
  const toml::node* node = parent.get(key);
  if (node == nullptr) return nullptr;
  const toml::table* table = node->as_table();
  if (table == nullptr) {
    fail(error, path + "." + key + " must be a table");
  }
  return table;
}

bool readString(const toml::table& table, const char* key, std::string* value,
                bool required, const std::string& path, std::string* error) {
  const toml::node* node = table.get(key);
  if (node == nullptr) {
    return required ? fail(error, path + "." + key + " is required") : true;
  }
  auto parsed = node->value<std::string>();
  if (!parsed) return fail(error, path + "." + key + " must be a string");
  *value = std::move(*parsed);
  return true;
}

bool readBool(const toml::table& table, const char* key, bool* value,
              const std::string& path, std::string* error) {
  const toml::node* node = table.get(key);
  if (node == nullptr) return true;
  auto parsed = node->value<bool>();
  if (!parsed) return fail(error, path + "." + key + " must be a boolean");
  *value = *parsed;
  return true;
}

bool readSize(const toml::table& table, const char* key, size_t* value,
              const std::string& path, std::string* error) {
  const toml::node* node = table.get(key);
  if (node == nullptr) return true;
  auto parsed = node->value<int64_t>();
  if (!parsed || *parsed < 0 ||
      (uint64_t)*parsed > std::numeric_limits<size_t>::max()) {
    return fail(error, path + "." + key + " must be a non-negative integer");
  }
  *value = (size_t)*parsed;
  return true;
}

bool readInt(const toml::table& table, const char* key, int* value, int min,
             int max, const std::string& path, std::string* error) {
  const toml::node* node = table.get(key);
  if (node == nullptr) return true;
  auto parsed = node->value<int64_t>();
  if (!parsed || *parsed < min || *parsed > max) {
    std::ostringstream message;
    message << path << "." << key << " must be an integer in [" << min
            << ", " << max << "]";
    return fail(error, message.str());
  }
  *value = (int)*parsed;
  return true;
}

bool readStringArray(const toml::table& table, const char* key,
                     std::vector<std::string>* values,
                     const std::string& path, std::string* error) {
  const toml::node* node = table.get(key);
  if (node == nullptr) return true;
  const toml::array* array = node->as_array();
  if (array == nullptr) {
    return fail(error, path + "." + key + " must be an array of strings");
  }

  std::set<std::string> unique;
  values->clear();
  for (const toml::node& entry : *array) {
    auto name = entry.value<std::string>();
    if (!name || name->empty()) {
      return fail(error, path + "." + key + " contains a non-string or empty name");
    }
    if (!std::all_of(name->begin(), name->end(), [](unsigned char c) {
          return std::isalnum(c) || c == '_' || c == '-';
        })) {
      return fail(error, path + "." + key + " contains invalid compressor name '" +
                             *name + "'");
    }
    if (!unique.insert(*name).second) {
      return fail(error, path + "." + key + " contains duplicate compressor '" +
                             *name + "'");
    }
    values->push_back(std::move(*name));
  }
  return true;
}

bool validCompressorName(const std::string& name) {
  return !name.empty() &&
      std::all_of(name.begin(), name.end(), [](unsigned char c) {
        return std::isalnum(c) || c == '_' || c == '-';
      });
}

bool scalarToString(const toml::node& node, std::string* value) {
  // Check the TOML node kind before requesting a value. toml++ permits some
  // numeric conversions, which would otherwise turn false into "0" instead
  // of the strict boolean spelling expected by the compressor SDK reader.
  if (node.is_string()) {
    auto stringValue = node.value<std::string>();
    if (!stringValue) return false;
    *value = std::move(*stringValue);
    return true;
  }
  if (node.is_boolean()) {
    auto boolValue = node.value<bool>();
    if (!boolValue) return false;
    *value = *boolValue ? "true" : "false";
    return true;
  }
  if (node.is_integer()) {
    auto integerValue = node.value<int64_t>();
    if (!integerValue) return false;
    *value = std::to_string(*integerValue);
    return true;
  }
  if (node.is_floating_point()) {
    auto floatValue = node.value<double>();
    if (!floatValue) return false;
    if (!std::isfinite(*floatValue)) return false;
    std::ostringstream stream;
    stream << std::setprecision(std::numeric_limits<double>::max_digits10)
           << *floatValue;
    *value = stream.str();
    return true;
  }
  return false;
}

bool parseOptionValues(const toml::table& table, cocclConfigValues* values,
                       const std::string& path, std::string* error) {
  values->clear();
  for (const auto& [key, node] : table) {
    std::string value;
    if (!scalarToString(node, &value)) {
      return fail(error, path + "." + std::string(key.str()) +
                             " must be a string, integer, float, or boolean");
    }
    values->emplace(std::string(key.str()), std::move(value));
  }
  return true;
}

bool parseCompressorScope(const toml::table* table,
                          cocclCompressorScopeEntry* scope,
                          const std::string& path, std::string* error) {
  *scope = {};
  if (table == nullptr) return true;
  scope->configured = true;
  scope->enabled = true;
  if (!validateKeys(*table, {"enabled", "compressor", "config"}, path,
                    error) ||
      !readBool(*table, "enabled", &scope->enabled, path, error) ||
      !readString(*table, "compressor", &scope->name, scope->enabled, path,
                  error)) {
    return false;
  }
  if (!scope->enabled) {
    if (!scope->name.empty() || table->get("config") != nullptr) {
      return fail(error, path +
                             " disabled scope cannot configure a compressor");
    }
    return true;
  }
  if (!validCompressorName(scope->name)) {
    return fail(error, path + ".compressor contains invalid compressor name '" +
                           scope->name + "'");
  }
  const toml::table* config = optionalTable(*table, "config", path, error);
  if (config == nullptr) return error == nullptr || error->empty();
  return parseOptionValues(*config, &scope->values, path + ".config", error);
}

bool parsePrimitivePolicy(const toml::table* table,
                          cocclPrimitivePolicy* policy,
                          size_t defaultThreshold, const std::string& path,
                          std::string* error) {
  policy->scopes = {};
  policy->thresholdBytes = defaultThreshold;
  if (table == nullptr) return true;
  if (!validateKeys(*table,
                    {"threshold_bytes", "default", "intra", "inter"},
                    path, error) ||
      !readSize(*table, "threshold_bytes", &policy->thresholdBytes, path,
                error)) {
    return false;
  }
  const toml::table* defaults = optionalTable(*table, "default", path, error);
  const toml::table* intra = optionalTable(*table, "intra", path, error);
  const toml::table* inter = optionalTable(*table, "inter", path, error);
  if (error != nullptr && !error->empty()) return false;
  return parseCompressorScope(
             defaults,
             &policy->scopes[static_cast<size_t>(
                 cocclCompressionScope::Default)],
             path + ".default", error) &&
         parseCompressorScope(
             intra,
             &policy->scopes[static_cast<size_t>(
                 cocclCompressionScope::Intra)],
             path + ".intra", error) &&
         parseCompressorScope(
             inter,
             &policy->scopes[static_cast<size_t>(
                 cocclCompressionScope::Inter)],
             path + ".inter", error);
}

bool parseCollectiveChildren(const toml::table* table,
                             cocclCollectivePolicies* policies,
                             size_t defaultThreshold, const std::string& path,
                             std::string* error) {
  const toml::table empty;
  const toml::table& parent = table == nullptr ? empty : *table;
  const toml::table* allGather = optionalTable(parent, "all_gather", path, error);
  if (allGather == nullptr && error != nullptr && !error->empty()) return false;
  const toml::table* reduceScatter =
      optionalTable(parent, "reduce_scatter", path, error);
  if (reduceScatter == nullptr && error != nullptr && !error->empty()) return false;
  const toml::table* allReduce = optionalTable(parent, "all_reduce", path, error);
  if (allReduce == nullptr && error != nullptr && !error->empty()) return false;
  return parsePrimitivePolicy(allGather, &policies->allGather, defaultThreshold,
                              path + ".all_gather", error) &&
         parsePrimitivePolicy(reduceScatter, &policies->reduceScatter,
                              defaultThreshold, path + ".reduce_scatter", error) &&
         parsePrimitivePolicy(allReduce, &policies->allReduce, defaultThreshold,
                              path + ".all_reduce", error);
}

bool parseCollectivePolicies(const toml::table* table,
                             cocclCollectivePolicies* policies,
                             size_t defaultThreshold, const std::string& path,
                             std::string* error) {
  return (table == nullptr ||
          validateKeys(*table, {"all_gather", "reduce_scatter", "all_reduce"},
                       path, error)) &&
         parseCollectiveChildren(table, policies, defaultThreshold, path,
                                 error);
}

bool parseNormal(const toml::table* table, cocclConfig* config,
                 std::string* error) {
  const std::string path = "normal";
  if (table != nullptr &&
      !validateKeys(*table, {"all_gather", "reduce_scatter", "all_reduce",
                             "all_to_all", "sendrecv"},
                    path, error)) {
    return false;
  }
  const toml::table empty;
  const toml::table& parent = table == nullptr ? empty : *table;
  if (!parseCollectiveChildren(table, &config->normal,
                               config->runtime.compressionThresholdBytes, path,
                               error)) return false;
  const toml::table* allToAll = optionalTable(parent, "all_to_all", path, error);
  const toml::table* sendRecv = optionalTable(parent, "sendrecv", path, error);
  return (error == nullptr || error->empty()) &&
         parsePrimitivePolicy(allToAll, &config->normal.allToAll,
                              config->runtime.compressionThresholdBytes,
                              path + ".all_to_all", error) &&
         parsePrimitivePolicy(sendRecv, &config->normal.sendRecv,
                              config->runtime.compressionThresholdBytes,
                              path + ".sendrecv", error);
}

bool parseTraining(const toml::table* table, cocclConfig* config,
                   std::string* error) {
  const std::string path = "training";
  if (table != nullptr &&
      !validateKeys(*table, {"observation_iterations", "max_events", "dp",
                             "tp", "pp"},
                    path, error)) {
    return false;
  }
  const toml::table empty;
  const toml::table& parent = table == nullptr ? empty : *table;
  if (!readInt(parent, "observation_iterations",
               &config->training.observationIterations, 2, 100, path, error) ||
      !readSize(parent, "max_events", &config->training.maxEvents, path,
                error)) {
    return false;
  }

  const toml::table* dp = optionalTable(parent, "dp", path, error);
  const toml::table* tp = optionalTable(parent, "tp", path, error);
  const toml::table* pp = optionalTable(parent, "pp", path, error);
  if ((error != nullptr && !error->empty()) ||
      !parseCollectivePolicies(dp, &config->trainingPolicies.dataParallel,
                               config->runtime.compressionThresholdBytes,
                               path + ".dp", error) ||
      !parseCollectivePolicies(tp, &config->trainingPolicies.tensorParallel,
                               config->runtime.compressionThresholdBytes,
                               path + ".tp", error)) {
    return false;
  }

  if (pp != nullptr && !validateKeys(*pp, {"sendrecv"}, path + ".pp", error)) {
    return false;
  }
  const toml::table& ppParent = pp == nullptr ? empty : *pp;
  const toml::table* sendRecv =
      optionalTable(ppParent, "sendrecv", path + ".pp", error);
  if (sendRecv != nullptr &&
      !validateKeys(*sendRecv, {"forward", "backward"},
                    path + ".pp.sendrecv", error)) {
    return false;
  }
  const toml::table& sendRecvParent = sendRecv == nullptr ? empty : *sendRecv;
  const toml::table* forward = optionalTable(
      sendRecvParent, "forward", path + ".pp.sendrecv", error);
  const toml::table* backward = optionalTable(
      sendRecvParent, "backward", path + ".pp.sendrecv", error);
  return (error == nullptr || error->empty()) &&
         parsePrimitivePolicy(
             forward, &config->trainingPolicies.pipelineSendRecvForward,
             config->runtime.compressionThresholdBytes,
             path + ".pp.sendrecv.forward", error) &&
         parsePrimitivePolicy(
             backward, &config->trainingPolicies.pipelineSendRecvBackward,
             config->runtime.compressionThresholdBytes,
             path + ".pp.sendrecv.backward", error);
}

bool parseRuntime(const toml::table& root, cocclConfig* config,
                  std::string* error) {
  const toml::table* runtime = optionalTable(root, "runtime", "root", error);
  if (runtime == nullptr) {
    return error != nullptr && !error->empty()
        ? false : fail(error, "runtime table is required");
  }
  if (!validateKeys(*runtime, {"mode", "compression_threshold_bytes"},
                    "runtime", error)) {
    return false;
  }
  std::string mode;
  if (!readString(*runtime, "mode", &mode, true, "runtime", error) ||
      !readSize(*runtime, "compression_threshold_bytes",
                &config->runtime.compressionThresholdBytes, "runtime", error)) {
    return false;
  }
  if (mode == "normal") {
    config->runtime.mode = cocclRuntimeMode::Normal;
  } else if (mode == "training") {
    config->runtime.mode = cocclRuntimeMode::Training;
  } else {
    return fail(error, "runtime.mode must be 'normal' or 'training'");
  }
  return true;
}

std::string configDirectory(const std::string& path) {
  size_t slash = path.find_last_of('/');
  return slash == std::string::npos ? "." : path.substr(0, slash);
}

std::string resolvePath(const std::string& configPath,
                        const std::string& path) {
  if (path.empty() || path.front() == '/') return path;
  std::string directory = configDirectory(configPath);
  return directory == "." ? path : directory + "/" + path;
}

bool parsePlugins(const toml::table& root, const std::string& configPath,
                  cocclConfig* config, std::string* error) {
  const toml::table* plugins =
      optionalTable(root, "compressor_plugins", "root", error);
  if (plugins == nullptr) return error == nullptr || error->empty();
  if (!validateKeys(*plugins, {"compressors", "library_path"},
                    "compressor_plugins", error) ||
      !readStringArray(*plugins, "compressors", &config->plugins.compressors,
                       "compressor_plugins", error) ||
      !readString(*plugins, "library_path", &config->plugins.libraryPath,
                  false, "compressor_plugins", error)) {
    return false;
  }
  if (!config->plugins.compressors.empty() &&
      config->plugins.libraryPath.empty()) {
    return fail(error,
                "compressor_plugins.library_path is required when compressors are configured");
  }
  config->plugins.libraryPath =
      resolvePath(configPath, config->plugins.libraryPath);
  return true;
}

bool parsePipeline(const toml::table& root, cocclConfig* config,
                   std::string* error) {
  const toml::table* pipeline = optionalTable(root, "pipeline", "root", error);
  if (pipeline == nullptr) return error == nullptr || error->empty();
  return validateKeys(*pipeline, {"depth"}, "pipeline", error) &&
         readInt(*pipeline, "depth", &config->pipeline.depth,
                 kCocclMinPipelineDepth, kCocclMaxPipelineDepth, "pipeline",
                 error);
}

bool parseBuffer(const toml::table& root, cocclConfig* config,
                 std::string* error) {
  const toml::table* buffer = optionalTable(root, "buffer", "root", error);
  if (buffer == nullptr) return error == nullptr || error->empty();
  return validateKeys(*buffer,
                      {"pool_limit_bytes", "legacy_block_bytes",
                       "physical_chunk_bytes"},
                      "buffer", error) &&
         readSize(*buffer, "pool_limit_bytes", &config->buffer.poolLimitBytes,
                  "buffer", error) &&
         readSize(*buffer, "legacy_block_bytes",
                  &config->buffer.legacyBlockBytes, "buffer", error) &&
         readSize(*buffer, "physical_chunk_bytes",
                  &config->buffer.physicalChunkBytes, "buffer", error) &&
         (config->buffer.physicalChunkBytes > 0 ||
          fail(error, "buffer.physical_chunk_bytes must be greater than zero"));
}

bool parseAutotune(const toml::table& root, cocclConfig* config,
                   std::string* error) {
  const toml::table* autotune = optionalTable(root, "autotune", "root", error);
  if (autotune == nullptr) return error == nullptr || error->empty();
  if (!validateKeys(*autotune,
                    {"enabled", "profile_min_bytes", "profile_max_bytes",
                     "warmup", "iterations", "reduce_scatter_algorithm",
                     "all_reduce_algorithm"},
                    "autotune", error) ||
      !readBool(*autotune, "enabled", &config->autotune.enabled, "autotune",
                error) ||
      !readSize(*autotune, "profile_min_bytes",
                &config->autotune.profileMinBytes, "autotune", error) ||
      !readSize(*autotune, "profile_max_bytes",
                &config->autotune.profileMaxBytes, "autotune", error) ||
      !readInt(*autotune, "warmup", &config->autotune.warmup, 0, 100000,
               "autotune", error) ||
      !readInt(*autotune, "iterations", &config->autotune.iterations, 1,
               100000, "autotune", error)) {
    return false;
  }
  if (config->autotune.profileMinBytes == 0 ||
      config->autotune.profileMaxBytes < config->autotune.profileMinBytes) {
    return fail(error,
                "autotune profile sizes require 0 < profile_min_bytes <= profile_max_bytes");
  }

  std::string rs = "auto";
  std::string ar = "auto";
  if (!readString(*autotune, "reduce_scatter_algorithm", &rs, false,
                  "autotune", error) ||
      !readString(*autotune, "all_reduce_algorithm", &ar, false, "autotune",
                  error)) {
    return false;
  }
  if (rs == "auto") config->autotune.reduceScatterAlgorithm = cocclReduceScatterAlgorithmPolicy::Auto;
  else if (rs == "oneshot") config->autotune.reduceScatterAlgorithm = cocclReduceScatterAlgorithmPolicy::OneShot;
  else if (rs == "twoshot") config->autotune.reduceScatterAlgorithm = cocclReduceScatterAlgorithmPolicy::TwoShot;
  else return fail(error, "autotune.reduce_scatter_algorithm must be auto, oneshot, or twoshot");

  if (ar == "auto") config->autotune.allReduceAlgorithm = cocclAllReduceAlgorithmPolicy::Auto;
  else if (ar == "oneshot") config->autotune.allReduceAlgorithm = cocclAllReduceAlgorithmPolicy::OneShot;
  else if (ar == "twoshot") config->autotune.allReduceAlgorithm = cocclAllReduceAlgorithmPolicy::TwoShot;
  else if (ar == "tripleshot") config->autotune.allReduceAlgorithm = cocclAllReduceAlgorithmPolicy::TripleShot;
  else return fail(error, "autotune.all_reduce_algorithm must be auto, oneshot, twoshot, or tripleshot");
  return true;
}

bool validateCatalogReferences(const cocclConfig& config, std::string* error) {
  const std::set<std::string> catalog(config.plugins.compressors.begin(),
                                      config.plugins.compressors.end());
  for (const cocclConfigPolicyView& configuredPolicy :
       cocclEnumeratePolicies(config)) {
    for (const cocclCompressorScopeEntry& scope :
         configuredPolicy.policy->scopes) {
      if (scope.enabled && catalog.count(scope.name) == 0) {
        return fail(error, "compressor '" + scope.name +
                               "' is used by a policy but missing from compressor_plugins.compressors");
      }
    }
  }
  return true;
}

bool parseConfigFileImpl(const std::string& path, cocclConfig* config,
                         std::string* error) {
  try {
    toml::table root = toml::parse_file(path);
    if (!validateKeys(root,
                      {"schema_version", "runtime", "compressor_plugins",
                       "pipeline", "buffer", "autotune", "normal",
                       "training"},
                      "root", error)) {
      return false;
    }
    const toml::node* versionNode = root.get("schema_version");
    auto version = versionNode == nullptr
        ? std::optional<int64_t>() : versionNode->value<int64_t>();
    if (!version || *version != kCocclConfigSchemaVersion) {
      return fail(error, "schema_version must be " +
                             std::to_string(kCocclConfigSchemaVersion));
    }
    if (!parseRuntime(root, config, error) ||
        !parsePlugins(root, path, config, error) ||
        !parsePipeline(root, config, error) ||
        !parseBuffer(root, config, error) ||
        !parseAutotune(root, config, error)) {
      return false;
    }
    const toml::table* normal = optionalTable(root, "normal", "root", error);
    const toml::table* training = optionalTable(root, "training", "root", error);
    return (error == nullptr || error->empty()) &&
           parseNormal(normal, config, error) &&
           parseTraining(training, config, error) &&
           validateCatalogReferences(*config, error);
  } catch (const toml::parse_error& parseError) {
    std::ostringstream message;
    message << parseError;
    return fail(error, message.str());
  } catch (const std::exception& exception) {
    return fail(error, exception.what());
  }
}

}  // namespace

const cocclCompressorScopeEntry& cocclConfiguredCompressorScope(
    const cocclPrimitivePolicy& policy, cocclCompressionScope scope) {
  return policy.scopes[static_cast<size_t>(scope)];
}

cocclEffectiveCompressorScope cocclEffectiveCompressorScopeFor(
    const cocclPrimitivePolicy& policy, cocclCompressionScope scope) {
  const cocclCompressorScopeEntry& configured =
      cocclConfiguredCompressorScope(policy, scope);
  if (configured.configured) return {&configured, scope};
  const cocclCompressorScopeEntry& defaults =
      cocclConfiguredCompressorScope(
          policy, cocclCompressionScope::Default);
  return defaults.configured
      ? cocclEffectiveCompressorScope{
            &defaults, cocclCompressionScope::Default}
      : cocclEffectiveCompressorScope{};
}

cocclConfigPolicyList cocclEnumeratePolicies(const cocclConfig& config) {
  return {{
      {cocclConfigPolicyId::NormalAllGather, "normal.all_gather",
       cocclRuntimeMode::Normal, cocclPolicyScope::Normal,
       cocclDefaultPolicy(cocclOperation::AllGather),
       &config.normal.allGather},
      {cocclConfigPolicyId::NormalReduceScatter, "normal.reduce_scatter",
       cocclRuntimeMode::Normal, cocclPolicyScope::Normal,
       cocclDefaultPolicy(cocclOperation::ReduceScatter),
       &config.normal.reduceScatter},
      {cocclConfigPolicyId::NormalAllReduce, "normal.all_reduce",
       cocclRuntimeMode::Normal, cocclPolicyScope::Normal,
       cocclDefaultPolicy(cocclOperation::AllReduce),
       &config.normal.allReduce},
      {cocclConfigPolicyId::NormalAllToAll, "normal.all_to_all",
       cocclRuntimeMode::Normal, cocclPolicyScope::Normal,
       cocclDefaultPolicy(cocclOperation::AllToAll),
       &config.normal.allToAll},
      {cocclConfigPolicyId::NormalSendRecv, "normal.sendrecv",
       cocclRuntimeMode::Normal, cocclPolicyScope::Normal,
       cocclDefaultPolicy(cocclOperation::SendRecv),
       &config.normal.sendRecv},
      {cocclConfigPolicyId::TrainingDpAllGather, "training.dp.all_gather",
       cocclRuntimeMode::Training, cocclPolicyScope::DataParallel,
       cocclDefaultPolicy(cocclOperation::AllGather),
       &config.trainingPolicies.dataParallel.allGather},
      {cocclConfigPolicyId::TrainingDpReduceScatter,
       "training.dp.reduce_scatter", cocclRuntimeMode::Training,
       cocclPolicyScope::DataParallel,
       cocclDefaultPolicy(cocclOperation::ReduceScatter),
       &config.trainingPolicies.dataParallel.reduceScatter},
      {cocclConfigPolicyId::TrainingDpAllReduce, "training.dp.all_reduce",
       cocclRuntimeMode::Training, cocclPolicyScope::DataParallel,
       cocclDefaultPolicy(cocclOperation::AllReduce),
       &config.trainingPolicies.dataParallel.allReduce},
      {cocclConfigPolicyId::TrainingTpAllGather, "training.tp.all_gather",
       cocclRuntimeMode::Training, cocclPolicyScope::TensorParallel,
       cocclDefaultPolicy(cocclOperation::AllGather),
       &config.trainingPolicies.tensorParallel.allGather},
      {cocclConfigPolicyId::TrainingTpReduceScatter,
       "training.tp.reduce_scatter", cocclRuntimeMode::Training,
       cocclPolicyScope::TensorParallel,
       cocclDefaultPolicy(cocclOperation::ReduceScatter),
       &config.trainingPolicies.tensorParallel.reduceScatter},
      {cocclConfigPolicyId::TrainingTpAllReduce, "training.tp.all_reduce",
       cocclRuntimeMode::Training, cocclPolicyScope::TensorParallel,
       cocclDefaultPolicy(cocclOperation::AllReduce),
       &config.trainingPolicies.tensorParallel.allReduce},
      {cocclConfigPolicyId::TrainingPpSendRecvForward,
       "training.pp.sendrecv.forward", cocclRuntimeMode::Training,
       cocclPolicyScope::PipelineParallel,
       cocclDirectionalPolicy(cocclOperation::SendRecv, true),
       &config.trainingPolicies.pipelineSendRecvForward},
      {cocclConfigPolicyId::TrainingPpSendRecvBackward,
       "training.pp.sendrecv.backward", cocclRuntimeMode::Training,
       cocclPolicyScope::PipelineParallel,
       cocclDirectionalPolicy(cocclOperation::SendRecv, false),
       &config.trainingPolicies.pipelineSendRecvBackward},
  }};
}

bool cocclLoadConfigFile(const std::string& path, cocclConfig* config,
                         std::string* error) {
  if (error != nullptr) error->clear();
  if (config == nullptr) return fail(error, "config output is required");
  if (path.empty()) return fail(error, "config file path is empty");

  cocclConfig parsed;
  if (!parseConfigFileImpl(path, &parsed, error)) return false;
  *config = std::move(parsed);
  return true;
}

#ifndef COCCL_CONFIG_PARSER_ONLY
namespace {

std::once_flag configOnce;
cocclConfig processConfig;
std::atomic<const cocclConfig*> publishedConfig{nullptr};

void initializeConfigOnce() {
  const char* enabled = getenv("COCCL_ENABLE");
  if (enabled == nullptr || strcmp(enabled, "0") == 0) return;
  if (strcmp(enabled, "1") != 0) {
    WARN("COCCL_ENABLE must be 0 or 1; COCCL disabled");
    return;
  }

  const char* path = getenv("COCCL_CONFIG_FILE");
  if (path == nullptr || path[0] == '\0') {
    WARN("COCCL_CONFIG_FILE is required when COCCL_ENABLE=1; COCCL disabled");
    return;
  }

  cocclConfig parsed;
  std::string error;
  if (!cocclLoadConfigFile(path, &parsed, &error)) {
    WARN("COCCL config %s is invalid: %s; falling back to native NCCL",
         path, error.c_str());
    return;
  }

  processConfig = std::move(parsed);
  publishedConfig.store(&processConfig, std::memory_order_release);
  cocclLogEffectiveConfig(processConfig, path);
}

}  // namespace

bool cocclConfigInitialize() {
  std::call_once(configOnce, initializeConfigOnce);
  return publishedConfig.load(std::memory_order_acquire) != nullptr;
}

const cocclConfig& cocclGetConfig() {
  const cocclConfig* config =
      publishedConfig.load(std::memory_order_acquire);
  assert(config != nullptr &&
         "cocclGetConfig requires successful cocclConfigInitialize");
  return *config;
}
#endif
