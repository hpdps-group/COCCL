#include "core/config/coccl_config_debug.h"

namespace {

const char* runtimeModeName(cocclRuntimeMode mode) {
  return mode == cocclRuntimeMode::Training ? "training" : "normal";
}

const char* reduceScatterAlgorithmName(
    cocclReduceScatterAlgorithmPolicy algorithm) {
  switch (algorithm) {
    case cocclReduceScatterAlgorithmPolicy::OneShot: return "oneshot";
    case cocclReduceScatterAlgorithmPolicy::TwoShot: return "twoshot";
    case cocclReduceScatterAlgorithmPolicy::Auto: return "auto";
  }
  return "unknown";
}

const char* allReduceAlgorithmName(cocclAllReduceAlgorithmPolicy algorithm) {
  switch (algorithm) {
    case cocclAllReduceAlgorithmPolicy::OneShot: return "oneshot";
    case cocclAllReduceAlgorithmPolicy::TwoShot: return "twoshot";
    case cocclAllReduceAlgorithmPolicy::TripleShot: return "tripleshot";
    case cocclAllReduceAlgorithmPolicy::Auto: return "auto";
  }
  return "unknown";
}

std::string quote(const std::string& value) {
  std::string quoted;
  quoted.reserve(value.size() + 2);
  quoted.push_back('"');
  for (unsigned char c : value) {
    switch (c) {
      case '\\': quoted += "\\\\"; break;
      case '"': quoted += "\\\""; break;
      case '\n': quoted += "\\n"; break;
      case '\r': quoted += "\\r"; break;
      case '\t': quoted += "\\t"; break;
      default:
        if (c >= 0x20) {
          quoted.push_back((char)c);
        } else {
          const char digits[] = "0123456789abcdef";
          quoted += "\\u00";
          quoted.push_back(digits[c >> 4]);
          quoted.push_back(digits[c & 0xf]);
        }
    }
  }
  quoted.push_back('"');
  return quoted;
}

std::string formatStringArray(const std::vector<std::string>& values) {
  std::string formatted = "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) formatted += ", ";
    formatted += quote(values[i]);
  }
  formatted += "]";
  return formatted;
}

void appendConfigValues(std::vector<std::string>* lines,
                        const std::string& path,
                        const cocclConfigValues& values) {
  if (values.empty()) {
    lines->push_back(path + " = {}");
    return;
  }
  for (const auto& value : values) {
    lines->push_back(path + "." + value.first + " = " +
                     quote(value.second));
  }
}

void appendPolicy(std::vector<std::string>* lines, const std::string& path,
                  const cocclPrimitivePolicy& policy) {
  lines->push_back(path + ".enabled = " +
                   (policy.compressor.name.empty() ? "false" : "true"));
  lines->push_back(path + ".threshold_bytes = " +
                   std::to_string(policy.thresholdBytes));
  lines->push_back(path + ".compressor = " +
                   quote(policy.compressor.name));
  if (policy.compressor.name.empty()) return;
  appendConfigValues(lines, path + ".config.default",
                     policy.compressor.defaultValues);
  if (policy.compressor.hasHierarchicalConfig) {
    appendConfigValues(lines, path + ".config.hierarchical",
                       policy.compressor.hierarchicalValues);
  }
}

}  // namespace

std::vector<std::string> cocclFormatEffectiveConfig(
    const cocclConfig& config) {
  std::vector<std::string> lines;
  lines.push_back("schema_version = " +
                  std::to_string(kCocclConfigSchemaVersion));
  lines.push_back("runtime.mode = " + quote(runtimeModeName(config.runtime.mode)));
  lines.push_back("runtime.compression_threshold_bytes = " +
                  std::to_string(config.runtime.compressionThresholdBytes));
  lines.push_back("compressor_plugins.library_path = " +
                  quote(config.plugins.libraryPath));
  lines.push_back("compressor_plugins.compressors = " +
                  formatStringArray(config.plugins.compressors));
  lines.push_back("pipeline.depth = " + std::to_string(config.pipeline.depth));
  lines.push_back("buffer.pool_limit_bytes = " +
                  std::to_string(config.buffer.poolLimitBytes));
  lines.push_back("buffer.legacy_block_bytes = " +
                  std::to_string(config.buffer.legacyBlockBytes));
  lines.push_back("buffer.physical_chunk_bytes = " +
                  std::to_string(config.buffer.physicalChunkBytes));
  lines.push_back("autotune.enabled = " +
                  std::string(config.autotune.enabled ? "true" : "false"));
  lines.push_back("autotune.profile_min_bytes = " +
                  std::to_string(config.autotune.profileMinBytes));
  lines.push_back("autotune.profile_max_bytes = " +
                  std::to_string(config.autotune.profileMaxBytes));
  lines.push_back("autotune.warmup = " +
                  std::to_string(config.autotune.warmup));
  lines.push_back("autotune.iterations = " +
                  std::to_string(config.autotune.iterations));
  lines.push_back("autotune.reduce_scatter_algorithm = " +
                  quote(reduceScatterAlgorithmName(
                      config.autotune.reduceScatterAlgorithm)));
  lines.push_back("autotune.all_reduce_algorithm = " +
                  quote(allReduceAlgorithmName(
                      config.autotune.allReduceAlgorithm)));
  lines.push_back("training.observation_iterations = " +
                  std::to_string(config.training.observationIterations));
  lines.push_back("training.max_events = " +
                  std::to_string(config.training.maxEvents));

  for (const cocclConfigPolicyView& policy :
       cocclEnumeratePolicies(config)) {
    appendPolicy(&lines, policy.path, *policy.policy);
  }
  return lines;
}
