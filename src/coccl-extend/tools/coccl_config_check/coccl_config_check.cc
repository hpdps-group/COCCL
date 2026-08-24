#include "core/config/coccl_config.h"
#include "core/config/coccl_config_debug.h"
#include "compressor_plugin/detail/coccl_compressor_abi.h"

#include <dlfcn.h>

#include <charconv>
#include <cstdio>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Options {
  std::string configPath;
  int nodes = 2;
  int devicesPerNode = 8;
};

struct LoadedPlugin {
  void* library = nullptr;
  const cocclCompressorPlugin* compressor = nullptr;
  std::string path;
};

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

void usage(const char* program) {
  fprintf(stderr,
          "Usage: %s [--nodes N] [--devices-per-node N] <config.toml>\n",
          program);
}

bool parsePositiveInt(const char* text, int* value) {
  if (text == nullptr || value == nullptr || text[0] == '\0') return false;
  const char* end = text;
  while (*end != '\0') ++end;
  int parsed = 0;
  auto result = std::from_chars(text, end, parsed, 10);
  if (result.ec != std::errc() || result.ptr != end || parsed <= 0) {
    return false;
  }
  *value = parsed;
  return true;
}

bool parseOptions(int argc, char** argv, Options* options) {
  if (options == nullptr) return false;
  for (int i = 1; i < argc; ++i) {
    const std::string argument(argv[i]);
    if (argument == "--nodes" || argument == "--devices-per-node") {
      if (++i >= argc) return false;
      int* destination = argument == "--nodes"
          ? &options->nodes : &options->devicesPerNode;
      if (!parsePositiveInt(argv[i], destination)) return false;
    } else if (!argument.empty() && argument.front() == '-') {
      return false;
    } else if (!options->configPath.empty()) {
      return false;
    } else {
      options->configPath = argument;
    }
  }
  return !options->configPath.empty();
}

std::string pluginPath(const cocclConfig& config, const std::string& name) {
  std::string path = config.plugins.libraryPath;
  if (!path.empty() && path.back() != '/') path.push_back('/');
  return path + "lib" + name + ".so";
}

bool validateDescriptor(const std::string& expectedName,
                        const cocclCompressorPlugin* compressor,
                        std::string* error) {
  char reason[192] = {};
  if (cocclValidateCompressorPlugin(
          expectedName.c_str(), compressor, reason, sizeof(reason))) {
    return true;
  }
  std::ostringstream message;
  message << "invalid descriptor for '" << expectedName << "': " << reason;
  if (compressor != nullptr) {
    message << " (abi=" << compressor->abiVersion
            << ", size=" << compressor->structSize << ")";
  }
  *error = message.str();
  return false;
}

bool loadPlugin(const cocclConfig& config, const std::string& name,
                LoadedPlugin* plugin, std::string* error) {
  *plugin = {};
  plugin->path = pluginPath(config, name);
  plugin->library = dlopen(plugin->path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (plugin->library == nullptr) {
    const char* loaderError = dlerror();
    *error = "failed to open " + plugin->path;
    if (loaderError != nullptr) *error += ": " + std::string(loaderError);
    return false;
  }
  dlerror();
  cocclGetCompressorPluginFn getPlugin =
      reinterpret_cast<cocclGetCompressorPluginFn>(
          dlsym(plugin->library, COCCL_COMPRESSOR_ENTRY_SYMBOL));
  const char* symbolError = dlerror();
  if (symbolError != nullptr) {
    *error = "failed to resolve symbol '" +
        std::string(COCCL_COMPRESSOR_ENTRY_SYMBOL) + "': " + symbolError;
  } else {
    plugin->compressor = getPlugin == nullptr ? nullptr : getPlugin();
  }
  if (symbolError == nullptr &&
      validateDescriptor(name, plugin->compressor, error)) {
    return true;
  }
  dlclose(plugin->library);
  *plugin = {};
  return false;
}

bool validatePluginConfig(const cocclCompressorPlugin* compressor,
                          const cocclConfigValues& values,
                          cocclCompressorConfigVariant variant,
                          const Options& options, const std::string& label) {
  ConfigViewStorage storage(values);
  const cocclConfigView view = storage.view();
  const cocclCompressorConfigContext context = {
      variant, options.nodes, options.devicesPerNode};
  void* parsedConfig = nullptr;
  char error[256] = {};
  ncclResult_t result = compressor->parseConfig(
      &view, &context, &parsedConfig, error, sizeof(error));
  if (result != ncclSuccess) {
    if (parsedConfig != nullptr) compressor->destroyConfig(parsedConfig);
    fprintf(stderr,
            "[ERROR] %s: compressor %s rejected configuration (result=%d%s%s)\n",
            label.c_str(), compressor->name, (int)result,
            error[0] == '\0' ? "" : ", error=",
            error[0] == '\0' ? "" : error);
    return false;
  }
  if (parsedConfig != nullptr) compressor->destroyConfig(parsedConfig);
  printf("[OK] %s (%s)\n", label.c_str(), compressor->name);
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  Options options;
  if (!parseOptions(argc, argv, &options)) {
    usage(argv[0]);
    return 2;
  }

  cocclConfig config;
  std::string error;
  if (!cocclLoadConfigFile(options.configPath, &config, &error)) {
    fprintf(stderr, "[ERROR] %s: %s\n", options.configPath.c_str(),
            error.c_str());
    return 1;
  }

  printf("COCCL effective configuration (%s):\n",
         options.configPath.c_str());
  for (const std::string& line : cocclFormatEffectiveConfig(config)) {
    printf("  %s\n", line.c_str());
  }
  printf("\nValidating compressor plugins and policy parameters "
         "(nodes=%d, devices-per-node=%d):\n",
         options.nodes, options.devicesPerNode);

  bool valid = true;
  std::map<std::string, LoadedPlugin> loadedPlugins;
  for (const std::string& name : config.plugins.compressors) {
    LoadedPlugin plugin;
    if (!loadPlugin(config, name, &plugin, &error)) {
      fprintf(stderr, "[ERROR] compressor %s: %s\n", name.c_str(),
              error.c_str());
      valid = false;
      continue;
    }
    printf("[OK] plugin %s: %s\n", name.c_str(), plugin.path.c_str());
    loadedPlugins.emplace(name, std::move(plugin));
  }

  const cocclConfigValues emptyValues;
  for (auto& loaded : loadedPlugins) {
    valid = validatePluginConfig(
                loaded.second.compressor, emptyValues,
                cocclCompressorConfigDefault, options,
                "compressor_plugins." + loaded.first + ".defaults") &&
        valid;
  }

  size_t validatedBindings = 0;
  for (const cocclConfigPolicyView& policy :
       cocclEnumeratePolicies(config)) {
    constexpr const char* kScopeNames[] = {"default", "intra", "inter"};
    for (size_t scopeIndex = 0;
         scopeIndex < static_cast<size_t>(cocclCompressionScope::Count);
         ++scopeIndex) {
      const cocclCompressionScope scope =
          static_cast<cocclCompressionScope>(scopeIndex);
      const cocclCompressorScopeEntry& configured =
          cocclConfiguredCompressorScope(*policy.policy, scope);
      if (!configured.configured || !configured.enabled) continue;
      auto plugin = loadedPlugins.find(configured.name);
      if (plugin == loadedPlugins.end()) {
        valid = false;
        continue;
      }
      const std::string label = std::string(policy.path) + "." +
          kScopeNames[scopeIndex] + ".config";
      valid = validatePluginConfig(
                  plugin->second.compressor, configured.values,
                  cocclCompressorConfigDefault, options, label) &&
          valid;
      ++validatedBindings;
    }
  }

  for (auto& loaded : loadedPlugins) {
    if (loaded.second.library != nullptr) dlclose(loaded.second.library);
  }
  if (!valid) {
    fprintf(stderr, "COCCL configuration check failed\n");
    return 1;
  }
  printf("COCCL configuration check passed: %zu plugins, "
         "%zu policy configurations\n",
         loadedPlugins.size(), validatedBindings);
  return 0;
}
