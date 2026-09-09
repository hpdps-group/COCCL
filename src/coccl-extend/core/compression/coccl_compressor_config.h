#ifndef COCCL_COMPRESSOR_CONFIG_H_
#define COCCL_COMPRESSOR_CONFIG_H_

#include "core/config/coccl_config.h"
#include "compressor_plugin/detail/coccl_compressor_abi.h"

#include <string>
#include <vector>

class cocclCompressorConfigViewStorage {
 public:
  explicit cocclCompressorConfigViewStorage(const cocclConfigValues& values) {
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

inline std::string cocclCompressorPluginPath(const cocclConfig& config, const std::string& name) {
  std::string path = config.plugins.libraryPath;
  if (!path.empty() && path.back() != '/') path.push_back('/');
  return path + "lib" + name + ".so";
}

#endif
