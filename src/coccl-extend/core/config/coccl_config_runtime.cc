#include "core/config/coccl_config.h"

#include "core/config/coccl_config_debug.h"
#include "debug.h"

#include <atomic>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <utility>

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
  assert(config != nullptr);
  return *config;
}
