#include "coccl_config_debug.h"

#include "debug.h"

void cocclLogEffectiveConfig(const cocclConfig& config,
                             const char* sourcePath) {
  INFO(NCCL_ENV, "COCCL loaded %s configuration from %s",
       config.runtime.mode == cocclRuntimeMode::Training ? "training"
                                                         : "normal",
       sourcePath);
  INFO(NCCL_ENV, "COCCL effective configuration begin");
  for (const std::string& line : cocclFormatEffectiveConfig(config)) {
    // NCCL's logger uses a bounded per-message buffer, so keep each setting in
    // an independent record instead of building one large multi-line message.
    INFO(NCCL_ENV, "COCCL config %s", line.c_str());
  }
  INFO(NCCL_ENV, "COCCL effective configuration end");
}
