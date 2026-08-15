#ifndef COCCL_CONFIG_DEBUG_H_
#define COCCL_CONFIG_DEBUG_H_

#include "coccl_config.h"

#include <string>
#include <vector>

// Debug-only presentation helpers. Formatting is kept separate from TOML
// parsing so the runtime configuration model remains focused on execution.
std::vector<std::string> cocclFormatEffectiveConfig(
    const cocclConfig& config);
void cocclLogEffectiveConfig(const cocclConfig& config,
                             const char* sourcePath);

#endif
