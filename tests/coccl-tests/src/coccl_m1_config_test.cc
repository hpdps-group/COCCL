#include "core/config/coccl_config.h"

#include <stdio.h>

#include <string>

namespace {

bool valueIs(const cocclConfigValues& values, const char* key,
             const char* expected) {
  auto value = values.find(key);
  return value != values.end() && value->second == expected;
}

const cocclCompressorScopeEntry& scope(
    const cocclPrimitivePolicy& policy, cocclCompressionScope value) {
  return cocclConfiguredCompressorScope(policy, value);
}

int checkCatalog(const cocclConfig& config) {
  return config.plugins.compressors.size() == 2 &&
         config.plugins.compressors[0] == "sdp4bit" &&
         config.plugins.compressors[1] == "zfp" &&
         config.plugins.libraryPath.find(
             "build/obj/coccl-extend/compressor_plugin/libcompress") !=
             std::string::npos
      ? 0 : 1;
}

int checkSdp(const char* path) {
  cocclConfig config;
  std::string error;
  if (!cocclLoadConfigFile(path, &config, &error)) {
    fprintf(stderr, "failed to parse SDP4Bit TOML: %s\n", error.c_str());
    return 1;
  }
  if (checkCatalog(config) || config.runtime.compressionThresholdBytes != 0 ||
      scope(config.normal.allToAll, cocclCompressionScope::Default).name !=
          "sdp4bit" ||
      scope(config.normal.allGather, cocclCompressionScope::Default).name !=
          "sdp4bit" ||
      scope(config.normal.allReduce, cocclCompressionScope::Default).name !=
          "sdp4bit" ||
      scope(config.normal.reduceScatter,
            cocclCompressionScope::Default).name != "sdp4bit") {
    fprintf(stderr, "SDP4Bit policy mapping is incomplete\n");
    return 1;
  }
  const auto& a2a = scope(
      config.normal.allToAll, cocclCompressionScope::Default).values;
  const auto& ar = scope(
      config.normal.allReduce, cocclCompressionScope::Default).values;
  const auto& arInter = scope(
      config.normal.allReduce, cocclCompressionScope::Inter).values;
  const auto& rs = scope(
      config.normal.reduceScatter, cocclCompressionScope::Default).values;
  if (!valueIs(a2a, "groupCount", "2048") ||
      !valueIs(a2a, "quantBits", "4") ||
      !valueIs(a2a, "quantType", "Symmetric") ||
      !valueIs(ar, "groupCount", "128") ||
      !valueIs(ar, "hadamard", "false") ||
      !valueIs(arInter, "quantBits", "4") ||
      !valueIs(arInter, "groupCount", "128") ||
      !valueIs(arInter, "hadamard", "true") ||
      arInter.count("inQuantBits") != 0 ||
      arInter.count("inGroupCount") != 0 ||
      !valueIs(rs, "groupCount", "128") ||
      !valueIs(rs, "hadamard", "true")) {
    fprintf(stderr, "SDP4Bit TOML values differ from the M0 configs\n");
    return 1;
  }
  return 0;
}

int checkZfp(const char* path) {
  cocclConfig config;
  std::string error;
  if (!cocclLoadConfigFile(path, &config, &error)) {
    fprintf(stderr, "failed to parse ZFP TOML: %s\n", error.c_str());
    return 1;
  }
  if (checkCatalog(config) ||
      scope(config.normal.allToAll, cocclCompressionScope::Default).name !=
          "zfp" ||
      scope(config.normal.allGather, cocclCompressionScope::Default).name !=
          "zfp" ||
      scope(config.normal.allReduce, cocclCompressionScope::Default).name !=
          "zfp" ||
      scope(config.normal.reduceScatter,
            cocclCompressionScope::Default).name != "zfp" ||
      !valueIs(scope(config.normal.allToAll,
                     cocclCompressionScope::Default).values,
               "rate", "4") ||
      !valueIs(scope(config.normal.allGather,
                     cocclCompressionScope::Default).values,
               "rate", "8") ||
      !valueIs(scope(config.normal.allReduce,
                     cocclCompressionScope::Default).values,
               "rate", "8") ||
      !valueIs(scope(config.normal.reduceScatter,
                     cocclCompressionScope::Default).values,
               "rate", "8")) {
    fprintf(stderr, "ZFP policy mapping differs from the M0 configs\n");
    return 1;
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3 || checkSdp(argv[1]) || checkZfp(argv[2])) return 1;
  printf("COCCL M1 TOML policy tests passed\n");
  return 0;
}
