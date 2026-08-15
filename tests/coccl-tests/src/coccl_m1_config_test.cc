#include "coccl_config.h"

#include <stdio.h>

#include <string>

namespace {

bool valueIs(const cocclConfigValues& values, const char* key,
             const char* expected) {
  auto value = values.find(key);
  return value != values.end() && value->second == expected;
}

int checkCatalog(const cocclConfig& config) {
  return config.plugins.compressors.size() == 2 &&
         config.plugins.compressors[0] == "sdp4bit" &&
         config.plugins.compressors[1] == "zfp" &&
         config.plugins.libraryPath.find(
             "build/obj/device/compress/libcompress") != std::string::npos
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
      config.normal.allToAll.compressor.name != "sdp4bit" ||
      config.normal.allGather.compressor.name != "sdp4bit" ||
      config.normal.allReduce.compressor.name != "sdp4bit" ||
      config.normal.reduceScatter.compressor.name != "sdp4bit") {
    fprintf(stderr, "SDP4Bit policy mapping is incomplete\n");
    return 1;
  }
  const auto& a2a = config.normal.allToAll.compressor.defaultValues;
  const auto& ar = config.normal.allReduce.compressor.defaultValues;
  const auto& arHier =
      config.normal.allReduce.compressor.hierarchicalValues;
  const auto& rs = config.normal.reduceScatter.compressor.defaultValues;
  if (!valueIs(a2a, "groupCount", "2048") ||
      !valueIs(a2a, "quantBits", "4") ||
      !valueIs(a2a, "quantType", "Symmetric") ||
      !valueIs(ar, "groupCount", "128") ||
      !valueIs(ar, "hadamard", "false") ||
      !valueIs(arHier, "inQuantBits", "4") ||
      !valueIs(arHier, "outQuantBits", "4") ||
      !valueIs(arHier, "hadamard", "true") ||
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
      config.normal.allToAll.compressor.name != "zfp" ||
      config.normal.allGather.compressor.name != "zfp" ||
      config.normal.allReduce.compressor.name != "zfp" ||
      config.normal.reduceScatter.compressor.name != "zfp" ||
      !valueIs(config.normal.allToAll.compressor.defaultValues,
               "rate", "4") ||
      !valueIs(config.normal.allGather.compressor.defaultValues,
               "rate", "8") ||
      !valueIs(config.normal.allReduce.compressor.defaultValues,
               "rate", "8") ||
      !valueIs(config.normal.reduceScatter.compressor.defaultValues,
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
