#include "compressor_plugin/detail/coccl_compressor_abi.h"

#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

#include <string>

namespace {

struct LoadedPlugin {
  void* library = nullptr;
  const cocclCompressorPlugin* descriptor = nullptr;
};

int loadPlugin(const char* directory, const char* name,
               LoadedPlugin* loaded) {
  const std::string path = std::string(directory) + "/lib" + name + ".so";
  loaded->library = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (loaded->library == nullptr) {
    fprintf(stderr, "%s: dlopen failed: %s\n", name, dlerror());
    return 1;
  }
  auto entry = reinterpret_cast<cocclGetCompressorPluginFn>(
      dlsym(loaded->library, COCCL_COMPRESSOR_ENTRY_SYMBOL));
  char error[160] = {};
  loaded->descriptor = entry == nullptr ? nullptr : entry();
  if (entry == nullptr ||
      !cocclValidateCompressorPlugin(name, loaded->descriptor, error,
                                     sizeof(error))) {
    fprintf(stderr, "%s: descriptor validation failed: %s\n", name, error);
    return 1;
  }
  return 0;
}

int parseConfig(const LoadedPlugin& plugin, const cocclConfigPair* pairs,
                size_t count, void** config) {
  const cocclConfigView values = {pairs, count};
  const cocclCompressorConfigContext context = {
      cocclCompressorConfigDefault, 2, 8};
  char error[160] = {};
  const ncclResult_t result = plugin.descriptor->parseConfig(
      &values, &context, config, error, sizeof(error));
  if (result != ncclSuccess) {
    fprintf(stderr, "%s: config failed: %s\n",
            plugin.descriptor->name, error);
    return 1;
  }
  return 0;
}

int queryBound(const LoadedPlugin& plugin, void* config,
               cocclCompressorOperation operation,
               size_t elements, size_t chunks, ncclDataType_t datatype,
               ncclResult_t expectedResult, size_t expectedBytes,
               size_t* actualBytes = nullptr) {
  const cocclCompressorSizeQuery query = {
      sizeof(cocclCompressorSizeQuery), operation,
      elements, chunks, datatype, config};
  size_t bytes = 0;
  const ncclResult_t result =
      plugin.descriptor->getEncodedSizeBound(&query, &bytes);
  if (result != expectedResult ||
      (result == ncclSuccess && expectedBytes != 0 &&
       bytes != expectedBytes)) {
    fprintf(stderr, "%s: operation %d expected result=%d bytes=%zu, "
                    "got result=%d bytes=%zu\n",
            plugin.descriptor->name, (int)operation, (int)expectedResult,
            expectedBytes, (int)result, bytes);
    return 1;
  }
  if (actualBytes != nullptr) *actualBytes = bytes;
  return 0;
}

int testQuantizedPlugin(const char* directory, const char* name,
                        bool tah) {
  LoadedPlugin plugin;
  if (loadPlugin(directory, name, &plugin)) return 1;
  const cocclConfigPair sdpPairs[] = {
      {"groupCount", "128"}, {"quantBits", "4"},
      {"hadamard", "false"}, {"quantType", "Symmetric"},
      {"inQuantBits", "4"}, {"outQuantBits", "8"},
      {"inGroupCount", "128"}, {"outGroupCount", "256"}};
  const cocclConfigPair tahPairs[] = {
      {"groupCount", "128"}, {"quantBits", "4"},
      {"hadamard", "false"}, {"pivotSwap", "false"},
      {"quantType", "Symmetric"}};
  void* config = nullptr;
  const int parseResult = tah
      ? parseConfig(plugin, tahPairs, 5, &config)
      : parseConfig(plugin, sdpPairs, 8, &config);
  int result = parseResult;
  if (result == 0) {
    result = queryBound(
        plugin, config, cocclCompressorOperationCompress,
        1024, 2, ncclFloat32, ncclSuccess, 544);
  }
  if (result == 0) {
    result = queryBound(
        plugin, config,
        cocclCompressorOperationDecompressReduceCompress,
        1024, 2, ncclFloat32, ncclSuccess, tah ? 544 : 1040);
  }
  if (config != nullptr) plugin.descriptor->destroyConfig(config);

  if (result == 0 && !tah) {
    const cocclConfigPair subAddPairs[] = {
        {"groupCount", "128"}, {"quantBits", "4"},
        {"hadamard", "false"}, {"quantType", "Symmetric"},
        {"subAdd", "true"}};
    config = nullptr;
    result = parseConfig(plugin, subAddPairs, 5, &config);
    if (result == 0) {
      result = queryBound(
          plugin, config, cocclCompressorOperationCompress,
          1024, 2, ncclFloat32, ncclSuccess, 4096);
    }
    if (config != nullptr) plugin.descriptor->destroyConfig(config);
  }
  dlclose(plugin.library);
  return result;
}

int testTaco(const char* directory) {
  LoadedPlugin plugin;
  if (loadPlugin(directory, "taco", &plugin)) return 1;
  const cocclConfigPair pairs[] = {{"groupSize", "128"}};
  void* config = nullptr;
  int result = parseConfig(plugin, pairs, 1, &config);
  if (result == 0) {
    result = queryBound(
        plugin, config, cocclCompressorOperationCompress,
        1024, 2, ncclFloat32, ncclSuccess, 1088);
  }
  if (result == 0) {
    result = queryBound(
        plugin, config,
        cocclCompressorOperationDecompressReduceCompress,
        1024, 2, ncclFloat32, ncclInvalidUsage, 0);
  }
  if (config != nullptr) plugin.descriptor->destroyConfig(config);
  dlclose(plugin.library);
  return result;
}

int testZfp(const char* directory) {
  LoadedPlugin plugin;
  if (loadPlugin(directory, "zfp", &plugin)) return 1;
  const cocclConfigPair pairs[] = {{"rate", "4"}};
  void* config = nullptr;
  int result = parseConfig(plugin, pairs, 1, &config);
  size_t first = 0;
  size_t second = 0;
  if (result == 0) {
    result = queryBound(
        plugin, config, cocclCompressorOperationCompress,
        1024, 2, ncclFloat32, ncclSuccess, 0, &first);
  }
  if (result == 0) {
    result = queryBound(
        plugin, config, cocclCompressorOperationCompress,
        1024, 2, ncclFloat32, ncclSuccess, 0, &second);
  }
  if (result == 0 && (first == 0 || first != second || first % 2 != 0)) {
    fprintf(stderr, "zfp: bound is zero, unstable, or not chunk aligned\n");
    result = 1;
  }
  if (result == 0) {
    result = queryBound(
        plugin, config,
        cocclCompressorOperationDecompressReduceCompress,
        1024, 2, ncclFloat32, ncclInvalidUsage, 0);
  }
  if (config != nullptr) plugin.descriptor->destroyConfig(config);
  dlclose(plugin.library);
  return result;
}

int testDietGpu(const char* directory) {
  LoadedPlugin plugin;
  if (loadPlugin(directory, "dietgpu", &plugin)) return 1;
  if ((plugin.descriptor->capabilities &
       cocclCompressorCapabilityFramed) == 0 ||
      (plugin.descriptor->capabilities &
       cocclCompressorCapabilityDecompressReduce) != 0 ||
      (plugin.descriptor->capabilities &
       cocclCompressorCapabilityDecompressReduceCompress) != 0) {
    fprintf(stderr, "dietgpu: framed capability set is invalid\n");
    dlclose(plugin.library);
    return 1;
  }
  const cocclConfigPair pairs[] = {{"probBits", "10"}};
  void* config = nullptr;
  int result = parseConfig(plugin, pairs, 1, &config);
  if (result == 0) {
    result = queryBound(
        plugin, config, cocclCompressorOperationCompress,
        1024, 2, ncclFloat32, ncclSuccess, 4096);
  }
  if (result == 0) {
    result = queryBound(
        plugin, config,
        cocclCompressorOperationDecompressReduceCompress,
        1024, 2, ncclFloat32, ncclInvalidUsage, 0);
  }
  if (config != nullptr) plugin.descriptor->destroyConfig(config);

  const cocclConfigPair invalidPairs[] = {{"probBits", "8"}};
  config = nullptr;
  if (result == 0) {
    const cocclConfigView values = {invalidPairs, 1};
    const cocclCompressorConfigContext context = {
        cocclCompressorConfigDefault, 2, 8};
    char error[160] = {};
    if (plugin.descriptor->parseConfig(
            &values, &context, &config, error, sizeof(error)) !=
        ncclInvalidArgument) {
      fprintf(stderr, "dietgpu: invalid probBits was accepted\n");
      result = 1;
    }
  }
  if (config != nullptr) plugin.descriptor->destroyConfig(config);
  dlclose(plugin.library);
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    fprintf(stderr, "Usage: %s <compressor-library-directory>\n", argv[0]);
    return 1;
  }
  if (testQuantizedPlugin(argv[1], "sdp4bit", false) ||
      testQuantizedPlugin(argv[1], "tahquant", true) ||
      testTaco(argv[1]) || testZfp(argv[1]) ||
      testDietGpu(argv[1])) {
    return 1;
  }
  printf("COCCL compressor size-bound tests passed\n");
  return 0;
}
