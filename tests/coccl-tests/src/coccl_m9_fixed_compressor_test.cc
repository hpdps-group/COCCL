#include "compressor_plugin/detail/coccl_compressor_abi.h"

#include <dlfcn.h>
#include <stdio.h>

#include <initializer_list>
#include <string>
#include <vector>

struct LoadedPlugin {
  void* library = nullptr;
  const cocclCompressorPlugin* descriptor = nullptr;
};

bool loadPlugin(const char* directory, const char* name,
                LoadedPlugin* loaded) {
  const std::string path = std::string(directory) + "/lib" + name + ".so";
  loaded->library = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (loaded->library == nullptr) {
    fprintf(stderr, "failed to load %s: %s\n", path.c_str(), dlerror());
    return false;
  }
  auto entry = reinterpret_cast<cocclGetCompressorPluginFn>(
      dlsym(loaded->library, COCCL_COMPRESSOR_ENTRY_SYMBOL));
  loaded->descriptor = entry == nullptr ? nullptr : entry();
  char error[192] = {};
  if (!cocclValidateCompressorPlugin(name, loaded->descriptor, error,
                                     sizeof(error))) {
    fprintf(stderr, "%s descriptor rejected: %s\n", name, error);
    return false;
  }
  return true;
}

bool hasCapabilities(const LoadedPlugin& plugin, uint64_t capabilities) {
  return (plugin.descriptor->capabilities & capabilities) == capabilities;
}

void* parseConfig(const LoadedPlugin& plugin,
                  std::initializer_list<cocclConfigPair> values,
                  bool expectSuccess) {
  std::vector<cocclConfigPair> storage(values);
  const cocclConfigView view = {storage.data(), storage.size()};
  const cocclCompressorConfigContext context = {
      cocclCompressorConfigDefault, 1, 4};
  char error[192] = {};
  void* config = nullptr;
  const ncclResult_t result = plugin.descriptor->parseConfig(
      &view, &context, &config, error, sizeof(error));
  if ((result == ncclSuccess) != expectSuccess) {
    fprintf(stderr, "%s configuration result %d: %s\n",
            plugin.descriptor->name, (int)result, error);
    if (result == ncclSuccess) plugin.descriptor->destroyConfig(config);
    return reinterpret_cast<void*>(1);
  }
  return config;
}

bool checkBound(const LoadedPlugin& plugin, void* config,
                cocclCompressorOperation operation, size_t elements,
                size_t chunks, ncclDataType_t datatype,
                size_t expectedBytes) {
  const cocclCompressorSizeQuery query = {
      sizeof(cocclCompressorSizeQuery), operation, elements, chunks,
      datatype, config};
  size_t encodedBytes = 0;
  const ncclResult_t result = plugin.descriptor->getEncodedSizeBound(
      &query, &encodedBytes);
  if (result != ncclSuccess || encodedBytes != expectedBytes) {
    fprintf(stderr, "%s size bound result=%d bytes=%zu expected=%zu\n",
            plugin.descriptor->name, (int)result, encodedBytes,
            expectedBytes);
    return false;
  }
  return true;
}

bool checkUnsupportedBound(const LoadedPlugin& plugin, void* config,
                           cocclCompressorOperation operation) {
  const cocclCompressorSizeQuery query = {
      sizeof(cocclCompressorSizeQuery), operation, 1024, 4,
      ncclFloat32, config};
  size_t encodedBytes = 0;
  return plugin.descriptor->getEncodedSizeBound(
             &query, &encodedBytes) == ncclInvalidUsage;
}

bool checkSdp4Bit(const LoadedPlugin& plugin) {
  const uint64_t fused = cocclCompressorCapabilityDecompressReduce |
      cocclCompressorCapabilityDecompressReduceCompress;
  if (!hasCapabilities(plugin, fused)) return false;
  void* config = parseConfig(plugin,
      {{"groupCount", "128"}, {"quantBits", "4"},
       {"quantType", "Symmetric"}}, true);
  if (config == nullptr || config == reinterpret_cast<void*>(1)) return false;
  const bool sizes =
      checkBound(plugin, config, cocclCompressorOperationCompress,
                 1024, 4, ncclFloat32, 544) &&
      checkBound(plugin, config,
                 cocclCompressorOperationDecompressReduceCompress,
                 1024, 4, ncclFloat32, 544) &&
      checkBound(plugin, config, cocclCompressorOperationCompress,
                 1024, 4, ncclFloat16, 576) &&
      checkBound(plugin, config, cocclCompressorOperationCompress,
                 1024, 4, ncclBfloat16, 576);
  plugin.descriptor->destroyConfig(config);
  if (!sizes) return false;
  void* separate = parseConfig(plugin,
      {{"groupCount", "128"}, {"quantBits", "4"},
       {"inQuantBits", "8"}, {"outQuantBits", "4"},
       {"inGroupCount", "128"}, {"outGroupCount", "256"},
       {"quantType", "Symmetric"}}, true);
  if (separate == nullptr || separate == reinterpret_cast<void*>(1)) {
    return false;
  }
  const bool separateSizes =
      checkBound(plugin, separate, cocclCompressorOperationCompress,
                 1024, 4, ncclFloat32, 1056) &&
      checkBound(plugin, separate,
                 cocclCompressorOperationDecompressReduceCompress,
                 1024, 4, ncclFloat32, 528);
  plugin.descriptor->destroyConfig(separate);
  if (!separateSizes) return false;
  return parseConfig(plugin,
      {{"groupCount", "128"}, {"quantBits", "2"}}, false) == nullptr &&
      parseConfig(plugin,
      {{"groupCount", "127"}, {"quantBits", "4"}}, false) == nullptr;
}

bool checkTahQuant(const LoadedPlugin& plugin) {
  const uint64_t fused = cocclCompressorCapabilityDecompressReduce |
      cocclCompressorCapabilityDecompressReduceCompress;
  if (!hasCapabilities(plugin, fused)) return false;
  void* config = parseConfig(plugin,
      {{"groupCount", "128"}, {"quantBits", "4"},
       {"quantType", "Symmetric"}}, true);
  if (config == nullptr || config == reinterpret_cast<void*>(1)) return false;
  const bool size =
      checkBound(plugin, config, cocclCompressorOperationCompress,
                 1024, 4, ncclFloat32, 544) &&
      checkBound(plugin, config,
                 cocclCompressorOperationDecompressReduceCompress,
                 1024, 4, ncclFloat32, 544);
  plugin.descriptor->destroyConfig(config);
  if (!size) return false;

  void* pivot = parseConfig(plugin,
      {{"groupCount", "128"}, {"quantBits", "4"},
       {"quantType", "Symmetric"}, {"hadamard", "true"},
       {"pivotSwap", "true"}}, true);
  if (pivot == nullptr || pivot == reinterpret_cast<void*>(1)) return false;
  const bool pivotSize = checkBound(plugin, pivot,
      cocclCompressorOperationCompress, 1024, 4, ncclFloat32, 616);
  plugin.descriptor->destroyConfig(pivot);
  if (!pivotSize) return false;
  return parseConfig(plugin,
      {{"groupCount", "128"}, {"quantBits", "2"}}, false) == nullptr &&
      parseConfig(plugin,
      {{"groupCount", "128"}, {"quantBits", "4"},
       {"pivotSwap", "true"}}, false) == nullptr;
}

bool checkTaco(const LoadedPlugin& plugin) {
  if (plugin.descriptor->capabilities !=
      COCCL_COMPRESSOR_REQUIRED_CAPABILITIES) return false;
  void* config = parseConfig(plugin,
      {{"groupSize", "128"}, {"fp8Format", "E4M3"}}, true);
  if (config == nullptr || config == reinterpret_cast<void*>(1)) return false;
  const bool size =
      checkBound(plugin, config, cocclCompressorOperationCompress,
                 1024, 4, ncclFloat32, 1088) &&
      checkUnsupportedBound(
          plugin, config,
          cocclCompressorOperationDecompressReduceCompress);
  plugin.descriptor->destroyConfig(config);
  return size && parseConfig(plugin,
      {{"groupSize", "96"}}, false) == nullptr;
}

bool checkZfp(const LoadedPlugin& plugin) {
  if (plugin.descriptor->capabilities !=
      COCCL_COMPRESSOR_REQUIRED_CAPABILITIES) return false;
  void* config = parseConfig(plugin, {{"rate", "4"}}, true);
  if (config == nullptr || config == reinterpret_cast<void*>(1)) return false;
  const cocclCompressorSizeQuery query = {
      sizeof(cocclCompressorSizeQuery), cocclCompressorOperationCompress,
      1024, 4, ncclFloat32, config};
  size_t encodedBytes = 0;
  const bool valid = plugin.descriptor->getEncodedSizeBound(
      &query, &encodedBytes) == ncclSuccess &&
      encodedBytes != 0 && encodedBytes <= 1024 * sizeof(float) &&
      checkUnsupportedBound(
          plugin, config,
          cocclCompressorOperationDecompressReduceCompress);
  plugin.descriptor->destroyConfig(config);
  return valid && parseConfig(plugin, {{"rate", "65"}}, false) == nullptr;
}

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  const char* names[] = {"sdp4bit", "tahquant", "taco", "zfp"};
  LoadedPlugin plugins[4];
  for (int index = 0; index < 4; ++index) {
    if (!loadPlugin(argv[1], names[index], &plugins[index])) return 1;
  }
  const bool passed = checkSdp4Bit(plugins[0]) &&
      checkTahQuant(plugins[1]) && checkTaco(plugins[2]) &&
      checkZfp(plugins[3]);
  for (LoadedPlugin& plugin : plugins) dlclose(plugin.library);
  if (!passed) return 1;
  printf("COCCL M9 fixed compressor tests passed\n");
  return 0;
}
