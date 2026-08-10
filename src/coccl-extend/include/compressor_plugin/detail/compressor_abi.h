#ifndef COCCL_DETAIL_COMPRESSOR_ABI_H_
#define COCCL_DETAIL_COMPRESSOR_ABI_H_

#include "nccl.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

// This is the only contract crossing the compressor shared-library boundary.
// Keep it POD-only: the public C++ SDK adapts user code to these structures.
constexpr uint32_t COCCL_COMPRESSOR_ABI_VERSION = 5;
constexpr uint32_t COCCL_COMPRESSOR_HOST_API_VERSION = 3;
constexpr const char* COCCL_COMPRESSOR_ENTRY_SYMBOL =
    "cocclGetCompressorPlugin";

enum cocclCompressorOperation : uint32_t {
  cocclCompressorOperationCompress = 0,
  cocclCompressorOperationDecompress,
  cocclCompressorOperationDecompressReduce,
  cocclCompressorOperationDecompressReduceCompress,
};

// Encoded byte streams use ncclInt8. The runtime reserves ncclUint8 to mark
// an uncompressed byte-for-byte copy selected when encoding would expand.
constexpr ncclDataType_t COCCL_COMPRESSOR_RAW_PASSTHROUGH = ncclUint8;

enum cocclCompressorCapability : uint64_t {
  cocclCompressorCapabilityCompress = 1ULL << 0,
  cocclCompressorCapabilityDecompress = 1ULL << 1,
  cocclCompressorCapabilityDecompressReduce = 1ULL << 2,
  cocclCompressorCapabilityDecompressReduceCompress = 1ULL << 3,
};

constexpr uint64_t COCCL_COMPRESSOR_REQUIRED_CAPABILITIES =
    cocclCompressorCapabilityCompress |
    cocclCompressorCapabilityDecompress;

enum cocclCompressorConfigVariant : uint32_t {
  cocclCompressorConfigDefault = 0,
  cocclCompressorConfigHierarchical = 1,
};

struct cocclConfigPair {
  const char* key;
  const char* value;
};

struct cocclConfigView {
  const cocclConfigPair* pairs;
  size_t count;
};

struct cocclCompressorConfigContext {
  cocclCompressorConfigVariant variant;
  int nodes;
  int devicesPerNode;
};

struct cocclCompressorDataView {
  const void* data;
  size_t bytes;
  size_t elements;
  size_t chunks;
  ncclDataType_t datatype;
};

struct cocclCompressorOutputView {
  void* data;
  size_t capacityBytes;
  size_t bytes;
  size_t elements;
  size_t chunks;
  ncclDataType_t datatype;
};

struct cocclCompressorBufferView {
  void* data;
  size_t bytes;
};

using cocclCompressorCreateStateFn = ncclResult_t (*)(void** state);
using cocclCompressorDestroyStateFn = void (*)(void* state);

struct cocclCompressorHostApi {
  uint32_t abiVersion;
  uint32_t structSize;
  ncclResult_t (*allocateScratch)(void* context, size_t bytes,
                                  cocclCompressorBufferView* buffer);
  ncclResult_t (*acquirePersistent)(void* context, size_t slot, size_t bytes,
                                    cocclCompressorBufferView* buffer);
  ncclResult_t (*getOrCreateState)(
      void* context, const void* typeKey,
      cocclCompressorCreateStateFn createState,
      cocclCompressorDestroyStateFn destroyState, void** state);
};

struct cocclCompressorExecutionContext {
  uint32_t structSize;
  const cocclCompressorHostApi* hostApi;
  void* hostContext;
  cudaStream_t stream;
  int cudaDev;
  int rank;
  int nRanks;
  int nodes;
  int devicesPerNode;
};

struct cocclCompressorCall {
  uint32_t structSize;
  cocclCompressorOperation operation;
  cocclCompressorDataView input;
  cocclCompressorOutputView* output;
  int rank;
  size_t reduceChunks;
  ncclDataType_t originalDatatype;
  size_t originalElements;
  const void* config;
  cocclCompressorExecutionContext* execution;
};

struct cocclCompressorPlugin {
  uint32_t abiVersion;
  uint32_t structSize;
  const char* name;
  uint64_t capabilities;
  ncclResult_t (*execute)(cocclCompressorCall* call);
  ncclResult_t (*parseConfig)(
      const cocclConfigView* values,
      const cocclCompressorConfigContext* context, void** config,
      char* error, size_t errorCapacity);
  void (*destroyConfig)(void* config);
};

using cocclGetCompressorPluginFn = const cocclCompressorPlugin* (*)();

inline bool cocclValidateCompressorPlugin(
    const char* expectedName, const cocclCompressorPlugin* plugin,
    char* error, size_t errorCapacity) {
  const char* message = nullptr;
  if (plugin == nullptr) {
    message = "entry returned a null descriptor";
  } else if (plugin->abiVersion != COCCL_COMPRESSOR_ABI_VERSION) {
    if (error != nullptr && errorCapacity != 0) {
      snprintf(error, errorCapacity,
               "compressor ABI v%u is unsupported; runtime requires v%u",
               plugin->abiVersion, COCCL_COMPRESSOR_ABI_VERSION);
    }
    return false;
  } else if (plugin->structSize != sizeof(cocclCompressorPlugin)) {
    message = "compressor descriptor size mismatch";
  } else if (plugin->name == nullptr || expectedName == nullptr ||
             strcmp(plugin->name, expectedName) != 0) {
    message = "compressor descriptor name mismatch";
  } else if ((plugin->capabilities &
              COCCL_COMPRESSOR_REQUIRED_CAPABILITIES) !=
             COCCL_COMPRESSOR_REQUIRED_CAPABILITIES) {
    message = "compressor is missing compress or decompress capability";
  } else if (plugin->execute == nullptr || plugin->parseConfig == nullptr ||
             plugin->destroyConfig == nullptr) {
    message = "compressor descriptor is missing a required callback";
  }
  if (message == nullptr) return true;
  if (error != nullptr && errorCapacity != 0) {
    snprintf(error, errorCapacity, "%s", message);
  }
  return false;
}

#endif
