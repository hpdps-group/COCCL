#ifndef COCCL_COMPRESSOR_ABI_H_
#define COCCL_COMPRESSOR_ABI_H_

#include "nccl.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

// This is the only contract crossing the compressor shared-library boundary.
// Keep it POD-only: the public C++ SDK adapts user code to these structures.
constexpr uint32_t COCCL_COMPRESSOR_ABI_VERSION = 8;
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
  cocclCompressorCapabilityFramed = 1ULL << 4,
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

enum cocclCompressorFrameEncoding : uint32_t {
  cocclCompressorFrameEncoded = 0,
  cocclCompressorFrameRaw = 1,
};

// One fixed-capacity payload slot is described by its actual wire size and
// whether the receiver should decode or copy it. Framed compressors write this
// metadata on device; byte-oriented Send/Recv also uses it as its wire header.
struct cocclCompressorFrameMetadata {
  uint64_t payloadBytes;
  uint32_t encoding;
  uint32_t reserved;
};

static_assert(sizeof(cocclCompressorFrameMetadata) == 16,
              "compressor frame metadata is a wire contract");

// Input and output share one physical view. For input, capacityBytes equals
// bytes; the C++ Input facade exposes data and frame metadata as read-only.
struct cocclCompressorView {
  void* data;
  size_t capacityBytes;
  size_t bytes;
  size_t elements;
  size_t chunks;
  ncclDataType_t datatype;
  cocclCompressorFrameMetadata* frameMetadata;
  size_t frameStrideBytes;
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
  cocclCompressorView input;
  cocclCompressorView* output;
  int rank;
  size_t reduceChunks;
  ncclDataType_t originalDatatype;
  size_t originalElements;
  const void* config;
  cocclCompressorExecutionContext* execution;
};

// Host-only raw shape query used by workspace planning. For DRC, this is the
// reduced shape that the recompress step will encode. A successful callback
// returns only a guaranteed encoded-size upper bound; it must not launch work
// or acquire execution resources.
struct cocclCompressorSizeQuery {
  uint32_t structSize;
  cocclCompressorOperation operation;
  size_t elements;
  size_t chunks;
  ncclDataType_t datatype;
  const void* config;
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
  ncclResult_t (*getEncodedSizeBound)(
      const cocclCompressorSizeQuery* query, size_t* encodedBytes);
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
