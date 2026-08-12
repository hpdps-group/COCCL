#include "compressor_plugin/coccl_compressor_plugin.h"

#include <stdio.h>
#include <string.h>

extern "C" const cocclCompressorPlugin* cocclGetCompressorPlugin();

namespace {

struct MinimalCompressor {
  static coccl::Status compress(const coccl::Input& input,
                                coccl::Output& output,
                                coccl::Context&) {
    return output.commitBytes(input.bytes(), input.chunks());
  }

  static coccl::Status decompress(const coccl::Input&, coccl::Output&,
                                  coccl::Context&) {
    return ncclSuccess;
  }
};

struct ResourceCompressor {
  struct Config {
    int bits = 4;
    bool useResources = false;
  };

  struct State {
    int calls = 0;
  };

  static coccl::Status configure(coccl::ConfigReader& reader, Config& config,
                                 const coccl::ConfigContext&) {
    return reader.get("bits", config.bits, 1, 8)
        .get("useResources", config.useResources)
        .finish();
  }

  static coccl::Status compress(const coccl::Input& input,
                                coccl::Output& output,
                                coccl::Context& context) {
    if (context.config<Config>().useResources) {
      coccl::Buffer scratch;
      coccl::Buffer persistent;
      State* state = nullptr;
      if (context.scratch(64, &scratch) != ncclSuccess ||
          context.persistent(3, 128, &persistent) != ncclSuccess ||
          context.instance(&state) != ncclSuccess || !scratch ||
          !persistent || state == nullptr) {
        return ncclInvalidUsage;
      }
      ++state->calls;
    }
    return output.commitBytes(input.bytes(), input.chunks());
  }

  static coccl::Status decompress(const coccl::Input&, coccl::Output&,
                                  coccl::Context&) {
    return ncclSuccess;
  }

  static coccl::Status decompressReduce(const coccl::Input&,
                                        coccl::Output&,
                                        coccl::Context&) {
    return ncclSuccess;
  }

  static coccl::Status encodedSizeBound(
      const coccl::Shape& input, size_t* encodedBytes,
      const coccl::SizeContext& context) {
    if (context.operation() != cocclCompressorOperationCompress) {
      return ncclInvalidUsage;
    }
    if (encodedBytes == nullptr || context.config<Config>().bits != 4) {
      return ncclInvalidArgument;
    }
    *encodedBytes = input.bytes() / 2;
    return ncclSuccess;
  }
};

struct ZeroEstimateCompressor {
  static coccl::Status compress(const coccl::Input& input,
                                coccl::Output& output,
                                coccl::Context&) {
    return output.commitBytes(input.bytes(), input.chunks());
  }

  static coccl::Status decompress(const coccl::Input&, coccl::Output&,
                                  coccl::Context&) {
    return ncclSuccess;
  }

  static coccl::Status encodedSizeBound(
      const coccl::Shape&, size_t* encodedBytes,
      const coccl::SizeContext&) {
    *encodedBytes = 0;
    return ncclSuccess;
  }
};

struct FramedCompressor {
  static constexpr bool kFramed = true;

  static coccl::Status compress(const coccl::Input&, coccl::Output& output,
                                coccl::Context&) {
    return output.commitFrames();
  }

  static coccl::Status decompress(const coccl::Input&, coccl::Output&,
                                  coccl::Context&) {
    return ncclSuccess;
  }
};

struct FakeHost {
  unsigned char scratch[64] = {};
  unsigned char persistent[128] = {};
  int scratchCalls = 0;
  int persistentCalls = 0;
  int stateCalls = 0;
  const void* stateType = nullptr;
  void* state = nullptr;
  cocclCompressorDestroyStateFn destroyState = nullptr;
};

ncclResult_t allocateScratch(void* opaque, size_t bytes,
                             cocclCompressorBufferView* buffer) {
  auto* host = static_cast<FakeHost*>(opaque);
  if (host == nullptr || buffer == nullptr || bytes > sizeof(host->scratch)) {
    return ncclInvalidArgument;
  }
  ++host->scratchCalls;
  *buffer = {host->scratch, sizeof(host->scratch)};
  return ncclSuccess;
}

ncclResult_t acquirePersistent(void* opaque, size_t slot, size_t bytes,
                               cocclCompressorBufferView* buffer) {
  auto* host = static_cast<FakeHost*>(opaque);
  if (host == nullptr || buffer == nullptr || slot != 3 ||
      bytes > sizeof(host->persistent)) {
    return ncclInvalidArgument;
  }
  ++host->persistentCalls;
  *buffer = {host->persistent, sizeof(host->persistent)};
  return ncclSuccess;
}

ncclResult_t getOrCreateState(void* opaque, const void* type,
                              cocclCompressorCreateStateFn create,
                              cocclCompressorDestroyStateFn destroy,
                              void** state) {
  auto* host = static_cast<FakeHost*>(opaque);
  if (host == nullptr || type == nullptr || create == nullptr ||
      destroy == nullptr || state == nullptr) {
    return ncclInvalidArgument;
  }
  if (host->state == nullptr) {
    ncclResult_t result = create(&host->state);
    if (result != ncclSuccess) return result;
    host->stateType = type;
    host->destroyState = destroy;
  } else if (host->stateType != type || host->destroyState != destroy) {
    return ncclInvalidUsage;
  }
  ++host->stateCalls;
  *state = host->state;
  return ncclSuccess;
}

const cocclCompressorHostApi kHostApi = {
    COCCL_COMPRESSOR_HOST_API_VERSION,
    sizeof(cocclCompressorHostApi),
    allocateScratch,
    acquirePersistent,
    getOrCreateState,
};

int testMinimalPlugin() {
  const cocclCompressorPlugin* plugin = cocclGetCompressorPlugin();
  char error[128] = {};
  if (!cocclValidateCompressorPlugin("minimal", plugin, error,
                                     sizeof(error)) ||
      plugin->capabilities != COCCL_COMPRESSOR_REQUIRED_CAPABILITIES) {
    fprintf(stderr, "minimal SDK plugin validation failed: %s\n", error);
    return 1;
  }

  cocclCompressorPlugin incompatible = *plugin;
  const uint32_t legacyVersions[] = {5u, 6u};
  for (uint32_t legacyVersion : legacyVersions) {
    incompatible = *plugin;
    incompatible.abiVersion = legacyVersion;
    char expected[16] = {};
    snprintf(expected, sizeof(expected), "ABI v%u", legacyVersion);
    if (cocclValidateCompressorPlugin("minimal", &incompatible, error,
                                      sizeof(error)) ||
        strstr(error, expected) == nullptr) {
      fprintf(stderr, "ABI v%u mismatch was not rejected\n",
              legacyVersion);
      return 1;
    }
  }

  incompatible = *plugin;
  incompatible.structSize--;
  if (cocclValidateCompressorPlugin("minimal", &incompatible, error,
                                    sizeof(error))) {
    fprintf(stderr, "descriptor size mismatch was not rejected\n");
    return 1;
  }
  incompatible = *plugin;
  incompatible.name = "different";
  if (cocclValidateCompressorPlugin("minimal", &incompatible, error,
                                    sizeof(error))) {
    fprintf(stderr, "descriptor name mismatch was not rejected\n");
    return 1;
  }
  incompatible = *plugin;
  incompatible.capabilities &= ~cocclCompressorCapabilityDecompress;
  if (cocclValidateCompressorPlugin("minimal", &incompatible, error,
                                    sizeof(error))) {
    fprintf(stderr, "missing required capability was not rejected\n");
    return 1;
  }
  incompatible = *plugin;
  incompatible.execute = nullptr;
  if (cocclValidateCompressorPlugin("minimal", &incompatible, error,
                                    sizeof(error))) {
    fprintf(stderr, "missing required callback was not rejected\n");
    return 1;
  }

  const cocclCompressorSizeQuery query = {
      sizeof(cocclCompressorSizeQuery), cocclCompressorOperationCompress,
      32, 1, ncclInt8, nullptr};
  size_t encodedBytes = 123;
  if (plugin->getEncodedSizeBound == nullptr ||
      plugin->getEncodedSizeBound(&query, &encodedBytes) !=
          ncclInvalidUsage ||
      encodedBytes != 0) {
    fprintf(stderr, "missing estimator did not report unavailable\n");
    return 1;
  }
  return 0;
}

int testFramedPlugin() {
  const cocclCompressorPlugin* plugin =
      coccl::detail::PluginAdapter<FramedCompressor>::descriptor("framed");
  if ((plugin->capabilities & cocclCompressorCapabilityFramed) == 0 ||
      sizeof(cocclCompressorFrameMetadata) != 16) {
    fprintf(stderr, "framed capability or metadata layout is invalid\n");
    return 1;
  }

  cocclCompressorExecutionContext execution = {
      sizeof(cocclCompressorExecutionContext), nullptr, nullptr,
      nullptr, 0, 0, 1, 1, 1};
  unsigned char raw[32] = {};
  unsigned char encoded[32] = {};
  cocclCompressorFrameMetadata metadata[2] = {};
  cocclCompressorOutputView compressed = {
      encoded, sizeof(encoded), 0, 0, 2, ncclInt8, metadata, 16};
  cocclCompressorCall compressCall = {
      sizeof(cocclCompressorCall),
      cocclCompressorOperationCompress,
      {raw, sizeof(raw), 8, 2, ncclFloat32, nullptr, 0},
      &compressed,
      0,
      0,
      ncclFloat32,
      8,
      nullptr,
      &execution,
  };
  if (plugin->execute(&compressCall) != ncclSuccess ||
      compressed.bytes != sizeof(encoded) ||
      compressed.elements != sizeof(encoded) ||
      compressed.datatype != ncclInt8 || compressed.chunks != 2) {
    fprintf(stderr, "framed output commit failed\n");
    return 1;
  }

  cocclCompressorOutputView decompressed = {
      raw, sizeof(raw), 0, 8, 2, ncclFloat32, nullptr, 0};
  cocclCompressorCall decompressCall = {
      sizeof(cocclCompressorCall),
      cocclCompressorOperationDecompress,
      {encoded, sizeof(encoded), sizeof(encoded), 2, ncclInt8,
       metadata, 16},
      &decompressed,
      0,
      0,
      ncclFloat32,
      8,
      nullptr,
      &execution,
  };
  if (plugin->execute(&decompressCall) != ncclSuccess ||
      decompressed.bytes != sizeof(raw)) {
    fprintf(stderr, "framed input dispatch failed\n");
    return 1;
  }

  compressCall.output->frameMetadata = nullptr;
  if (plugin->execute(&compressCall) != ncclInvalidArgument) {
    fprintf(stderr, "missing framed output metadata was accepted\n");
    return 1;
  }
  return 0;
}

int testPassthroughMetadata() {
  float inputData[8] = {};
  unsigned char outputData[sizeof(inputData)] = {};
  const cocclCompressorDataView inputView = {
      inputData, sizeof(inputData), 8, 2, ncclFloat32};
  coccl::Input input(inputView);
  if (coccl::shouldPassthrough(input, sizeof(inputData)) ||
      !coccl::shouldPassthrough(input, sizeof(inputData) + 1)) {
    fprintf(stderr, "passthrough size decision is not strict\n");
    return 1;
  }

  cocclCompressorOutputView outputView = {
      outputData, sizeof(outputData), 0, 0, 2, ncclInt8};
  coccl::Output output(&outputView);
  if (output.commitPassthrough(input) != ncclSuccess ||
      outputView.bytes != sizeof(inputData) ||
      outputView.elements != sizeof(inputData) || outputView.chunks != 2 ||
      outputView.datatype != COCCL_COMPRESSOR_RAW_PASSTHROUGH) {
    fprintf(stderr, "passthrough metadata commit failed\n");
    return 1;
  }
  const cocclCompressorDataView passthroughView = {
      outputView.data, outputView.bytes, outputView.elements,
      outputView.chunks, outputView.datatype};
  if (!coccl::isPassthrough(coccl::Input(passthroughView))) {
    fprintf(stderr, "passthrough metadata was not recognized\n");
    return 1;
  }
  return 0;
}

int testOptionalResources() {
  const cocclCompressorPlugin* plugin =
      coccl::detail::PluginAdapter<ResourceCompressor>::descriptor("resource");
  if ((plugin->capabilities &
       cocclCompressorCapabilityDecompressReduce) == 0) {
    fprintf(stderr, "optional capability was not detected\n");
    return 1;
  }

  const cocclConfigPair pairs[] = {
      {"bits", "4"}, {"useResources", "true"}};
  const cocclConfigView values = {pairs, 2};
  const cocclCompressorConfigContext configContext = {
      cocclCompressorConfigDefault, 2, 8};
  char error[128] = {};
  void* config = nullptr;
  if (plugin->parseConfig(&values, &configContext, &config, error,
                          sizeof(error)) != ncclSuccess ||
      config == nullptr) {
    fprintf(stderr, "typed SDK config failed: %s\n", error);
    return 1;
  }
  cocclCompressorSizeQuery sizeQuery = {
      sizeof(cocclCompressorSizeQuery), cocclCompressorOperationCompress,
      32, 1, ncclInt8, config};
  size_t encodedBytes = 0;
  if (plugin->getEncodedSizeBound == nullptr ||
      plugin->getEncodedSizeBound(&sizeQuery, &encodedBytes) != ncclSuccess ||
      encodedBytes != 16 || coccl::Shape(&sizeQuery).bytes() != 32) {
    fprintf(stderr, "SDK encoded-size estimator dispatch failed\n");
    plugin->destroyConfig(config);
    return 1;
  }
  sizeQuery.structSize--;
  if (plugin->getEncodedSizeBound(&sizeQuery, &encodedBytes) !=
      ncclInvalidArgument) {
    fprintf(stderr, "malformed size query was not rejected\n");
    plugin->destroyConfig(config);
    return 1;
  }
  sizeQuery.structSize = sizeof(cocclCompressorSizeQuery);
  cocclCompressorSizeQuery invalidShape = sizeQuery;
  invalidShape.chunks = 3;
  if (plugin->getEncodedSizeBound(&invalidShape, &encodedBytes) !=
      ncclInvalidArgument) {
    fprintf(stderr, "non-divisible size query was not rejected\n");
    plugin->destroyConfig(config);
    return 1;
  }
  invalidShape = sizeQuery;
  invalidShape.elements = SIZE_MAX;
  invalidShape.datatype = ncclFloat64;
  if (plugin->getEncodedSizeBound(&invalidShape, &encodedBytes) !=
      ncclInvalidArgument) {
    fprintf(stderr, "overflowing size query was not rejected\n");
    plugin->destroyConfig(config);
    return 1;
  }

  FakeHost host;
  cocclCompressorExecutionContext execution = {
      sizeof(cocclCompressorExecutionContext), &kHostApi, &host,
      nullptr, 0, 0, 1, 1, 1};
  unsigned char inputData[32] = {};
  unsigned char outputData[32] = {};
  cocclCompressorOutputView output = {
      outputData, sizeof(outputData), 0, 0, 1, ncclInt8};
  cocclCompressorCall call = {
      sizeof(cocclCompressorCall),
      cocclCompressorOperationCompress,
      {inputData, sizeof(inputData), sizeof(inputData), 1, ncclInt8},
      &output,
      0,
      0,
      ncclInt8,
      sizeof(inputData),
      config,
      &execution,
  };
  const ncclResult_t result = plugin->execute(&call);
  plugin->destroyConfig(config);
  if (host.destroyState != nullptr) host.destroyState(host.state);
  host.state = nullptr;
  if (result != ncclSuccess || output.bytes != sizeof(inputData) ||
      host.scratchCalls != 1 || host.persistentCalls != 1 ||
      host.stateCalls != 1) {
    fprintf(stderr, "lazy SDK resource dispatch failed\n");
    return 1;
  }

  const cocclConfigView empty = {nullptr, 0};
  config = nullptr;
  if (plugin->parseConfig(&empty, &configContext, &config, error,
                          sizeof(error)) != ncclSuccess) {
    fprintf(stderr, "default SDK config failed: %s\n", error);
    return 1;
  }
  output = {outputData, sizeof(outputData), 0, 0, 1, ncclInt8};
  call.output = &output;
  call.config = config;
  const ncclResult_t noResourceResult = plugin->execute(&call);
  plugin->destroyConfig(config);
  if (noResourceResult != ncclSuccess || host.scratchCalls != 1 ||
      host.persistentCalls != 1 || host.stateCalls != 1) {
    fprintf(stderr, "unused SDK resources were allocated\n");
    return 1;
  }

  const cocclCompressorPlugin* zeroPlugin =
      coccl::detail::PluginAdapter<ZeroEstimateCompressor>::descriptor(
          "zero");
  sizeQuery.config = nullptr;
  encodedBytes = 1;
  if (zeroPlugin->getEncodedSizeBound(&sizeQuery, &encodedBytes) !=
          ncclInvalidArgument ||
      encodedBytes != 0) {
    fprintf(stderr, "zero encoded-size bound was not rejected\n");
    return 1;
  }

  const cocclConfigPair unknownPair[] = {{"unknown", "1"}};
  const cocclConfigView unknown = {unknownPair, 1};
  config = nullptr;
  if (plugin->parseConfig(&unknown, &configContext, &config, error,
                          sizeof(error)) != ncclInvalidArgument ||
      strstr(error, "unknown") == nullptr) {
    fprintf(stderr, "unknown SDK config key was not rejected\n");
    return 1;
  }
  return 0;
}

}  // namespace

COCCL_REGISTER_COMPRESSOR("minimal", MinimalCompressor);

int main() {
  if (testMinimalPlugin() || testPassthroughMetadata() ||
      testOptionalResources() || testFramedPlugin()) {
    return 1;
  }
  printf("COCCL compressor SDK tests passed\n");
  return 0;
}
