#include "compressor_plugin/detail/coccl_compressor_abi.h"

#include <cuda_runtime.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

namespace {

constexpr size_t kGuardBytes = 64;
constexpr unsigned char kEncodedGuard = 0xa5;
constexpr unsigned char kDecodedGuard = 0x5a;

enum class InputPattern {
  Mixed,
  Compressible,
  Random,
  Ratio130,
};

enum class FrameExpectation {
  Any,
  Mixed,
  RawOnly,
};

struct ScratchAllocations {
  std::vector<void*> buffers;
};

ncclResult_t allocateScratch(void* context, size_t bytes,
                             cocclCompressorBufferView* buffer) {
  if (context == nullptr || bytes == 0 || buffer == nullptr) {
    return ncclInvalidArgument;
  }
  void* storage = nullptr;
  if (cudaMalloc(&storage, bytes) != cudaSuccess) {
    return ncclUnhandledCudaError;
  }
  static_cast<ScratchAllocations*>(context)->buffers.push_back(storage);
  *buffer = {storage, bytes};
  return ncclSuccess;
}

ncclResult_t unusedPersistent(
    void*, size_t, size_t, cocclCompressorBufferView*) {
  return ncclInvalidUsage;
}

ncclResult_t unusedState(
    void*, const void*, cocclCompressorCreateStateFn,
    cocclCompressorDestroyStateFn, void**) {
  return ncclInvalidUsage;
}

void releaseScratch(ScratchAllocations* scratch) {
  for (void* buffer : scratch->buffers) cudaFree(buffer);
  scratch->buffers.clear();
}

uint32_t nextRandom(uint32_t* state) {
  uint32_t value = *state;
  value ^= value << 13;
  value ^= value >> 17;
  value ^= value << 5;
  *state = value;
  return value;
}

bool guardsEqual(const std::vector<unsigned char>& bytes,
                 unsigned char expected) {
  for (unsigned char value : bytes) {
    if (value != expected) return false;
  }
  return true;
}

size_t datatypeBytes(ncclDataType_t datatype) {
  switch (datatype) {
    case ncclFloat16:
    case ncclBfloat16:
      return 2;
    case ncclFloat32:
      return 4;
    default:
      return 1;
  }
}

const char* datatypeName(ncclDataType_t datatype) {
  if (datatype == ncclFloat16) return "fp16";
  if (datatype == ncclBfloat16) return "bf16";
  if (datatype == ncclFloat32) return "fp32";
  return "uint8";
}

int runRoundTrip(const cocclCompressorPlugin* plugin, int probBits,
                 size_t frameBytes, size_t frames, InputPattern pattern,
                 FrameExpectation expectation, bool printProbe,
                 ncclDataType_t datatype = ncclUint8) {
  size_t rawBytes = frameBytes * frames;
  const size_t typeBytes = datatypeBytes(datatype);
  if (frameBytes == 0 || frames == 0 || rawBytes / frames != frameBytes ||
      frameBytes % typeBytes != 0) {
    return 1;
  }
  const size_t rawElements = rawBytes / typeBytes;

  std::vector<unsigned char> input(rawBytes, 0);
  uint32_t randomState = 0x6d2b79f5u;
  for (size_t frame = 0; frame < frames; ++frame) {
    if (pattern == InputPattern::Compressible ||
        (pattern == InputPattern::Mixed && frame % 2 == 0)) {
      continue;
    }
    for (size_t offset = 0; offset < frameBytes; ++offset) {
      const uint32_t value = nextRandom(&randomState);
      input[frame * frameBytes + offset] = static_cast<unsigned char>(
          pattern == InputPattern::Ratio130 ? value & 0x3f : value);
    }
  }

  void* config = nullptr;
  char probBitsText[8] = {};
  snprintf(probBitsText, sizeof(probBitsText), "%d", probBits);
  const cocclConfigPair pair = {"probBits", probBitsText};
  const cocclConfigView configView = {&pair, 1};
  const cocclCompressorConfigContext configContext = {
      cocclCompressorConfigDefault, 1, 1};
  if (plugin->parseConfig(
          &configView, &configContext, &config, nullptr, 0) != ncclSuccess ||
      config == nullptr) {
    fprintf(stderr, "probBits=%d configuration failed\n", probBits);
    return 1;
  }

  unsigned char* deviceInput = nullptr;
  unsigned char* deviceEncodedStorage = nullptr;
  unsigned char* deviceDecodedStorage = nullptr;
  cocclCompressorFrameMetadata* deviceMetadataStorage = nullptr;
  cudaStream_t stream = nullptr;
  cudaEvent_t compressBegin = nullptr;
  cudaEvent_t compressEnd = nullptr;
  cudaEvent_t decompressBegin = nullptr;
  cudaEvent_t decompressEnd = nullptr;
  ScratchAllocations scratch;
  int result = 1;

  if (cudaMalloc(&deviceInput, rawBytes) != cudaSuccess ||
      cudaMalloc(&deviceEncodedStorage, rawBytes + 2 * kGuardBytes) !=
          cudaSuccess ||
      cudaMalloc(&deviceDecodedStorage, rawBytes + 2 * kGuardBytes) !=
          cudaSuccess ||
      cudaMalloc(&deviceMetadataStorage,
                 (frames + 2) * sizeof(cocclCompressorFrameMetadata)) !=
          cudaSuccess ||
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) !=
          cudaSuccess ||
      cudaEventCreate(&compressBegin) != cudaSuccess ||
      cudaEventCreate(&compressEnd) != cudaSuccess ||
      cudaEventCreate(&decompressBegin) != cudaSuccess ||
      cudaEventCreate(&decompressEnd) != cudaSuccess) {
    fprintf(stderr, "CUDA allocation failed\n");
    goto exit;
  }

  if (cudaMemcpyAsync(deviceInput, input.data(), rawBytes,
                      cudaMemcpyHostToDevice, stream) != cudaSuccess ||
      cudaMemsetAsync(deviceEncodedStorage, kEncodedGuard,
                      rawBytes + 2 * kGuardBytes, stream) != cudaSuccess ||
      cudaMemsetAsync(deviceDecodedStorage, kDecodedGuard,
                      rawBytes + 2 * kGuardBytes, stream) != cudaSuccess ||
      cudaMemsetAsync(deviceMetadataStorage, 0xcc,
                      (frames + 2) * sizeof(cocclCompressorFrameMetadata),
                      stream) != cudaSuccess) {
    fprintf(stderr, "CUDA initialization failed\n");
    goto exit;
  }

  {
    cocclCompressorHostApi hostApi = {
        COCCL_COMPRESSOR_HOST_API_VERSION, sizeof(cocclCompressorHostApi),
        allocateScratch, unusedPersistent, unusedState};
    cocclCompressorExecutionContext execution = {
        sizeof(cocclCompressorExecutionContext), &hostApi, &scratch,
        stream, 0, 0, 1, 1, 1};
    cocclCompressorFrameMetadata* deviceMetadata =
        deviceMetadataStorage + 1;
    unsigned char* deviceEncoded = deviceEncodedStorage + kGuardBytes;
    unsigned char* deviceDecoded = deviceDecodedStorage + kGuardBytes;

    const cocclCompressorView raw = {
        deviceInput, rawBytes, rawBytes, rawElements, frames, datatype,
        nullptr, 0};
    cocclCompressorView encoded = {
        deviceEncoded, rawBytes, 0, 0, frames, ncclInt8,
        deviceMetadata, frameBytes};
    cocclCompressorCall compressCall = {
        sizeof(cocclCompressorCall), cocclCompressorOperationCompress,
        raw, &encoded, 0, 0, datatype, rawElements, config, &execution};
    cudaEventRecord(compressBegin, stream);
    const ncclResult_t compressResult = plugin->execute(&compressCall);
    cudaEventRecord(compressEnd, stream);
    if (compressResult != ncclSuccess) {
      fprintf(stderr, "probBits=%d compression failed: %d\n",
              probBits, (int)compressResult);
      goto exit;
    }

    const cocclCompressorView compressed = {
        encoded.data, encoded.bytes, encoded.bytes, encoded.elements,
        encoded.chunks, encoded.datatype, encoded.frameMetadata,
        encoded.frameStrideBytes};
    cocclCompressorView decoded = {
        deviceDecoded, rawBytes, 0, rawElements, frames, datatype,
        nullptr, 0};
    cocclCompressorCall decompressCall = {
        sizeof(cocclCompressorCall), cocclCompressorOperationDecompress,
        compressed, &decoded, 0, 0, datatype, rawElements, config,
        &execution};
    cudaEventRecord(decompressBegin, stream);
    const ncclResult_t decompressResult = plugin->execute(&decompressCall);
    cudaEventRecord(decompressEnd, stream);
    if (decompressResult != ncclSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
      fprintf(stderr, "probBits=%d decompression failed: %d\n",
              probBits, (int)decompressResult);
      goto exit;
    }

    std::vector<unsigned char> decodedHost(rawBytes);
    std::vector<cocclCompressorFrameMetadata> metadata(frames);
    std::vector<unsigned char> encodedPrefix(kGuardBytes);
    std::vector<unsigned char> encodedSuffix(kGuardBytes);
    std::vector<unsigned char> decodedPrefix(kGuardBytes);
    std::vector<unsigned char> decodedSuffix(kGuardBytes);
    cocclCompressorFrameMetadata metadataGuards[2] = {};
    if (cudaMemcpy(decodedHost.data(), deviceDecoded, rawBytes,
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(metadata.data(), deviceMetadata,
                   frames * sizeof(cocclCompressorFrameMetadata),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(encodedPrefix.data(), deviceEncodedStorage, kGuardBytes,
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(encodedSuffix.data(), deviceEncoded + rawBytes,
                   kGuardBytes, cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(decodedPrefix.data(), deviceDecodedStorage, kGuardBytes,
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(decodedSuffix.data(), deviceDecoded + rawBytes,
                   kGuardBytes,
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(metadataGuards, deviceMetadataStorage,
                   sizeof(cocclCompressorFrameMetadata),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(metadataGuards + 1, deviceMetadata + frames,
                   sizeof(cocclCompressorFrameMetadata),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        decodedHost != input ||
        !guardsEqual(encodedPrefix, kEncodedGuard) ||
        !guardsEqual(encodedSuffix, kEncodedGuard) ||
        !guardsEqual(decodedPrefix, kDecodedGuard) ||
        !guardsEqual(decodedSuffix, kDecodedGuard)) {
      fprintf(stderr, "probBits=%d round-trip or payload guard failed\n",
              probBits);
      goto exit;
    }

    cocclCompressorFrameMetadata guardPattern = {};
    memset(&guardPattern, 0xcc, sizeof(guardPattern));
    if (memcmp(metadataGuards, &guardPattern, sizeof(guardPattern)) != 0 ||
        memcmp(metadataGuards + 1, &guardPattern,
               sizeof(guardPattern)) != 0) {
      fprintf(stderr, "probBits=%d metadata guard failed\n", probBits);
      goto exit;
    }

    bool sawEncoded = false;
    bool sawRaw = false;
    uint64_t payloadBytes = 0;
    for (const auto& frame : metadata) {
      if (frame.reserved != 0 || frame.payloadBytes == 0 ||
          frame.payloadBytes > frameBytes) {
        fprintf(stderr, "probBits=%d emitted invalid frame metadata\n",
                probBits);
        goto exit;
      }
      if (frame.encoding == cocclCompressorFrameEncoded) {
        sawEncoded = true;
      } else if (frame.encoding == cocclCompressorFrameRaw &&
                 frame.payloadBytes == frameBytes) {
        sawRaw = true;
      } else {
        fprintf(stderr, "probBits=%d emitted invalid frame encoding\n",
                probBits);
        goto exit;
      }
      payloadBytes += frame.payloadBytes;
    }
    if (expectation == FrameExpectation::Mixed &&
        (!sawEncoded || !sawRaw)) {
      fprintf(stderr, "probBits=%d did not produce mixed ANS/Raw frames\n",
              probBits);
      goto exit;
    }
    if (expectation == FrameExpectation::RawOnly && sawEncoded) {
      fprintf(stderr, "unaligned input did not use Raw frames\n");
      goto exit;
    }
    if (printProbe) {
      float encodeMs = 0.0f;
      float decodeMs = 0.0f;
      cudaEventElapsedTime(&encodeMs, compressBegin, compressEnd);
      cudaEventElapsedTime(&decodeMs, decompressBegin, decompressEnd);
      const char* patternName = pattern == InputPattern::Compressible
          ? "compressible"
          : (pattern == InputPattern::Ratio130 ? "ratio130" : "random");
      printf("COCCL_DIETGPU_PROBE pattern=%s prob_bits=%d raw_bytes=%zu "
             "frames=%zu datatype=%s payload_bytes=%llu ratio=%.9f "
             "encode_us=%.3f decode_us=%.3f\n",
             patternName, probBits, rawBytes, frames,
             datatypeName(datatype),
             (unsigned long long)payloadBytes,
             static_cast<double>(payloadBytes) / rawBytes,
             encodeMs * 1000.0f, decodeMs * 1000.0f);
    }
  }
  result = 0;

exit:
  if (stream != nullptr) cudaStreamSynchronize(stream);
  releaseScratch(&scratch);
  if (decompressEnd != nullptr) cudaEventDestroy(decompressEnd);
  if (decompressBegin != nullptr) cudaEventDestroy(decompressBegin);
  if (compressEnd != nullptr) cudaEventDestroy(compressEnd);
  if (compressBegin != nullptr) cudaEventDestroy(compressBegin);
  if (stream != nullptr) cudaStreamDestroy(stream);
  if (deviceMetadataStorage != nullptr) cudaFree(deviceMetadataStorage);
  if (deviceDecodedStorage != nullptr) cudaFree(deviceDecodedStorage);
  if (deviceEncodedStorage != nullptr) cudaFree(deviceEncodedStorage);
  if (deviceInput != nullptr) cudaFree(deviceInput);
  plugin->destroyConfig(config);
  return result;
}

}  // namespace

ncclDataType_t parseDatatype(const char* name) {
  if (strcmp(name, "fp16") == 0) return ncclFloat16;
  if (strcmp(name, "bf16") == 0) return ncclBfloat16;
  if (strcmp(name, "fp32") == 0) return ncclFloat32;
  return ncclUint8;
}

int main(int argc, char** argv) {
  const bool probe = (argc == 7 || argc == 8) &&
      strcmp(argv[2], "--probe") == 0;
  if (argc != 2 && !probe) {
    fprintf(stderr,
            "usage: %s /path/to/libdietgpu.so "
            "[--probe compressible|random|ratio130 FRAME_BYTES FRAMES "
            "PROB_BITS [uint8|fp16|bf16|fp32]]\n",
            argv[0]);
    return 2;
  }
  void* library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (library == nullptr) {
    fprintf(stderr, "dlopen failed: %s\n", dlerror());
    return 1;
  }
  auto entry = reinterpret_cast<cocclGetCompressorPluginFn>(
      dlsym(library, COCCL_COMPRESSOR_ENTRY_SYMBOL));
  const cocclCompressorPlugin* plugin = entry == nullptr ? nullptr : entry();
  char error[256] = {};
  constexpr uint64_t requiredCapabilities =
      cocclCompressorCapabilityFramed |
      cocclCompressorCapabilityBytewiseLossless;
  if (!cocclValidateCompressorPlugin("dietgpu", plugin,
                                     error, sizeof(error)) ||
      (plugin->capabilities & requiredCapabilities) != requiredCapabilities) {
    fprintf(stderr, "invalid dietGPU plugin: %s\n", error);
    dlclose(library);
    return 1;
  }

  int result = 0;
  if (probe) {
    const InputPattern pattern = strcmp(argv[3], "compressible") == 0
        ? InputPattern::Compressible
        : (strcmp(argv[3], "ratio130") == 0
               ? InputPattern::Ratio130 : InputPattern::Random);
    result = runRoundTrip(
        plugin, atoi(argv[6]), strtoull(argv[4], nullptr, 10),
        strtoull(argv[5], nullptr, 10), pattern, FrameExpectation::Any, true,
        argc == 8 ? parseDatatype(argv[7]) : ncclUint8);
    dlclose(library);
    return result;
  }
  for (int probBits = 9; probBits <= 11 && result == 0; ++probBits) {
    result = runRoundTrip(plugin, probBits, 4096, 4, InputPattern::Mixed,
                          FrameExpectation::Mixed, false);
  }
  const ncclDataType_t floatTypes[] = {
      ncclFloat16, ncclBfloat16, ncclFloat32};
  for (ncclDataType_t datatype : floatTypes) {
    if (result == 0) {
      result = runRoundTrip(
          plugin, 10, 4096, 4, InputPattern::Mixed,
          FrameExpectation::Mixed, false, datatype);
    }
  }
  if (result == 0) {
    result = runRoundTrip(
        plugin, 10, 65540, 3, InputPattern::Mixed,
        FrameExpectation::Any, false, ncclFloat32);
  }
  if (result == 0) {
    result = runRoundTrip(plugin, 10, 4095, 3, InputPattern::Mixed,
                          FrameExpectation::RawOnly, false);
  }
  if (result == 0) {
    result = runRoundTrip(plugin, 10, 4, 65536, InputPattern::Mixed,
                          FrameExpectation::RawOnly, false);
  }
  dlclose(library);
  if (result == 0) printf("COCCL dietGPU codec tests passed\n");
  return result;
}
