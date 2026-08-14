#include "compressor_plugin/detail/coccl_compressor_abi.h"

#include <cuda_runtime.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <vector>

namespace {

constexpr size_t kGuardBytes = 64;
constexpr unsigned char kEncodedGuard = 0xa5;
constexpr unsigned char kDecodedGuard = 0x5a;

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

int runRoundTrip(const cocclCompressorPlugin* plugin, int probBits,
                 size_t frameBytes, size_t frames, bool expectMixed) {
  size_t rawBytes = frameBytes * frames;
  if (frameBytes == 0 || frames == 0 || rawBytes / frames != frameBytes) {
    return 1;
  }

  std::vector<unsigned char> input(rawBytes, 0);
  uint32_t randomState = 0x6d2b79f5u;
  for (size_t frame = 1; frame < frames; frame += 2) {
    for (size_t offset = 0; offset < frameBytes; ++offset) {
      input[frame * frameBytes + offset] =
          static_cast<unsigned char>(nextRandom(&randomState));
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
          cudaSuccess) {
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
        COCCL_COMPRESSOR_HOST_API_VERSION,
        sizeof(cocclCompressorHostApi), allocateScratch, nullptr, nullptr};
    cocclCompressorExecutionContext execution = {
        sizeof(cocclCompressorExecutionContext), &hostApi, &scratch,
        stream, 0, 0, 1, 1, 1};
    cocclCompressorFrameMetadata* deviceMetadata =
        deviceMetadataStorage + 1;
    unsigned char* deviceEncoded = deviceEncodedStorage + kGuardBytes;
    unsigned char* deviceDecoded = deviceDecodedStorage + kGuardBytes;

    const cocclCompressorView raw = {
        deviceInput, rawBytes, rawBytes, rawBytes, frames, ncclUint8,
        nullptr, 0};
    cocclCompressorView encoded = {
        deviceEncoded, rawBytes, 0, 0, frames, ncclInt8,
        deviceMetadata, frameBytes};
    cocclCompressorCall compressCall = {
        sizeof(cocclCompressorCall), cocclCompressorOperationCompress,
        raw, &encoded, 0, 0, ncclUint8, rawBytes, config, &execution};
    if (plugin->execute(&compressCall) != ncclSuccess) {
      fprintf(stderr, "probBits=%d compression failed\n", probBits);
      goto exit;
    }

    const cocclCompressorView compressed = {
        encoded.data, encoded.bytes, encoded.bytes, encoded.elements,
        encoded.chunks, encoded.datatype, encoded.frameMetadata,
        encoded.frameStrideBytes};
    cocclCompressorView decoded = {
        deviceDecoded, rawBytes, 0, rawBytes, frames, ncclUint8,
        nullptr, 0};
    cocclCompressorCall decompressCall = {
        sizeof(cocclCompressorCall), cocclCompressorOperationDecompress,
        compressed, &decoded, 0, 0, ncclUint8, rawBytes, config,
        &execution};
    if (plugin->execute(&decompressCall) != ncclSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
      fprintf(stderr, "probBits=%d decompression failed\n", probBits);
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
    }
    if (expectMixed && (!sawEncoded || !sawRaw)) {
      fprintf(stderr, "probBits=%d did not produce mixed ANS/Raw frames\n",
              probBits);
      goto exit;
    }
    if (!expectMixed && sawEncoded) {
      fprintf(stderr, "unaligned input did not use Raw frames\n");
      goto exit;
    }
  }
  result = 0;

exit:
  if (stream != nullptr) cudaStreamSynchronize(stream);
  releaseScratch(&scratch);
  if (stream != nullptr) cudaStreamDestroy(stream);
  if (deviceMetadataStorage != nullptr) cudaFree(deviceMetadataStorage);
  if (deviceDecodedStorage != nullptr) cudaFree(deviceDecodedStorage);
  if (deviceEncodedStorage != nullptr) cudaFree(deviceEncodedStorage);
  if (deviceInput != nullptr) cudaFree(deviceInput);
  plugin->destroyConfig(config);
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s /path/to/libdietgpu.so\n", argv[0]);
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
  if (!cocclValidateCompressorPlugin("dietgpu", plugin,
                                     error, sizeof(error)) ||
      (plugin->capabilities & cocclCompressorCapabilityFramed) == 0) {
    fprintf(stderr, "invalid dietGPU plugin: %s\n", error);
    dlclose(library);
    return 1;
  }

  int result = 0;
  for (int probBits = 9; probBits <= 11 && result == 0; ++probBits) {
    result = runRoundTrip(plugin, probBits, 4096, 4, true);
  }
  if (result == 0) {
    result = runRoundTrip(plugin, 10, 4095, 3, false);
  }
  if (result == 0) {
    result = runRoundTrip(plugin, 10, 4, 65536, false);
  }
  dlclose(library);
  if (result == 0) printf("COCCL dietGPU codec tests passed\n");
  return result;
}
