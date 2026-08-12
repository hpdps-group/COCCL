#include "compressor_plugin/coccl_compressor_plugin.h"

#include "dietgpu/ans/BatchPrefixSum.cuh"
#include "dietgpu/ans/GpuANSCodec.h"
#include "dietgpu/ans/GpuANSEncode.cuh"
#include "dietgpu/ans/GpuANSUtils.cuh"
#include "dietgpu/utils/StackDeviceMemory.h"

#include <cuda_runtime.h>
#include <limits.h>
#include <stdint.h>

namespace {

constexpr size_t kScratchAlignment = dietgpu::kSDMAlignment;
constexpr uint32_t kMaxBatchFrames = 65535;

struct DietGpuConfig {
  int probBits = dietgpu::kANSDefaultProbBits;
};

bool alignedBytes(size_t bytes, size_t* aligned) {
  if (aligned == nullptr) return false;
  bytes = bytes == 0 ? 1 : bytes;
  size_t padded = 0;
  if (!coccl::checkedAdd(bytes, kScratchAlignment - 1, &padded)) {
    return false;
  }
  *aligned = padded / kScratchAlignment * kScratchAlignment;
  return *aligned >= kScratchAlignment;
}

bool alignedProduct(size_t count, size_t elementBytes, size_t* bytes) {
  size_t rawBytes = 0;
  return coccl::checkedMultiply(count, elementBytes, &rawBytes) &&
      alignedBytes(rawBytes, bytes);
}

bool addBytes(size_t bytes, size_t* total) {
  return coccl::checkedAdd(*total, bytes, total);
}

bool encodeStackBytes(uint32_t frames, uint32_t frameBytes,
                      size_t* stackBytes) {
  if (frames == 0 || frameBytes == 0 || stackBytes == nullptr) return false;
  const uint32_t blocks =
      dietgpu::divUp(frameBytes, dietgpu::kDefaultBlockSize);
  const size_t uncoalescedStride =
      dietgpu::getMaxBlockSizeUnCoalesced(dietgpu::kDefaultBlockSize);

  size_t table = 0;
  size_t histogram = 0;
  size_t checksum = 0;
  size_t compressedBlocks = 0;
  size_t compressedWords = 0;
  size_t compressedPrefix = 0;
  if (!alignedProduct((size_t)frames * dietgpu::kNumSymbols,
                      sizeof(uint4), &table) ||
      !alignedProduct((size_t)frames * dietgpu::kNumSymbols,
                      sizeof(uint32_t), &histogram) ||
      !alignedProduct(frames, sizeof(uint32_t), &checksum)) {
    return false;
  }
  size_t frameBlocks = 0;
  if (!coccl::checkedMultiply((size_t)frames, (size_t)blocks,
                              &frameBlocks) ||
      !alignedProduct(frameBlocks, uncoalescedStride, &compressedBlocks) ||
      !alignedProduct(frameBlocks, sizeof(uint32_t), &compressedWords) ||
      !alignedProduct(frameBlocks, sizeof(uint32_t), &compressedPrefix)) {
    return false;
  }

  size_t histogramPhase = 0;
  if (!coccl::checkedAdd(table, histogram, &histogramPhase)) return false;
  size_t encodePhase = table;
  if (!addBytes(checksum, &encodePhase) ||
      !addBytes(compressedBlocks, &encodePhase) ||
      !addBytes(compressedWords, &encodePhase) ||
      !addBytes(compressedPrefix, &encodePhase)) {
    return false;
  }
  const size_t prefixTemp =
      dietgpu::getBatchExclusivePrefixSumTempSize(frames, blocks);
  if (prefixTemp != 0) {
    size_t alignedPrefixTemp = 0;
    if (!alignedBytes(prefixTemp, &alignedPrefixTemp) ||
        !addBytes(alignedPrefixTemp, &encodePhase)) {
      return false;
    }
  }
  *stackBytes = histogramPhase > encodePhase
      ? histogramPhase : encodePhase;
  return *stackBytes >= kScratchAlignment;
}

bool decodeStackBytes(uint32_t frames, int probBits, size_t* stackBytes) {
  if (frames == 0 || probBits < 9 || probBits > 11 ||
      stackBytes == nullptr) {
    return false;
  }
  return alignedProduct(
      (size_t)frames * ((size_t)1 << probBits), sizeof(uint32_t),
      stackBytes);
}

bool ansBatchShapeSupported(size_t frames, size_t frameBytes) {
  return frames <= kMaxBatchFrames && frameBytes != 0 &&
      frameBytes <= (size_t)INT32_MAX / 2 &&
      frameBytes % dietgpu::kANSRequiredAlignment == 0;
}

bool ansShapeSupported(const coccl::Input& input, size_t frameBytes) {
  return ansBatchShapeSupported(input.chunks(), frameBytes) &&
      ((uintptr_t)input.data() % dietgpu::kANSRequiredAlignment) == 0;
}

unsigned int rawFrameGrid(size_t frames) {
  return (unsigned int)(frames < kMaxBatchFrames
      ? frames : (size_t)kMaxBatchFrames);
}

__global__ void finalizeFrames(
    const uint8_t* input, size_t inputStride,
    const uint8_t* candidate, size_t candidateStride,
    const uint32_t* candidateBytes, uint8_t* output, size_t outputStride,
    cocclCompressorFrameMetadata* metadata, size_t frameBytes,
    size_t frames, bool useCandidate) {
  for (size_t frame = blockIdx.x; frame < frames; frame += gridDim.x) {
    const uint32_t encodedBytes =
        useCandidate ? candidateBytes[frame] : UINT32_MAX;
    const bool encoded = encodedBytes > 0 && encodedBytes < frameBytes;
    const size_t copyBytes = encoded ? (size_t)encodedBytes : frameBytes;
    const uint8_t* source = encoded
        ? candidate + frame * candidateStride
        : input + frame * inputStride;
    uint8_t* destination = output + frame * outputStride;
    for (size_t offset = threadIdx.x; offset < copyBytes;
         offset += blockDim.x) {
      destination[offset] = source[offset];
    }
    if (threadIdx.x == 0) {
      metadata[frame] = {
          copyBytes,
          encoded ? cocclCompressorFrameEncoded : cocclCompressorFrameRaw,
          0};
    }
  }
}

__global__ void prepareDecodeFrames(
    const uint8_t* input, size_t inputStride,
    const cocclCompressorFrameMetadata* metadata, uint8_t* output,
    size_t outputStride, size_t frameBytes, size_t frames,
    uint8_t* active, bool forceRaw) {
  for (size_t frame = blockIdx.x; frame < frames; frame += gridDim.x) {
    const cocclCompressorFrameMetadata frameMetadata = metadata[frame];
    const bool encoded = !forceRaw &&
        frameMetadata.encoding == cocclCompressorFrameEncoded;
    if (threadIdx.x == 0 && active != nullptr) {
      active[frame] = encoded ? 1 : 0;
    }
    if (!encoded) {
      for (size_t offset = threadIdx.x; offset < frameBytes;
           offset += blockDim.x) {
        output[frame * outputStride + offset] =
            input[frame * inputStride + offset];
      }
    }
  }
}

struct DietGpuCompressor {
  using Config = DietGpuConfig;
  static constexpr bool kFramed = true;

  static coccl::Status configure(coccl::ConfigReader& reader,
                                 Config& config,
                                 const coccl::ConfigContext&) {
    const coccl::Status result =
        reader.get("probBits", config.probBits, 9, 11).finish();
    if (result != ncclSuccess) return result;
    return config.probBits == 9 || config.probBits == 10 ||
            config.probBits == 11
        ? ncclSuccess : ncclInvalidArgument;
  }

  static coccl::Status encodedSizeBound(
      const coccl::Shape& input, size_t* encodedBytes,
      const coccl::SizeContext& context) {
    if (encodedBytes == nullptr ||
        context.operation() != cocclCompressorOperationCompress ||
        input.bytes() == 0) {
      return ncclInvalidUsage;
    }
    *encodedBytes = input.bytes();
    return ncclSuccess;
  }

  static coccl::Status compress(const coccl::Input& input,
                                coccl::Output& output,
                                coccl::Context& context) {
    if (!output.framed() || input.framed() || input.chunks() == 0 ||
        input.bytes() % input.chunks() != 0 ||
        output.chunks() != input.chunks()) {
      return ncclInvalidArgument;
    }
    const size_t frameBytes = input.bytes() / input.chunks();
    if (output.frameStrideBytes() != frameBytes ||
        output.capacityBytes() < input.bytes()) {
      return ncclInvalidArgument;
    }
    const dim3 grid(rawFrameGrid(input.chunks()));
    constexpr int kThreads = 256;
    if (!ansShapeSupported(input, frameBytes)) {
      finalizeFrames<<<grid, kThreads, 0, context.stream()>>>(
          static_cast<const uint8_t*>(input.data()), frameBytes,
          nullptr, 0, nullptr, static_cast<uint8_t*>(output.data()),
          frameBytes, output.frameMetadata(), frameBytes, input.chunks(),
          false);
      if (cudaGetLastError() != cudaSuccess) return ncclUnhandledCudaError;
      return output.commitFrames();
    }

    const uint32_t frames = (uint32_t)input.chunks();
    const uint32_t rawFrameBytes = (uint32_t)frameBytes;
    const uint32_t candidateStride =
        dietgpu::getMaxCompressedSize(rawFrameBytes);
    size_t candidateBytes = 0;
    size_t candidateRegion = 0;
    size_t sizeRegion = 0;
    size_t stackBytes = 0;
    size_t scratchBytes = 0;
    if (!coccl::checkedMultiply(
            (size_t)candidateStride, (size_t)frames, &candidateBytes) ||
        !alignedBytes(candidateBytes, &candidateRegion) ||
        !alignedProduct(frames, sizeof(uint32_t), &sizeRegion) ||
        !encodeStackBytes(frames, rawFrameBytes, &stackBytes) ||
        !coccl::checkedAdd(candidateRegion, sizeRegion, &scratchBytes) ||
        !coccl::checkedAdd(scratchBytes, stackBytes, &scratchBytes)) {
      return ncclInvalidArgument;
    }

    coccl::Buffer scratch;
    coccl::Status result = context.scratch(scratchBytes, &scratch);
    if (result != ncclSuccess) return result;
    char* scratchBase = static_cast<char*>(scratch.data());
    void* candidate = scratchBase;
    uint32_t* sizes = reinterpret_cast<uint32_t*>(
        scratchBase + candidateRegion);
    void* stackBase = scratchBase + candidateRegion + sizeRegion;
    dietgpu::StackDeviceMemory stack(
        context.cudaDevice(), stackBase, stackBytes);
    const DietGpuConfig& config = context.config<DietGpuConfig>();
    dietgpu::ansEncodeBatchStride(
        stack, dietgpu::ANSCodecConfig(config.probBits, false), frames,
        input.data(), rawFrameBytes, rawFrameBytes, nullptr, candidate,
        candidateStride, sizes, context.stream());
    finalizeFrames<<<grid, kThreads, 0, context.stream()>>>(
        static_cast<const uint8_t*>(input.data()), frameBytes,
        static_cast<const uint8_t*>(candidate), candidateStride, sizes,
        static_cast<uint8_t*>(output.data()), frameBytes,
        output.frameMetadata(), frameBytes, input.chunks(), true);
    if (cudaGetLastError() != cudaSuccess) return ncclUnhandledCudaError;
    return output.commitFrames();
  }

  static coccl::Status decompress(const coccl::Input& input,
                                  coccl::Output& output,
                                  coccl::Context& context) {
    if (!input.framed() || output.framed() || input.chunks() == 0 ||
        input.bytes() % input.chunks() != 0 ||
        output.chunks() != input.chunks()) {
      return ncclInvalidArgument;
    }
    const size_t frameBytes = input.frameStrideBytes();
    size_t outputBytes = 0;
    if (!coccl::checkedMultiply(
            output.elements(), coccl::dataTypeSize(output.datatype()),
            &outputBytes) || outputBytes != input.bytes() ||
        frameBytes == 0 || outputBytes > output.capacityBytes() ||
        outputBytes / output.chunks() != frameBytes) {
      return ncclInvalidArgument;
    }
    constexpr int kThreads = 256;
    if (!ansBatchShapeSupported(input.chunks(), frameBytes)) {
      prepareDecodeFrames<<<rawFrameGrid(input.chunks()), kThreads, 0,
                            context.stream()>>>(
          static_cast<const uint8_t*>(input.data()), frameBytes,
          input.frameMetadata(), static_cast<uint8_t*>(output.data()),
          frameBytes, frameBytes, input.chunks(), nullptr, true);
      if (cudaGetLastError() != cudaSuccess) return ncclUnhandledCudaError;
      return output.commitPlanned();
    }
    const uint32_t frames = (uint32_t)input.chunks();
    size_t activeRegion = 0;
    size_t stackBytes = 0;
    size_t scratchBytes = 0;
    const DietGpuConfig& config = context.config<DietGpuConfig>();
    if (!alignedProduct(frames, sizeof(uint8_t), &activeRegion) ||
        !decodeStackBytes(frames, config.probBits, &stackBytes) ||
        !coccl::checkedAdd(activeRegion, stackBytes, &scratchBytes)) {
      return ncclInvalidArgument;
    }
    coccl::Buffer scratch;
    coccl::Status result = context.scratch(scratchBytes, &scratch);
    if (result != ncclSuccess) return result;
    uint8_t* active = static_cast<uint8_t*>(scratch.data());
    void* stackBase = active + activeRegion;
    dietgpu::StackDeviceMemory stack(
        context.cudaDevice(), stackBase, stackBytes);

    const dim3 grid(frames);
    prepareDecodeFrames<<<grid, kThreads, 0, context.stream()>>>(
        static_cast<const uint8_t*>(input.data()), frameBytes,
        input.frameMetadata(), static_cast<uint8_t*>(output.data()),
        frameBytes, frameBytes, input.chunks(), active, false);
    if (cudaGetLastError() != cudaSuccess) return ncclUnhandledCudaError;
    const dietgpu::ANSDecodeStatus decodeStatus =
        dietgpu::ansDecodeBatchStrideMasked(
            stack, dietgpu::ANSCodecConfig(config.probBits, false), frames,
            input.data(), (uint32_t)frameBytes, active, output.data(),
            (uint32_t)frameBytes, (uint32_t)frameBytes, nullptr, nullptr,
            context.stream());
    if (decodeStatus.error != dietgpu::ANSDecodeError::None ||
        cudaGetLastError() != cudaSuccess) {
      return ncclUnhandledCudaError;
    }
    return output.commitPlanned();
  }
};

}  // namespace

COCCL_REGISTER_COMPRESSOR("dietgpu", DietGpuCompressor)
