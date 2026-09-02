#include "compressor_plugin/coccl_compressor_plugin.h"

#include "dietgpu/ans/BatchPrefixSum.cuh"
#include "dietgpu/ans/GpuANSCodec.h"
#include "dietgpu/ans/GpuANSEncode.cuh"
#include "dietgpu/ans/GpuANSUtils.cuh"
#include "dietgpu/float/GpuFloatCodec.h"
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

bool ansEncodeStackBytes(uint32_t frames, uint32_t frameBytes,
                         bool precomputedHistogram,
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

  size_t histogramPhase = table;
  if (!precomputedHistogram) {
    if (!alignedProduct((size_t)frames * dietgpu::kNumSymbols,
                        sizeof(uint32_t), &histogram) ||
        !addBytes(histogram, &histogramPhase)) {
      return false;
    }
  }
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

bool floatTypeFor(ncclDataType_t datatype, dietgpu::FloatType* floatType,
                  uint32_t* wordBytes) {
  switch (datatype) {
    case ncclFloat16:
      *floatType = dietgpu::FloatType::kFloat16;
      *wordBytes = 2;
      return true;
    case ncclBfloat16:
      *floatType = dietgpu::FloatType::kBFloat16;
      *wordBytes = 2;
      return true;
    case ncclFloat32:
      *floatType = dietgpu::FloatType::kFloat32;
      *wordBytes = 4;
      return true;
    default:
      return false;
  }
}

bool floatEncodeStackBytes(uint32_t frames, uint32_t frameElements,
                           size_t* stackBytes) {
  const uint32_t symbolStride =
      dietgpu::roundUp(frameElements, (uint32_t)sizeof(uint4));
  size_t checksum = 0;
  size_t symbols = 0;
  size_t histogram = 0;
  size_t ans = 0;
  if (!alignedProduct(frames, sizeof(uint32_t), &checksum) ||
      !alignedProduct(frames, symbolStride, &symbols) ||
      !alignedProduct((size_t)frames * dietgpu::kNumSymbols,
                      sizeof(uint32_t), &histogram) ||
      !ansEncodeStackBytes(frames, frameElements, true, &ans)) {
    return false;
  }
  *stackBytes = checksum;
  return addBytes(symbols, stackBytes) &&
      addBytes(histogram, stackBytes) && addBytes(ans, stackBytes);
}

bool decodeStackBytes(uint32_t frames, int probBits, size_t* stackBytes);

bool floatDecodeStackBytes(uint32_t frames, uint32_t frameElements,
                           int probBits, bool alignedOutput,
                           size_t* stackBytes) {
  if (!decodeStackBytes(frames, probBits, stackBytes)) return false;
  if (alignedOutput) return true;
  const uint32_t symbolStride =
      dietgpu::roundUp(frameElements, (uint32_t)sizeof(uint4));
  size_t symbols = 0;
  return alignedProduct(frames, symbolStride, &symbols) &&
      addBytes(symbols, stackBytes);
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

uint32_t ansCandidateStride(size_t frameBytes) {
  if (frameBytes == 0 || frameBytes > UINT32_MAX ||
      frameBytes % dietgpu::kANSRequiredAlignment != 0) {
    return 0;
  }

  const uint64_t blocks =
      (frameBytes + dietgpu::kDefaultBlockSize - 1) /
      dietgpu::kDefaultBlockSize;
  uint64_t bytes =
      dietgpu::ANSCoalescedHeader::getCompressedOverhead(
          dietgpu::kDefaultBlockSize) +
      blocks * dietgpu::getMaxBlockSizeCoalesced(
          dietgpu::kDefaultBlockSize);
  bytes = (bytes + sizeof(uint4) - 1) / sizeof(uint4) * sizeof(uint4);
  return bytes <= INT32_MAX ? (uint32_t)bytes : 0;
}

uint32_t floatCandidateStride(dietgpu::FloatType floatType,
                              size_t frameElements, size_t frameBytes) {
  if (frameElements == 0 || frameElements > UINT32_MAX ||
      frameBytes > INT32_MAX) return 0;
  const uint64_t bytes = dietgpu::getMaxFloatCompressedSize(
      floatType, (uint32_t)frameElements);
  const uint64_t aligned =
      (bytes + sizeof(uint4) - 1) / sizeof(uint4) * sizeof(uint4);
  return aligned <= UINT32_MAX ? (uint32_t)aligned : 0;
}

bool encodedFramesAligned(const coccl::Output& output) {
  constexpr size_t alignment = alignof(dietgpu::ANSCoalescedHeader);
  return (uintptr_t)output.data() % alignment == 0 &&
      output.frameStrideBytes() % alignment == 0;
}

unsigned int frameCopyBlocks(size_t frameBytes) {
  constexpr size_t kBytesPerBlock = 2048;
  constexpr size_t kMaxBlocks = 8192;
  const size_t blocks = (frameBytes - 1) / kBytesPerBlock + 1;
  return (unsigned int)(blocks < kMaxBlocks ? blocks : kMaxBlocks);
}

__global__ void markRawFrames(
    cocclCompressorFrameMetadata* metadata, size_t frameBytes,
    size_t frames) {
  for (size_t frame = blockIdx.x * blockDim.x + threadIdx.x;
       frame < frames; frame += (size_t)blockDim.x * gridDim.x) {
    metadata[frame] = {
        frameBytes, cocclCompressorFrameRaw, 0};
  }
}

__global__ void finalizeFrames(
    const uint8_t* input, size_t inputStride,
    const uint8_t* candidate, size_t candidateStride,
    const uint32_t* candidateBytes, uint8_t* output, size_t outputStride,
    cocclCompressorFrameMetadata* metadata, size_t frameBytes,
    size_t frames, bool useCandidate) {
  for (size_t frame = blockIdx.y; frame < frames; frame += gridDim.y) {
    const uint32_t encodedBytes =
        useCandidate ? candidateBytes[frame] : UINT32_MAX;
    const bool encoded = encodedBytes > 0 && encodedBytes < frameBytes;
    const size_t copyBytes = encoded ? (size_t)encodedBytes : frameBytes;
    const uint8_t* source = encoded
        ? candidate + frame * candidateStride
        : input + frame * inputStride;
    uint8_t* destination = output + frame * outputStride;
    for (size_t offset = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         offset < copyBytes; offset += (size_t)gridDim.x * blockDim.x) {
      destination[offset] = source[offset];
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
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
    uint8_t* active) {
  for (size_t frame = blockIdx.y; frame < frames; frame += gridDim.y) {
    const cocclCompressorFrameMetadata frameMetadata = metadata[frame];
    const bool encoded =
        frameMetadata.encoding == cocclCompressorFrameEncoded;
    if (blockIdx.x == 0 && threadIdx.x == 0 && active != nullptr) {
      active[frame] = encoded ? 1 : 0;
    }
    if (!encoded) {
      for (size_t offset = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
           offset < frameBytes; offset += (size_t)gridDim.x * blockDim.x) {
        output[frame * outputStride + offset] =
            input[frame * inputStride + offset];
      }
    }
  }
}

struct DietGpuCompressor {
  using Config = DietGpuConfig;
  static constexpr bool kFramed = true;
  static constexpr bool kBytewiseLossless = true;

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
    constexpr size_t alignment = alignof(dietgpu::ANSCoalescedHeader);
    const size_t frameBytes = input.bytes() / input.chunks();
    size_t paddedFrameBytes = 0;
    return coccl::checkedAdd(
               frameBytes, alignment - 1, &paddedFrameBytes) &&
            coccl::checkedMultiply(
               paddedFrameBytes / alignment * alignment,
               input.chunks(), encodedBytes)
        ? ncclSuccess : ncclInvalidUsage;
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
    dietgpu::FloatType floatType = dietgpu::FloatType::kUndefined;
    uint32_t wordBytes = 0;
    const bool floatCodec =
        floatTypeFor(input.datatype(), &floatType, &wordBytes);
    if (floatCodec && frameBytes % wordBytes != 0) {
      return ncclInvalidArgument;
    }
    const size_t frameElements = floatCodec ? frameBytes / wordBytes : 0;
    const uint32_t candidateStride = floatCodec
        ? floatCandidateStride(floatType, frameElements, frameBytes)
        : ansCandidateStride(frameBytes);
    size_t outputBytes = 0;
    if (output.frameStrideBytes() < frameBytes ||
        !coccl::checkedMultiply(output.frameStrideBytes(), input.chunks(),
                                &outputBytes) ||
        outputBytes > output.capacityBytes()) {
      return ncclInvalidArgument;
    }
    constexpr int kThreads = 256;
    if (input.chunks() > kMaxBatchFrames || candidateStride == 0 ||
        ((uintptr_t)input.data() % dietgpu::kANSRequiredAlignment) != 0 ||
        !encodedFramesAligned(output)) {
      if (cudaMemcpy2DAsync(
              output.data(), output.frameStrideBytes(), input.data(),
              frameBytes, frameBytes, input.chunks(),
              cudaMemcpyDeviceToDevice, context.stream()) != cudaSuccess) {
        return ncclUnhandledCudaError;
      }
      const unsigned int metadataBlocks = (unsigned int)(
          (input.chunks() + kThreads - 1) / kThreads);
      markRawFrames<<<metadataBlocks, kThreads, 0, context.stream()>>>(
          output.frameMetadata(), frameBytes, input.chunks());
      if (cudaGetLastError() != cudaSuccess) return ncclUnhandledCudaError;
      return output.commitFrames();
    }

    const uint32_t frames = (uint32_t)input.chunks();
    const uint32_t rawFrameBytes = (uint32_t)frameBytes;
    size_t candidateBytes = 0;
    size_t candidateRegion = 0;
    size_t sizeRegion = 0;
    size_t stackBytes = 0;
    size_t scratchBytes = 0;
    if (!coccl::checkedMultiply(
            (size_t)candidateStride, (size_t)frames, &candidateBytes) ||
        !alignedBytes(candidateBytes, &candidateRegion) ||
        !alignedProduct(frames, sizeof(uint32_t), &sizeRegion) ||
        !(floatCodec
              ? floatEncodeStackBytes(
                    frames, (uint32_t)frameElements, &stackBytes)
              : ansEncodeStackBytes(frames, rawFrameBytes, false,
                                    &stackBytes)) ||
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
    const dietgpu::ANSCodecConfig ansConfig(config.probBits, false);
    if (floatCodec) {
      const bool alignedInput =
          (uintptr_t)input.data() % sizeof(uint4) == 0 &&
          frameBytes % sizeof(uint4) == 0;
      dietgpu::floatCompressBatchStride(
          stack,
          dietgpu::FloatCompressConfig(
              floatType, ansConfig, alignedInput, false),
          frames, input.data(), (uint32_t)frameElements, rawFrameBytes,
          candidate, candidateStride, sizes, context.stream());
    } else {
      dietgpu::ansEncodeBatchStride(
          stack, ansConfig, frames, input.data(), rawFrameBytes,
          rawFrameBytes, nullptr, candidate, candidateStride, sizes,
          context.stream());
    }
    const dim3 finalizeGrid(
        frameCopyBlocks(frameBytes), (unsigned int)input.chunks());
    finalizeFrames<<<finalizeGrid, kThreads, 0, context.stream()>>>(
        static_cast<const uint8_t*>(input.data()), frameBytes,
        static_cast<const uint8_t*>(candidate), candidateStride, sizes,
        static_cast<uint8_t*>(output.data()), output.frameStrideBytes(),
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
    const size_t frameStrideBytes = input.frameStrideBytes();
    size_t outputBytes = 0;
    size_t inputBytes = 0;
    if (!coccl::checkedMultiply(
            output.elements(), coccl::dataTypeSize(output.datatype()),
            &outputBytes) ||
        !coccl::checkedMultiply(
            frameStrideBytes, input.chunks(), &inputBytes) ||
        inputBytes != input.bytes() || outputBytes > output.capacityBytes() ||
        outputBytes % output.chunks() != 0) {
      return ncclInvalidArgument;
    }
    const size_t frameBytes = outputBytes / output.chunks();
    if (frameBytes == 0 || frameBytes > frameStrideBytes) {
      return ncclInvalidArgument;
    }
    dietgpu::FloatType floatType = dietgpu::FloatType::kUndefined;
    uint32_t wordBytes = 0;
    const bool floatCodec =
        floatTypeFor(output.datatype(), &floatType, &wordBytes);
    if (floatCodec && frameBytes % wordBytes != 0) {
      return ncclInvalidArgument;
    }
    const size_t frameElements = floatCodec ? frameBytes / wordBytes : 0;
    const uint32_t candidateStride = floatCodec
        ? floatCandidateStride(floatType, frameElements, frameBytes)
        : ansCandidateStride(frameBytes);
    constexpr int kThreads = 256;
    if (input.chunks() > kMaxBatchFrames || candidateStride == 0) {
      if (cudaMemcpy2DAsync(
              output.data(), frameBytes, input.data(), frameStrideBytes,
              frameBytes, input.chunks(), cudaMemcpyDeviceToDevice,
              context.stream()) != cudaSuccess) {
        return ncclUnhandledCudaError;
      }
      return output.commitPlanned();
    }
    const uint32_t frames = (uint32_t)input.chunks();
    size_t activeRegion = 0;
    size_t stackBytes = 0;
    size_t scratchBytes = 0;
    const DietGpuConfig& config = context.config<DietGpuConfig>();
    const bool alignedOutput =
        (uintptr_t)output.data() % sizeof(uint4) == 0 &&
        frameBytes % sizeof(uint4) == 0;
    if (!alignedProduct(frames, sizeof(uint8_t), &activeRegion) ||
        !(floatCodec
              ? floatDecodeStackBytes(
                    frames, (uint32_t)frameElements, config.probBits,
                    alignedOutput, &stackBytes)
              : decodeStackBytes(frames, config.probBits, &stackBytes)) ||
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

    const dim3 grid(frameCopyBlocks(frameBytes), frames);
    prepareDecodeFrames<<<grid, kThreads, 0, context.stream()>>>(
        static_cast<const uint8_t*>(input.data()), frameStrideBytes,
        input.frameMetadata(), static_cast<uint8_t*>(output.data()),
        frameBytes, frameBytes, input.chunks(), active);
    if (cudaGetLastError() != cudaSuccess) return ncclUnhandledCudaError;
    const dietgpu::ANSCodecConfig ansConfig(config.probBits, false);
    if (floatCodec) {
      const dietgpu::FloatDecompressStatus decodeStatus =
          dietgpu::floatDecompressBatchStrideMasked(
              stack,
              dietgpu::FloatDecompressConfig(
                  floatType, ansConfig, alignedOutput, false),
              frames, input.data(), (uint32_t)frameStrideBytes, active,
              output.data(), (uint32_t)frameBytes,
              (uint32_t)frameElements, nullptr, nullptr, context.stream());
      if (decodeStatus.error != dietgpu::FloatDecompressError::None) {
        return ncclUnhandledCudaError;
      }
    } else {
      const dietgpu::ANSDecodeStatus decodeStatus =
          dietgpu::ansDecodeBatchStrideMasked(
              stack, ansConfig, frames, input.data(),
              (uint32_t)frameStrideBytes, active, output.data(),
              (uint32_t)frameBytes, (uint32_t)frameBytes, nullptr, nullptr,
              context.stream());
      if (decodeStatus.error != dietgpu::ANSDecodeError::None) {
        return ncclUnhandledCudaError;
      }
    }
    if (cudaGetLastError() != cudaSuccess) return ncclUnhandledCudaError;
    return output.commitPlanned();
  }
};

}  // namespace

COCCL_REGISTER_COMPRESSOR("dietgpu", DietGpuCompressor)
