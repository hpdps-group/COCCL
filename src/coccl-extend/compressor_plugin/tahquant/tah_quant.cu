#include "quantization.h"
#include "compressor_plugin/compressor_plugin.h"

#include <climits>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

struct TahQuantConfig {
  int groupCount = 2048;
  int quantBits = 8;
  bool hadamard = false;
  bool pivotSwap = false;
  quantize::Type quantType = quantize::Type::Symmetric;

  int kernelGroupCount() const {
    return hadamard && groupCount > 128 ? 128 : groupCount;
  }
};

bool validQuantBits(int value) {
  return value == 1 || value == 2 || value == 4 || value == 8;
}

bool compressedBytes(const coccl::Input& input, const TahQuantConfig& config,
                     size_t* bytes) {
  const int groupCount = config.kernelGroupCount();
  if (input.elementsPerChunk() == 0 || groupCount <= 0 ||
      !validQuantBits(config.quantBits)) {
    return false;
  }

  const size_t groups =
      1 + (input.elementsPerChunk() - 1) / (size_t)groupCount;
  const size_t quantBytes =
      (size_t)groupCount / (size_t)(8 / config.quantBits);
  const size_t parameterBytes = input.datatype() == ncclFloat32
      ? (config.quantType == quantize::Type::Symmetric ? 1u : 2u) *
            sizeof(float)
      : 2u * sizeof(float);
  size_t groupBytes = 0;
  if (!coccl::checkedAdd(quantBytes, parameterBytes, &groupBytes)) {
    return false;
  }
  if (config.hadamard && config.pivotSwap &&
      !coccl::checkedAdd(groupBytes, sizeof(int64_t) + 1, &groupBytes)) {
    return false;
  }
  size_t chunkBytes = 0;
  return coccl::checkedMultiply(groups, groupBytes, &chunkBytes) &&
         coccl::checkedMultiply(chunkBytes, input.chunks(), bytes);
}

struct TahQuantCompressor {
  using Config = TahQuantConfig;

  static coccl::Status configure(coccl::ConfigReader& reader, Config& config,
                                 const coccl::ConfigContext&) {
    coccl::Status result =
        reader.get("groupCount", config.groupCount, 1, INT_MAX)
            .get("quantBits", config.quantBits, 1, 8)
            .get("hadamard", config.hadamard)
            .get("pivotSwap", config.pivotSwap)
            .getEnum("quantType", config.quantType,
                     {{"Symmetric", quantize::Type::Symmetric},
                      {"Asymmetric", quantize::Type::Asymmetric}})
            .finish();
    if (result != ncclSuccess) return result;
    return validQuantBits(config.quantBits) ? ncclSuccess
                                            : ncclInvalidArgument;
  }

  static coccl::Status compress(const coccl::Input& input,
                                coccl::Output& output,
                                coccl::Context& context) {
    const Config& config = context.config<Config>();
    size_t requiredBytes = 0;
    if (!compressedBytes(input, config, &requiredBytes)) {
      return ncclInvalidArgument;
    }
    if (coccl::shouldPassthrough(input, requiredBytes)) {
      return output.passthrough(input, context.stream());
    }
    if (requiredBytes > output.capacityBytes()) return ncclInvalidArgument;

    const int groupCount = config.kernelGroupCount();
    const size_t groupsPerChunk =
        1 + (input.elementsPerChunk() - 1) / (size_t)groupCount;
    size_t totalGroups = 0;
    if (!coccl::checkedMultiply(groupsPerChunk, input.chunks(),
                                &totalGroups) ||
        totalGroups > INT_MAX) {
      return ncclInvalidArgument;
    }
    const size_t quantBytes =
        (size_t)groupCount / (size_t)(8 / config.quantBits);
    const size_t parameterBytes = input.datatype() == ncclFloat32
        ? (config.quantType == quantize::Type::Symmetric ? 1u : 2u) *
              sizeof(float)
        : 2u * sizeof(float);
    const size_t positionOffset =
        groupsPerChunk * (quantBytes + parameterBytes);
    const size_t flagOffset = positionOffset +
        groupsPerChunk * sizeof(int64_t);

    const bool usePivot = config.hadamard && config.pivotSwap;
    coccl::Buffer scratch;
    if (usePivot) {
      coccl::Status result = context.scratch(input.bytes(), &scratch);
      if (result != ncclSuccess) return result;
      cudaError_t cudaResult = cudaMemcpyAsync(
          scratch.data(), input.data(), input.bytes(), cudaMemcpyDeviceToDevice,
          context.stream());
      if (cudaResult != cudaSuccess) return coccl::fromCuda(cudaResult);
    }

    auto launch = [&](const auto* typedInput, auto* typedScratch) {
      if (!config.hadamard) {
        launch_quant(output.dataAs<int8_t>(), nullptr, typedInput,
                     (int)totalGroups, groupCount, config.quantBits,
                     config.quantType, context.stream());
      } else if (!usePivot) {
        launch_quant_ht(output.dataAs<int8_t>(), nullptr, typedInput,
                        (int)totalGroups, groupCount, config.quantBits,
                        config.quantType, context.stream());
      } else {
        auto* positions = reinterpret_cast<int64_t*>(
            static_cast<char*>(output.data()) + positionOffset);
        auto* flags = reinterpret_cast<bool*>(
            static_cast<char*>(output.data()) + flagOffset);
        launch_pivot_swap_experimental(typedScratch, (int)totalGroups,
                                       groupCount, positions, flags,
                                       context.stream());
        launch_quant_heuristic_ht(
            output.dataAs<int8_t>(), nullptr, typedScratch, (int)totalGroups,
            groupCount, config.quantBits, config.quantType, flags,
            context.stream());
      }
    };
    switch (input.datatype()) {
      case ncclFloat32:
        launch(input.dataAs<float>(), scratch.dataAs<float>());
        break;
      case ncclFloat16:
        launch(input.dataAs<__half>(), scratch.dataAs<__half>());
        break;
      case ncclBfloat16:
        launch(input.dataAs<__nv_bfloat16>(),
               scratch.dataAs<__nv_bfloat16>());
        break;
      default:
        return ncclInvalidArgument;
    }

    cudaError_t cudaResult = cudaGetLastError();
    if (cudaResult != cudaSuccess) return coccl::fromCuda(cudaResult);
    return output.commitBytes(requiredBytes, input.chunks());
  }

  static coccl::Status decompress(const coccl::Input& input,
                                  coccl::Output& output,
                                  coccl::Context& context) {
    const Config& config = context.config<Config>();
    const int groupCount = config.kernelGroupCount();
    const size_t elementsPerChunk = output.elements() / output.chunks();
    const size_t groupsPerChunk = elementsPerChunk == 0
        ? 0
        : 1 + (elementsPerChunk - 1) / (size_t)groupCount;
    if (groupsPerChunk == 0 || output.elements() > INT64_MAX) {
      return ncclInvalidArgument;
    }

    const size_t quantBytes =
        (size_t)groupCount / (size_t)(8 / config.quantBits);
    const size_t parameterBytes = output.datatype() == ncclFloat32
        ? (config.quantType == quantize::Type::Symmetric ? 1u : 2u) *
              sizeof(float)
        : 2u * sizeof(float);
    const size_t positionOffset =
        groupsPerChunk * (quantBytes + parameterBytes);
    const size_t flagOffset = positionOffset +
        groupsPerChunk * sizeof(int64_t);
    const bool usePivot = config.hadamard && config.pivotSwap;
    size_t totalGroups = 0;
    if (usePivot &&
        (!coccl::checkedMultiply(groupsPerChunk, output.chunks(),
                                 &totalGroups) ||
         totalGroups > INT_MAX)) {
      return ncclInvalidArgument;
    }
    char* encoded = static_cast<char*>(const_cast<void*>(input.data()));
    auto* positions = reinterpret_cast<int64_t*>(encoded + positionOffset);
    auto* flags = reinterpret_cast<bool*>(encoded + flagOffset);

    auto launch = [&](auto* typedOutput) {
      if (!config.hadamard) {
        launch_dequantize_kernel(
            typedOutput, input.dataAs<int8_t>(), nullptr, config.quantType,
            config.quantBits, groupCount, (int64_t)output.elements(),
            context.stream());
      } else if (!usePivot) {
        launch_dequantize_ht_kernel(
            typedOutput, input.dataAs<int8_t>(), nullptr, config.quantType,
            config.quantBits, groupCount, (int64_t)output.elements(),
            context.stream());
      } else {
        launch_dequantize_heuristic_ht_kernel(
            typedOutput, input.dataAs<int8_t>(), nullptr, config.quantType,
            config.quantBits, groupCount, (int64_t)output.elements(), flags,
            context.stream());
        launch_swap_back_experimental(typedOutput, (int)totalGroups,
                                      groupCount, positions, flags,
                                      context.stream());
      }
    };
    switch (output.datatype()) {
      case ncclFloat32: launch(output.dataAs<float>()); break;
      case ncclFloat16: launch(output.dataAs<__half>()); break;
      case ncclBfloat16: launch(output.dataAs<__nv_bfloat16>()); break;
      default: return ncclInvalidArgument;
    }
    return coccl::fromCuda(cudaGetLastError());
  }

  static coccl::Status decompressReduceCompress(
      const coccl::Input& input, coccl::Output& output,
      coccl::Context& context) {
    const size_t reduceChunks = context.reduceChunks();
    if (reduceChunks == 0 || input.chunks() % reduceChunks != 0 ||
        input.elements() % reduceChunks != 0 || reduceChunks > INT_MAX) {
      return ncclInvalidArgument;
    }

    const Config& config = context.config<Config>();
    const int groupDataBytes = config.groupCount / (8 / config.quantBits);
    const int parameterBytes =
        (config.quantType == quantize::Type::Symmetric ? 1 : 2) *
        (int)sizeof(float);
    const int encodedGroupBytes = groupDataBytes + parameterBytes;
    const size_t inputElements = input.elements() / reduceChunks;
    const size_t inputGroups = inputElements == 0
        ? 0
        : 1 + (inputElements - 1) / (size_t)encodedGroupBytes;
    size_t originalElements = 0;
    if (inputGroups == 0 ||
        !coccl::checkedMultiply(inputGroups, (size_t)config.groupCount,
                                &originalElements)) {
      return ncclInvalidArgument;
    }
    const size_t outputGroups =
        1 + (originalElements - 1) / (size_t)config.groupCount;
    size_t inputChunkBytes = 0;
    size_t outputBytes = 0;
    if (!coccl::checkedMultiply(inputGroups, (size_t)groupDataBytes,
                                &inputChunkBytes) ||
        !coccl::checkedMultiply(outputGroups, (size_t)encodedGroupBytes,
                                &outputBytes) ||
        inputGroups > INT_MAX || outputGroups > INT_MAX ||
        inputChunkBytes > INT64_MAX || outputBytes > output.capacityBytes()) {
      return ncclInvalidArgument;
    }

    launch_dequant_reduce_quant(
        output.dataAs<int8_t>(), nullptr, input.dataAs<int8_t>(), nullptr,
        (int)reduceChunks, config.quantBits, config.quantBits,
        config.quantType, (int)outputGroups, groupDataBytes,
        (int64_t)inputChunkBytes, (int)inputGroups, groupDataBytes,
        context.stream());
    cudaError_t cudaResult = cudaGetLastError();
    if (cudaResult != cudaSuccess) return coccl::fromCuda(cudaResult);
    return output.commitBytes(outputBytes, input.chunks() / reduceChunks);
  }
};

}  // namespace

COCCL_REGISTER_COMPRESSOR("tahquant", TahQuantCompressor);
