#include "quantization.h"
#include "compressor_plugin/coccl_compressor_plugin.h"

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
  return value == 4 || value == 8;
}

bool validPacking(int groupCount, int quantBits) {
  return validQuantBits(quantBits) && groupCount > 0 &&
      groupCount % (8 / quantBits) == 0;
}

bool parameterBytes(ncclDataType_t datatype, quantize::Type quantType,
                    size_t* bytes) {
  switch (datatype) {
    case ncclFloat32:
      *bytes = (quantType == quantize::Type::Symmetric ? 1u : 2u) *
          sizeof(float);
      return true;
    case ncclFloat16:
    case ncclBfloat16:
      *bytes = 2u * sizeof(float);
      return true;
    default:
      return false;
  }
}

struct TahEncodedLayout {
  size_t parameterBytes = 0;
  size_t groupDataBytes = 0;
  size_t groups = 0;
  size_t positionOffset = 0;
  size_t flagOffset = 0;
  size_t bytes = 0;
};

bool encodedLayout(size_t elements, size_t chunks, int groupCount,
                   int quantBits, size_t parameterBytes,
                   bool pivotSwap, TahEncodedLayout* layout) {
  if (layout == nullptr || elements == 0 || chunks == 0 ||
      elements % chunks != 0 || !validPacking(groupCount, quantBits)) {
    return false;
  }

  const size_t elementsPerChunk = elements / chunks;
  if (elementsPerChunk % (size_t)groupCount != 0) return false;
  const size_t groupsPerChunk = elementsPerChunk / (size_t)groupCount;
  layout->parameterBytes = parameterBytes;
  layout->groupDataBytes =
      (size_t)groupCount / (size_t)(8 / quantBits);
  size_t groupBytes = 0;
  if (!coccl::checkedAdd(layout->groupDataBytes, parameterBytes,
                         &groupBytes) ||
      !coccl::checkedMultiply(groupsPerChunk, chunks, &layout->groups) ||
      !coccl::checkedMultiply(layout->groups, groupBytes,
                              &layout->positionOffset)) {
    return false;
  }
  layout->flagOffset = layout->positionOffset;
  layout->bytes = layout->positionOffset;
  if (!pivotSwap) return true;
  size_t positionBytes = 0;
  return coccl::checkedMultiply(layout->groups, sizeof(int64_t),
                                &positionBytes) &&
      coccl::checkedAdd(layout->positionOffset, positionBytes,
                        &layout->flagOffset) &&
      coccl::checkedAdd(layout->flagOffset, layout->groups, &layout->bytes);
}

template <typename Shape>
bool compressedLayout(const Shape& input, const TahQuantConfig& config,
                      TahEncodedLayout* layout) {
  size_t params = 0;
  if (!parameterBytes(input.datatype(), config.quantType, &params)) {
    return false;
  }
  return encodedLayout(input.elements(), input.chunks(),
                       config.kernelGroupCount(), config.quantBits, params,
                       config.hadamard && config.pivotSwap, layout);
}

template <typename Shape>
bool compressedBytes(const Shape& input, const TahQuantConfig& config,
                     size_t* bytes) {
  TahEncodedLayout layout;
  if (bytes == nullptr || !compressedLayout(input, config, &layout)) {
    return false;
  }
  *bytes = layout.bytes;
  return true;
}

struct TahDrcLayout {
  size_t parameterBytes = 0;
  size_t groupDataBytes = 0;
  size_t inputGroups = 0;
  size_t outputGroups = 0;
  size_t inputTensorDataBytes = 0;
  size_t outputBytes = 0;
};

template <typename Shape>
bool drcLayout(const Shape& input, const TahQuantConfig& config,
               size_t reduceChunks, ncclDataType_t originalDatatype,
               size_t originalElements,
               TahDrcLayout* layout) {
  if (layout == nullptr || reduceChunks == 0 || input.chunks() == 0 ||
      input.chunks() % reduceChunks != 0 ||
      input.bytes() % input.chunks() != 0 ||
      !validPacking(config.groupCount, config.quantBits)) {
    return false;
  }
  const size_t outputChunks = input.chunks() / reduceChunks;
  if (outputChunks == 0 || originalElements == 0 ||
      originalElements % outputChunks != 0) {
    return false;
  }

  size_t params = 0;
  if (!parameterBytes(originalDatatype, config.quantType, &params)) {
    return false;
  }
  TahEncodedLayout outputLayout;
  if (!encodedLayout(originalElements, outputChunks, config.groupCount,
                     config.quantBits, params, false, &outputLayout)) {
    return false;
  }
  layout->parameterBytes = params;
  layout->groupDataBytes = outputLayout.groupDataBytes;
  layout->outputGroups = outputLayout.groups;
  layout->outputBytes = outputLayout.bytes;
  size_t encodedGroupBytes = 0;
  if (!coccl::checkedAdd(layout->groupDataBytes, params,
                         &encodedGroupBytes) || encodedGroupBytes == 0) {
    return false;
  }
  const size_t encodedInputChunkBytes = input.bytes() / input.chunks();
  if (encodedInputChunkBytes % encodedGroupBytes != 0) return false;
  const size_t inputGroupsPerChunk =
      encodedInputChunkBytes / encodedGroupBytes;
  return inputGroupsPerChunk != 0 &&
      coccl::checkedMultiply(inputGroupsPerChunk, outputChunks,
                             &layout->inputGroups) &&
      coccl::checkedMultiply(layout->inputGroups, layout->groupDataBytes,
                             &layout->inputTensorDataBytes);
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
    return validPacking(config.kernelGroupCount(), config.quantBits) &&
            (!config.pivotSwap || config.hadamard)
        ? ncclSuccess : ncclInvalidArgument;
  }

  static coccl::Status encodedSizeBound(
      const coccl::Shape& input, size_t* encodedBytes,
      const coccl::SizeContext& context) {
    if (encodedBytes == nullptr) return ncclInvalidArgument;
    const Config& config = context.config<Config>();
    if (context.operation() == cocclCompressorOperationCompress) {
      return compressedBytes(input, config, encodedBytes)
          ? ncclSuccess : ncclInvalidArgument;
    }
    if (context.operation() !=
        cocclCompressorOperationDecompressReduceCompress) {
      return ncclInvalidUsage;
    }
    if (config.hadamard || input.datatype() != ncclFloat32 ||
        (config.quantType == quantize::Type::Asymmetric &&
         config.quantBits != 8)) {
      return ncclInvalidArgument;
    }
    size_t params = 0;
    if (!parameterBytes(input.datatype(), config.quantType, &params)) {
      return ncclInvalidArgument;
    }
    TahEncodedLayout layout;
    if (!encodedLayout(input.elements(), input.chunks(), config.groupCount,
                       config.quantBits, params, false, &layout)) {
      return ncclInvalidArgument;
    }
    *encodedBytes = layout.bytes;
    return ncclSuccess;
  }

  static coccl::Status compress(const coccl::Input& input,
                                coccl::Output& output,
                                coccl::Context& context) {
    const Config& config = context.config<Config>();
    TahEncodedLayout layout;
    if (!compressedLayout(input, config, &layout)) {
      return ncclInvalidArgument;
    }
    const size_t requiredBytes = layout.bytes;
    if (coccl::shouldPassthrough(input, requiredBytes)) {
      return output.passthrough(input, context.stream());
    }
    if (requiredBytes > output.capacityBytes()) return ncclInvalidArgument;

    const int groupCount = config.kernelGroupCount();
    if (layout.groups > INT_MAX) {
      return ncclInvalidArgument;
    }
    const int totalGroups = (int)layout.groups;

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
            static_cast<char*>(output.data()) + layout.positionOffset);
        auto* flags = reinterpret_cast<bool*>(
            static_cast<char*>(output.data()) + layout.flagOffset);
        launch_pivot_swap_experimental(typedScratch, totalGroups,
                                       groupCount, positions, flags,
                                       context.stream());
        launch_quant_heuristic_ht(
            output.dataAs<int8_t>(), nullptr, typedScratch, totalGroups,
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
    TahEncodedLayout layout;
    if (!compressedLayout(output, config, &layout) ||
        layout.bytes != input.bytes() || output.elements() > INT64_MAX) {
      return ncclInvalidArgument;
    }

    const int groupCount = config.kernelGroupCount();
    const bool usePivot = config.hadamard && config.pivotSwap;
    if (usePivot && layout.groups > INT_MAX) {
      return ncclInvalidArgument;
    }
    char* encoded = static_cast<char*>(const_cast<void*>(input.data()));
    auto* positions =
        reinterpret_cast<int64_t*>(encoded + layout.positionOffset);
    auto* flags = reinterpret_cast<bool*>(encoded + layout.flagOffset);

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
        launch_swap_back_experimental(typedOutput, (int)layout.groups,
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

  static coccl::Status decompressReduce(const coccl::Input& input,
                                        coccl::Output& output,
                                        coccl::Context& context) {
    const Config& config = context.config<Config>();
    const size_t reduceChunks = context.reduceChunks();
    if (output.datatype() != ncclFloat32 || output.chunks() != 1 ||
        config.pivotSwap || reduceChunks != input.chunks() ||
        output.elements() % (size_t)config.kernelGroupCount() != 0) {
      return ncclInvalidArgument;
    }

    const int groupCount = config.kernelGroupCount();
    const size_t groups = output.elements() / (size_t)groupCount;
    const size_t groupDataBytes =
        (size_t)groupCount / (size_t)(8 / config.quantBits);
    const size_t encodedGroupBytes = groupDataBytes +
        (config.quantType == quantize::Type::Symmetric ? 1u : 2u) *
            sizeof(float);
    size_t encodedChunkBytes = 0;
    size_t expectedInputBytes = 0;
    size_t inputTensorDataBytes = 0;
    if (!coccl::checkedMultiply(groups, encodedGroupBytes,
                                &encodedChunkBytes) ||
        !coccl::checkedMultiply(encodedChunkBytes, input.chunks(),
                                &expectedInputBytes) ||
        !coccl::checkedMultiply(groups, groupDataBytes,
                                &inputTensorDataBytes) ||
        input.bytes() != expectedInputBytes || groups > INT_MAX ||
        groupDataBytes > INT_MAX || inputTensorDataBytes > INT64_MAX ||
        reduceChunks > INT_MAX) {
      return ncclInvalidArgument;
    }

    auto launch = config.hadamard ? launch_dequant_reduce_ht
                                  : launch_dequant_reduce;
    launch(output.dataAs<float>(), input.dataAs<int8_t>(), nullptr,
           (int)reduceChunks, config.quantBits, config.quantType,
           (int)groups, (int)groupDataBytes,
           (int64_t)inputTensorDataBytes, (int)groups,
           (int)groupDataBytes, context.stream());
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
    if (config.hadamard || context.originalDatatype() != ncclFloat32 ||
        (config.quantType == quantize::Type::Asymmetric &&
         config.quantBits != 8)) {
      return ncclInvalidArgument;
    }
    TahDrcLayout layout;
    if (!drcLayout(input, config, reduceChunks, context.originalDatatype(),
                   context.originalElements(), &layout) ||
        layout.groupDataBytes > INT_MAX || layout.inputGroups > INT_MAX ||
        layout.outputGroups > INT_MAX ||
        layout.inputTensorDataBytes > INT64_MAX) {
      return ncclInvalidArgument;
    }
    if (layout.outputBytes > output.capacityBytes()) return ncclInvalidUsage;

    launch_dequant_reduce_quant(
        output.dataAs<int8_t>(), nullptr, input.dataAs<int8_t>(), nullptr,
        (int)reduceChunks, config.quantBits, config.quantBits,
        config.quantType, (int)layout.outputGroups,
        (int)layout.groupDataBytes, (int64_t)layout.inputTensorDataBytes,
        (int)layout.inputGroups, (int)layout.groupDataBytes,
        context.stream());
    cudaError_t cudaResult = cudaGetLastError();
    if (cudaResult != cudaSuccess) return coccl::fromCuda(cudaResult);
    return output.commitBytes(layout.outputBytes,
                              input.chunks() / reduceChunks);
  }
};

}  // namespace

COCCL_REGISTER_COMPRESSOR("tahquant", TahQuantCompressor);
