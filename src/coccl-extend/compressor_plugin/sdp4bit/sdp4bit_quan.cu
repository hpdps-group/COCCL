#include "quantization.h"
#include "compressor_plugin/compressor_plugin.h"

#include <climits>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <type_traits>

namespace {

struct Sdp4BitConfig {
  int groupCount = 2048;
  int quantBits = 8;
  bool hadamard = false;
  quantize::Type quantType = quantize::Type::Symmetric;
  int inQuantBits = 0;
  int outQuantBits = 0;
  int inGroupCount = 0;
  int outGroupCount = 0;
  int pipelineSize = 1;
  bool subAdd = false;
  bool hierarchical = false;

  int inputBits() const {
    return inQuantBits == 0 ? quantBits : inQuantBits;
  }
  int outputBits() const {
    return outQuantBits == 0 ? quantBits : outQuantBits;
  }
  int inputGroups(bool limitHadamard = true) const {
    int groups = inGroupCount == 0 ? groupCount : inGroupCount;
    return limitHadamard && hadamard && groups > 128 ? 128 : groups;
  }
  int outputGroups(bool limitHadamard = true) const {
    int groups = outGroupCount == 0 ? groupCount : outGroupCount;
    return limitHadamard && hadamard && groups > 128 ? 128 : groups;
  }
};

// Created lazily only when the sub/add algorithm is selected.
struct Sdp4BitState {
  void* shardParams = nullptr;
  ncclDataType_t datatype = ncclNumTypes;
  bool initialized = false;
};

bool validQuantBits(int value, bool allowUnset) {
  return (allowUnset && value == 0) || value == 1 || value == 2 ||
         value == 4 || value == 8;
}

bool parameterBytes(ncclDataType_t datatype, quantize::Type quantType,
                    size_t* bytes) {
  if (bytes == nullptr) return false;
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

bool compressedBytes(const coccl::Input& input, const Sdp4BitConfig& config,
                     size_t* bytes) {
  const int groupCount = config.inputGroups();
  const int quantBits = config.inputBits();
  if (input.elementsPerChunk() == 0 || groupCount <= 0 ||
      !validQuantBits(quantBits, false)) {
    return false;
  }

  const size_t groups =
      1 + (input.elementsPerChunk() - 1) / (size_t)groupCount;
  const size_t quantBytes =
      (size_t)groupCount / (size_t)(8 / quantBits);
  size_t paramsBytes = 0;
  if (!parameterBytes(input.datatype(), config.quantType, &paramsBytes)) {
    return false;
  }
  size_t groupBytes = 0;
  size_t chunkBytes = 0;
  return coccl::checkedAdd(quantBytes, paramsBytes, &groupBytes) &&
         coccl::checkedMultiply(groups, groupBytes, &chunkBytes) &&
         coccl::checkedMultiply(chunkBytes, input.chunks(), bytes);
}

template <typename T>
void launchQuant(int8_t* output, const T* input, const coccl::Input& shape,
                 const Sdp4BitConfig& config, cudaStream_t stream) {
  const int groupCount = config.inputGroups();
  const int quantBits = config.inputBits();
  if (config.hadamard) {
    launch_quant_ht(output, nullptr, input, (int)shape.chunks(),
                    (int64_t)shape.elementsPerChunk(), groupCount, quantBits,
                    config.quantType, stream);
  } else {
    launch_quant(output, nullptr, input, (int)shape.chunks(),
                 (int64_t)shape.elementsPerChunk(), groupCount, quantBits,
                 config.quantType, stream);
  }
}

template <typename T>
void launchHierarchicalQuant(int8_t* output, const T* input,
                             const coccl::Input& shape,
                             const Sdp4BitConfig& config,
                             const coccl::Context& context) {
  const int groupCount = config.inputGroups();
  const int quantBits = config.inputBits();
  if (config.hadamard) {
    launch_swizzled_quant_ht(
        output, nullptr, input, quantBits, config.quantType,
        (int)shape.chunks(), (int64_t)shape.elementsPerChunk(), groupCount,
        config.pipelineSize, context.nodes(), context.devicesPerNode(),
        context.stream());
  } else {
    launch_swizzled_quant(
        output, nullptr, input, quantBits, config.quantType,
        (int)shape.chunks(), (int64_t)shape.elementsPerChunk(), groupCount,
        config.pipelineSize, context.nodes(), context.devicesPerNode(),
        context.stream());
  }
}

template <typename T>
void launchDequant(T* output, const coccl::Input& input,
                   const coccl::Output& shape,
                   const Sdp4BitConfig& config, cudaStream_t stream) {
  const int groupCount = config.outputGroups();
  const int quantBits = config.outputBits();
  const int64_t elementsPerChunk =
      (int64_t)(shape.elements() / shape.chunks());
  if (config.hadamard) {
    launch_dequantize_ht_kernel(
        output, input.dataAs<int8_t>(), nullptr, config.quantType, quantBits,
        groupCount, elementsPerChunk, (int64_t)shape.elements(), stream);
  } else {
    launch_dequantize_kernel(
        output, input.dataAs<int8_t>(), nullptr, config.quantType, quantBits,
        groupCount, elementsPerChunk, (int64_t)shape.elements(), stream);
  }
}

struct Sdp4BitCompressor {
  using Config = Sdp4BitConfig;

  static coccl::Status configure(coccl::ConfigReader& reader, Config& config,
                                 const coccl::ConfigContext& context) {
    config.hierarchical = context.hierarchical();
    coccl::Status result =
        reader.get("groupCount", config.groupCount, 1, INT_MAX)
            .get("quantBits", config.quantBits, 1, 8)
            .get("hadamard", config.hadamard)
            .getEnum("quantType", config.quantType,
                     {{"Symmetric", quantize::Type::Symmetric},
                      {"Asymmetric", quantize::Type::Asymmetric}})
            .get("inQuantBits", config.inQuantBits, 0, 8)
            .get("outQuantBits", config.outQuantBits, 0, 8)
            .get("inGroupCount", config.inGroupCount, 0, INT_MAX)
            .get("outGroupCount", config.outGroupCount, 0, INT_MAX)
            .get("pipelineSize", config.pipelineSize, 1, INT_MAX)
            .get("subAdd", config.subAdd)
            .finish();
    if (result != ncclSuccess) return result;
    return validQuantBits(config.quantBits, false) &&
                   validQuantBits(config.inQuantBits, true) &&
                   validQuantBits(config.outQuantBits, true) &&
                   !(config.subAdd && config.hadamard)
        ? ncclSuccess
        : ncclInvalidArgument;
  }

  static coccl::Status compress(const coccl::Input& input,
                                coccl::Output& output,
                                coccl::Context& context) {
    const Config& config = context.config<Config>();
    Sdp4BitState* state = nullptr;
    bool passthrough = false;
    if (config.subAdd) {
      size_t paramsBytes = 0;
      if (!parameterBytes(input.datatype(), config.quantType, &paramsBytes)) {
        return ncclInvalidArgument;
      }
      coccl::Status result = context.instance(&state);
      if (result != ncclSuccess) return result;
      if (state->datatype != input.datatype()) {
        state->datatype = input.datatype();
        state->initialized = false;
      }
      passthrough = !state->initialized;
    }

    size_t requiredBytes = input.bytes();
    if (!passthrough && !compressedBytes(input, config, &requiredBytes)) {
      return ncclInvalidArgument;
    }
    if (!passthrough && coccl::shouldPassthrough(input, requiredBytes)) {
      return output.passthrough(input, context.stream());
    }
    if (requiredBytes > output.capacityBytes()) return ncclInvalidArgument;

    cudaError_t cudaResult = cudaSuccess;
    if (passthrough) {
      cudaResult = cudaMemcpyAsync(output.data(), input.data(), input.bytes(),
                                   cudaMemcpyDeviceToDevice,
                                   context.stream());
    } else if (config.subAdd) {
      auto launch = [&](const auto* typedInput) {
        using T = typename std::remove_cv<typename std::remove_pointer<
            decltype(typedInput)>::type>::type;
        launch_fused_sub_quant_cuda(
            output.dataAs<int8_t>(),
            static_cast<const T*>(state->shardParams) +
                context.rank() * input.elementsPerChunk(),
            typedInput, config.inputBits(), config.quantType,
            config.inputGroups(), (int64_t)input.elementsPerChunk(), 1, 0,
            context.stream());
      };
      switch (input.datatype()) {
        case ncclFloat32: launch(input.dataAs<float>()); break;
        case ncclFloat16: launch(input.dataAs<__half>()); break;
        case ncclBfloat16: launch(input.dataAs<__nv_bfloat16>()); break;
        default: return ncclInvalidArgument;
      }
      cudaResult = cudaGetLastError();
    } else {
      auto launch = [&](const auto* typedInput) {
        if (config.hierarchical && !config.subAdd) {
          launchHierarchicalQuant(output.dataAs<int8_t>(), typedInput, input,
                                  config, context);
        } else {
          launchQuant(output.dataAs<int8_t>(), typedInput, input, config,
                      context.stream());
        }
      };
      switch (input.datatype()) {
        case ncclFloat32: launch(input.dataAs<float>()); break;
        case ncclFloat16: launch(input.dataAs<__half>()); break;
        case ncclBfloat16: launch(input.dataAs<__nv_bfloat16>()); break;
        default: return ncclInvalidArgument;
      }
      cudaResult = cudaGetLastError();
    }
    if (cudaResult != cudaSuccess) return coccl::fromCuda(cudaResult);
    return passthrough
        ? output.commit(input.elements(), input.datatype(), input.chunks())
        : output.commitBytes(requiredBytes, input.chunks());
  }

  static coccl::Status decompress(const coccl::Input& input,
                                  coccl::Output& output,
                                  coccl::Context& context) {
    const Config& config = context.config<Config>();
    if (config.subAdd) {
      size_t paramsBytes = 0;
      if (!parameterBytes(output.datatype(), config.quantType, &paramsBytes)) {
        return ncclInvalidArgument;
      }

      Sdp4BitState* state = nullptr;
      coccl::Status result = context.instance(&state);
      if (result != ncclSuccess) return result;
      size_t requiredBytes = 0;
      const size_t datatypeBytes = coccl::dataTypeSize(output.datatype());
      if (datatypeBytes == 0 ||
          !coccl::checkedMultiply(output.elements(), datatypeBytes,
                                  &requiredBytes)) {
        return ncclInvalidArgument;
      }
      coccl::Buffer persistent;
      result = context.persistent(0, requiredBytes, &persistent);
      if (result != ncclSuccess) return result;
      if (state->shardParams != persistent.data() ||
          state->datatype != output.datatype()) {
        state->shardParams = persistent.data();
        state->datatype = output.datatype();
        state->initialized = false;
      }

      cudaError_t cudaResult = cudaSuccess;
      if (!state->initialized) {
        if (input.bytes() < requiredBytes) return ncclInvalidArgument;
        cudaResult = cudaMemcpyAsync(output.data(), input.data(), requiredBytes,
                                     cudaMemcpyDeviceToDevice,
                                     context.stream());
        if (cudaResult == cudaSuccess) {
          cudaResult = cudaMemcpyAsync(
              state->shardParams, input.data(), requiredBytes,
              cudaMemcpyDeviceToDevice, context.stream());
        }
        if (cudaResult == cudaSuccess) state->initialized = true;
      } else {
        auto launch = [&](auto* typedOutput) {
          using T = typename std::remove_pointer<decltype(typedOutput)>::type;
          launch_fused_dequant_add_cuda(
              typedOutput, static_cast<const T*>(state->shardParams),
              input.dataAs<int8_t>(), config.outputBits(), config.quantType,
              config.outputGroups(),
              (int64_t)(output.elements() / output.chunks()),
              (int)output.chunks(), context.stream());
        };
        switch (output.datatype()) {
          case ncclFloat32: launch(output.dataAs<float>()); break;
          case ncclFloat16: launch(output.dataAs<__half>()); break;
          case ncclBfloat16: launch(output.dataAs<__nv_bfloat16>()); break;
          default: return ncclInvalidArgument;
        }
        cudaResult = cudaGetLastError();
        if (cudaResult == cudaSuccess) {
          cudaResult = cudaMemcpyAsync(
              state->shardParams, output.data(), requiredBytes,
              cudaMemcpyDeviceToDevice, context.stream());
        }
      }
      return coccl::fromCuda(cudaResult);
    }

    auto launch = [&](auto* typedOutput) {
      launchDequant(typedOutput, input, output, config, context.stream());
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
    size_t paramsBytes = 0;
    if (reduceChunks == 0 || input.chunks() % reduceChunks != 0 ||
        input.elements() % reduceChunks != 0 ||
        !parameterBytes(output.datatype(), config.quantType, &paramsBytes)) {
      return ncclInvalidArgument;
    }

    const int groupCount = config.outputGroups();
    const int quantBits = config.outputBits();
    const size_t groups = output.elements() == 0
        ? 0
        : 1 + (output.elements() - 1) / (size_t)groupCount;
    const size_t groupBytes =
        (size_t)groupCount / (size_t)(8 / quantBits);
    size_t chunkBytes = 0;
    if (groups == 0 ||
        !coccl::checkedMultiply(groups, groupBytes, &chunkBytes) ||
        chunkBytes > INT64_MAX || reduceChunks > INT_MAX) {
      return ncclInvalidArgument;
    }

    auto launch = [&](auto* typedOutput) {
      if (config.hadamard) {
        launch_dequant_reduce_ht(
            typedOutput, input.dataAs<int8_t>(), nullptr,
            (int)reduceChunks, quantBits, config.quantType,
            (int64_t)chunkBytes, (int)groupBytes, context.stream());
      } else {
        launch_dequant_reduce(
            typedOutput, input.dataAs<int8_t>(), nullptr,
            (int)reduceChunks, quantBits, config.quantType,
            (int64_t)chunkBytes, (int)groupBytes, context.stream());
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
    const Config& config = context.config<Config>();
    const size_t reduceChunks = context.reduceChunks();
    size_t paramsBytes = 0;
    if (reduceChunks == 0 || input.chunks() % reduceChunks != 0 ||
        input.elements() % reduceChunks != 0 || reduceChunks > INT_MAX ||
        config.hadamard ||
        config.quantType != quantize::Type::Symmetric ||
        !parameterBytes(context.originalDatatype(), config.quantType,
                        &paramsBytes)) {
      return ncclInvalidArgument;
    }

    const int inGroupCount = config.inputGroups(false);
    const int outGroupCount = config.outputGroups(false);
    const int inQuantBits = config.inputBits();
    const int outQuantBits = config.outputBits();
    const size_t inGroupBytes =
        (size_t)inGroupCount / (size_t)(8 / inQuantBits);
    const size_t outGroupBytes =
        (size_t)outGroupCount / (size_t)(8 / outQuantBits);
    size_t encodedInputGroupBytes = 0;
    if (!coccl::checkedAdd(inGroupBytes, paramsBytes,
                           &encodedInputGroupBytes)) {
      return ncclInvalidArgument;
    }

    const size_t inputElements = input.elements() / reduceChunks;
    const size_t inputGroups = inputElements == 0
        ? 0
        : 1 + (inputElements - 1) / encodedInputGroupBytes;
    size_t originalElements = 0;
    if (inputGroups == 0 ||
        !coccl::checkedMultiply(inputGroups, (size_t)inGroupCount,
                                &originalElements)) {
      return ncclInvalidArgument;
    }
    const size_t outputGroups =
        1 + (originalElements - 1) / (size_t)outGroupCount;
    size_t encodedOutputGroupBytes = 0;
    size_t inputChunkBytes = 0;
    size_t outputBytes = 0;
    if (!coccl::checkedAdd(outGroupBytes, paramsBytes,
                           &encodedOutputGroupBytes) ||
        !coccl::checkedMultiply(inputGroups, inGroupBytes,
                                &inputChunkBytes) ||
        !coccl::checkedMultiply(outputGroups, encodedOutputGroupBytes,
                                &outputBytes) ||
        inputGroups > INT_MAX || outputGroups > INT_MAX ||
        inputChunkBytes > INT64_MAX || outputBytes > output.capacityBytes()) {
      return ncclInvalidArgument;
    }

    launch_dequant_reduce_quant(
        output.dataAs<int8_t>(), nullptr, input.dataAs<int8_t>(), nullptr,
        (int)reduceChunks, inQuantBits, outQuantBits, config.quantType,
        (int)outputGroups, (int)outGroupBytes, (int)inputGroups,
        (int)inGroupBytes, (int64_t)inputChunkBytes, (int)paramsBytes,
        context.stream());
    cudaError_t cudaResult = cudaGetLastError();
    if (cudaResult != cudaSuccess) return coccl::fromCuda(cudaResult);
    return output.commitBytes(outputBytes, input.chunks() / reduceChunks);
  }
};

}  // namespace

COCCL_REGISTER_COMPRESSOR("sdp4bit", Sdp4BitCompressor);
