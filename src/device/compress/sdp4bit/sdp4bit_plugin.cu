#include "quantization.h"
#include "sdp4bit_config.h"
#include "compressor_plugin/coccl_compressor_plugin.h"

#include <climits>
#include <cuda_runtime.h>

cudaError_t launchQuantize(
    const void* input, void** output, size_t chunkElements,
    ncclDataType_t datatype, size_t* encodedChunkElements,
    ncclDataType_t* encodedDatatype, size_t chunks, int rank, void* config,
    cudaMemPool_t pool, cudaStream_t stream);
cudaError_t launchDequantize(
    void* output, const void* input, size_t outputChunkElements,
    ncclDataType_t outputDatatype, size_t inputChunkElements,
    ncclDataType_t inputDatatype, size_t chunks, void* config,
    cudaStream_t stream);
cudaError_t launchDequanReduce(
    void* output, const void* input, size_t inputChunkElements,
    ncclDataType_t inputDatatype, size_t outputChunkElements,
    ncclDataType_t outputDatatype, size_t chunks, void* config,
    cudaStream_t stream);
cudaError_t launchDequanReduceQuan(
    const void* input, void** output, size_t inputChunkElements,
    ncclDataType_t inputDatatype, size_t* outputChunkElements,
    ncclDataType_t* outputDatatype, size_t chunks, void* config,
    cudaMemPool_t pool, cudaStream_t stream);

namespace {

int inputBits(const sdp4bitConfig& config) {
  return config.inQuantBits == 0 ? config.quantBits : config.inQuantBits;
}

int outputBits(const sdp4bitConfig& config) {
  return config.outQuantBits == 0 ? config.quantBits : config.outQuantBits;
}

int inputGroups(const sdp4bitConfig& config) {
  int groups = config.inGroupCount == 0
      ? config.groupCount : config.inGroupCount;
  return config.hadamard && groups > 128 ? 128 : groups;
}

bool encodedBytes(const coccl::Input& input, const sdp4bitConfig& config,
                  size_t* bytes) {
  const int bits = inputBits(config);
  const int groupElements = inputGroups(config);
  if ((bits != 4 && bits != 8) || groupElements <= 0 ||
      input.chunks() == 0 || input.elements() % input.chunks() != 0) {
    return false;
  }
  const size_t elementsPerChunk = input.elementsPerChunk();
  const size_t groups =
      (elementsPerChunk + static_cast<size_t>(groupElements) - 1) /
      static_cast<size_t>(groupElements);
  const size_t payload =
      static_cast<size_t>(groupElements) / static_cast<size_t>(8 / bits);
  const size_t parameters = input.datatype() == ncclFloat32
      ? (config.quantType == quantize::Type::Symmetric ? 4u : 8u)
      : 8u;
  size_t chunkBytes = 0;
  return coccl::checkedAdd(payload, parameters, &chunkBytes) &&
      coccl::checkedMultiply(chunkBytes, groups, &chunkBytes) &&
      coccl::checkedMultiply(chunkBytes, input.chunks(), bytes);
}

struct Sdp4BitCompressor {
  using Config = sdp4bitConfig;

  static coccl::Status configure(coccl::ConfigReader& reader, Config& config,
                                 const coccl::ConfigContext& context) {
    config.intraAndInter = context.hierarchical();
    config.nodes = context.nodes();
    config.devicesPerNodes = context.devicesPerNode();
    coccl::Status result =
        reader.get("groupCount", config.groupCount, 1, INT_MAX)
            .get("quantBits", config.quantBits, 4, 8)
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
    const int inBits = inputBits(config);
    const int outBits = outputBits(config);
    return (inBits == 4 || inBits == 8) &&
                   (outBits == 4 || outBits == 8)
        ? ncclSuccess : ncclInvalidArgument;
  }

  static coccl::Status compress(const coccl::Input& input,
                                coccl::Output& output,
                                coccl::Context& context) {
    const Config& config = context.config<Config>();
    size_t requiredBytes = 0;
    if (!encodedBytes(input, config, &requiredBytes)) {
      return ncclInvalidArgument;
    }
    if (coccl::shouldPassthrough(input, requiredBytes)) {
      return output.passthrough(input, context.stream());
    }
    if (requiredBytes > output.capacityBytes()) return ncclInvalidArgument;

    void* encoded = output.data();
    size_t encodedChunkElements = 0;
    ncclDataType_t encodedDatatype = ncclInt8;
    cudaError_t result = launchQuantize(
        input.data(), &encoded, input.elementsPerChunk(), input.datatype(),
        &encodedChunkElements, &encodedDatatype, input.chunks(),
        context.rank(), const_cast<Config*>(&config), nullptr,
        context.stream());
    if (result != cudaSuccess) return coccl::fromCuda(result);
    return output.commit(encodedChunkElements * input.chunks(),
                         encodedDatatype, input.chunks());
  }

  static coccl::Status decompress(const coccl::Input& input,
                                  coccl::Output& output,
                                  coccl::Context& context) {
    if (coccl::isPassthrough(input)) {
      cudaError_t result = cudaMemcpyAsync(
          output.data(), input.data(), input.bytes(), cudaMemcpyDeviceToDevice,
          context.stream());
      return coccl::fromCuda(result);
    }
    const Config& config = context.config<Config>();
    return coccl::fromCuda(launchDequantize(
        output.data(), input.data(), output.elements() / output.chunks(),
        output.datatype(), input.elementsPerChunk(), input.datatype(),
        input.chunks(), const_cast<Config*>(&config), context.stream()));
  }

  static coccl::Status decompressReduce(const coccl::Input& input,
                                        coccl::Output& output,
                                        coccl::Context& context) {
    if (output.datatype() != ncclFloat32) return ncclInvalidArgument;
    const Config& config = context.config<Config>();
    return coccl::fromCuda(launchDequanReduce(
        output.data(), input.data(), input.elementsPerChunk(),
        input.datatype(), output.elements(), output.datatype(),
        context.reduceChunks(), const_cast<Config*>(&config),
        context.stream()));
  }

  static coccl::Status decompressReduceCompress(
      const coccl::Input& input, coccl::Output& output,
      coccl::Context& context) {
    const Config& config = context.config<Config>();
    void* encoded = output.data();
    size_t encodedChunkElements = 0;
    ncclDataType_t encodedDatatype = ncclInt8;
    cudaError_t result = launchDequanReduceQuan(
        input.data(), &encoded, input.elementsPerChunk(), input.datatype(),
        &encodedChunkElements, &encodedDatatype, context.reduceChunks(),
        const_cast<Config*>(&config), nullptr, context.stream());
    if (result != cudaSuccess) return coccl::fromCuda(result);
    return output.commit(encodedChunkElements, encodedDatatype, 1);
  }
};

}  // namespace

COCCL_REGISTER_COMPRESSOR("sdp4bit", Sdp4BitCompressor);
