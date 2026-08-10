#include "compressor_plugin/compressor_plugin.h"
#include "zfp.h"
#include "zfp/internal/zfp/macros.h"

#include <climits>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

struct ZfpConfig {
  int rate = 4;
};

zfp_type zfpType(ncclDataType_t datatype) {
  switch (datatype) {
    case ncclFloat32: return zfp_type_float;
    case ncclFloat16: return zfp_type_float16;
    case ncclBfloat16: return zfp_type_bfloat16;
    default: return zfp_type_none;
  }
}

bool configureStream(zfp_stream* stream, zfp_type type, int rate) {
  if (stream == nullptr || type == zfp_type_none ||
      zfp_stream_set_rate(stream, rate, type, 1, zfp_false) == 0 ||
      !zfp_stream_set_execution(stream, zfp_exec_cuda)) {
    return false;
  }
  return true;
}

struct ZfpCompressor {
  using Config = ZfpConfig;

  static coccl::Status configure(coccl::ConfigReader& reader, Config& config,
                                 const coccl::ConfigContext&) {
    return reader.get("rate", config.rate, 1, 64).finish();
  }

  static coccl::Status compress(const coccl::Input& input,
                                coccl::Output& output,
                                coccl::Context& context) {
    const zfp_type type = zfpType(input.datatype());
    const Config& config = context.config<Config>();
    if (type == zfp_type_none || input.elementsPerChunk() == 0 ||
        input.chunks() > INT_MAX) {
      return ncclInvalidArgument;
    }

    zfp_stream* stream = zfp_stream_open(nullptr);
    zfp_field* field = zfp_field_alloc();
    if (stream == nullptr || field == nullptr ||
        !configureStream(stream, type, config.rate)) {
      zfp_field_free(field);
      zfp_stream_close(stream);
      return ncclInternalError;
    }
    cudaStream_t cudaStream = context.stream();
    stream->exec.params = &cudaStream;

    zfp_field_set_type(field, type);
    zfp_field_set_size_1d(field, input.elementsPerChunk());
    const size_t maxChunkBytes = zfp_stream_maximum_size(stream, field);
    size_t maxBytes = 0;
    if (maxChunkBytes == 0 ||
        !coccl::checkedMultiply(maxChunkBytes, input.chunks(), &maxBytes)) {
      zfp_field_free(field);
      zfp_stream_close(stream);
      return ncclInvalidArgument;
    }
    if (coccl::shouldPassthrough(input, maxBytes)) {
      zfp_field_free(field);
      zfp_stream_close(stream);
      return output.passthrough(input, context.stream());
    }
    if (maxBytes > output.capacityBytes()) {
      zfp_field_free(field);
      zfp_stream_close(stream);
      return ncclInvalidArgument;
    }

    const size_t inputChunkBytes = input.bytes() / input.chunks();
    size_t compressedChunkBytes = 0;
    coccl::Status result = ncclSuccess;
    for (size_t chunk = 0; chunk < input.chunks(); ++chunk) {
      const size_t outputOffset = chunk * compressedChunkBytes;
      void* outputPtr = static_cast<char*>(output.data()) + outputOffset;
      bitstream* bits = stream_open(outputPtr, maxChunkBytes);
      if (bits == nullptr) {
        result = ncclInternalError;
        break;
      }

      zfp_stream_set_bit_stream(stream, bits);
      zfp_stream_rewind(stream);
      zfp_field_set_pointer(
          field, const_cast<char*>(static_cast<const char*>(input.data())) +
                     chunk * inputChunkBytes);
      const size_t bytes = zfp_compress(stream, field);
      stream_close(bits);

      if (bytes == 0 || (chunk != 0 && bytes != compressedChunkBytes)) {
        result = ncclInternalError;
        break;
      }
      if (chunk == 0) compressedChunkBytes = bytes;
    }

    zfp_field_free(field);
    zfp_stream_close(stream);
    if (result != ncclSuccess) return result;

    size_t compressedBytes = 0;
    if (!coccl::checkedMultiply(compressedChunkBytes, input.chunks(),
                                &compressedBytes)) {
      return ncclInvalidArgument;
    }
    return output.commitBytes(compressedBytes, input.chunks());
  }

  static coccl::Status decompress(const coccl::Input& input,
                                  coccl::Output& output,
                                  coccl::Context& context) {
    const zfp_type type = zfpType(output.datatype());
    const Config& config = context.config<Config>();
    if (type == zfp_type_none || output.elements() == 0 ||
        output.chunks() == 0 || input.chunks() != output.chunks()) {
      return ncclInvalidArgument;
    }

    zfp_stream* stream = zfp_stream_open(nullptr);
    zfp_field* field = zfp_field_alloc();
    if (stream == nullptr || field == nullptr ||
        !configureStream(stream, type, config.rate)) {
      zfp_field_free(field);
      zfp_stream_close(stream);
      return ncclInternalError;
    }
    cudaStream_t cudaStream = context.stream();
    stream->exec.params = &cudaStream;

    const size_t inputChunkBytes = input.bytes() / input.chunks();
    const size_t outputChunkElements = output.elements() / output.chunks();
    const size_t outputTypeBytes = coccl::dataTypeSize(output.datatype());
    size_t outputChunkBytes = 0;
    if (inputChunkBytes == 0 || outputTypeBytes == 0 ||
        !coccl::checkedMultiply(outputChunkElements, outputTypeBytes,
                                &outputChunkBytes)) {
      zfp_field_free(field);
      zfp_stream_close(stream);
      return ncclInvalidArgument;
    }

    zfp_field_set_type(field, type);
    zfp_field_set_size_1d(field, outputChunkElements);
    coccl::Status result = ncclSuccess;
    for (size_t chunk = 0; chunk < input.chunks(); ++chunk) {
      bitstream* bits = stream_open(
          const_cast<char*>(static_cast<const char*>(input.data())) +
              chunk * inputChunkBytes,
          inputChunkBytes);
      if (bits == nullptr) {
        result = ncclInternalError;
        break;
      }

      zfp_stream_set_bit_stream(stream, bits);
      zfp_stream_rewind(stream);
      zfp_field_set_pointer(
          field, static_cast<char*>(output.data()) + chunk * outputChunkBytes);
      const size_t bytes = zfp_decompress(stream, field);
      stream_close(bits);
      if (bytes == 0) {
        result = ncclInternalError;
        break;
      }
    }

    zfp_field_free(field);
    zfp_stream_close(stream);
    return result;
  }
};

}  // namespace

COCCL_REGISTER_COMPRESSOR("zfp", ZfpCompressor);
