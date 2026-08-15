#include "compressor_plugin/coccl_compressor_plugin.h"
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

bool maximumCompressedBytes(zfp_stream* stream, zfp_field* field,
                            zfp_type type, size_t elements,
                            size_t* maxBytes) {
  if (stream == nullptr || field == nullptr || type == zfp_type_none ||
      elements == 0 || maxBytes == nullptr) {
    return false;
  }
  zfp_field_set_type(field, type);
  zfp_field_set_size_1d(field, elements);
  *maxBytes = zfp_stream_maximum_size(stream, field);
  return *maxBytes != 0;
}

struct ZfpCompressor {
  using Config = ZfpConfig;

  static coccl::Status configure(coccl::ConfigReader& reader, Config& config,
                                 const coccl::ConfigContext&) {
    return reader.get("rate", config.rate, 1, 64).finish();
  }

  static coccl::Status encodedSizeBound(
      const coccl::Shape& input, size_t* encodedBytes,
      const coccl::SizeContext& context) {
    if (context.operation() != cocclCompressorOperationCompress ||
        encodedBytes == nullptr || input.chunks() > INT_MAX) {
      return context.operation() == cocclCompressorOperationCompress
          ? ncclInvalidArgument : ncclInvalidUsage;
    }
    const zfp_type type = zfpType(input.datatype());
    const Config& config = context.config<Config>();
    zfp_stream* stream = zfp_stream_open(nullptr);
    zfp_field* field = zfp_field_alloc();
    if (stream == nullptr || field == nullptr ||
        !configureStream(stream, type, config.rate)) {
      zfp_field_free(field);
      zfp_stream_close(stream);
      return ncclInternalError;
    }
    const bool valid = maximumCompressedBytes(
        stream, field, type, input.elements(), encodedBytes);
    zfp_field_free(field);
    zfp_stream_close(stream);
    return valid ? ncclSuccess : ncclInvalidArgument;
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

    size_t maxBytes = 0;
    if (!maximumCompressedBytes(
            stream, field, type, input.elements(), &maxBytes)) {
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

    bitstream* bits = stream_open(output.data(), maxBytes);
    if (bits == nullptr) {
      zfp_field_free(field);
      zfp_stream_close(stream);
      return ncclInternalError;
    }
    zfp_stream_set_bit_stream(stream, bits);
    zfp_stream_rewind(stream);
    zfp_field_set_pointer(field, const_cast<void*>(input.data()));
    const size_t compressedBytes = zfp_compress(stream, field);
    stream_close(bits);

    zfp_field_free(field);
    zfp_stream_close(stream);
    if (compressedBytes == 0) return ncclInternalError;
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

    if (input.bytes() == 0) {
      zfp_field_free(field);
      zfp_stream_close(stream);
      return ncclInvalidArgument;
    }

    zfp_field_set_type(field, type);
    zfp_field_set_size_1d(field, output.elements());
    bitstream* bits = stream_open(const_cast<void*>(input.data()),
                                  input.bytes());
    if (bits == nullptr) {
      zfp_field_free(field);
      zfp_stream_close(stream);
      return ncclInternalError;
    }
    zfp_stream_set_bit_stream(stream, bits);
    zfp_stream_rewind(stream);
    zfp_field_set_pointer(field, output.data());
    const size_t decompressedBytes = zfp_decompress(stream, field);
    stream_close(bits);

    zfp_field_free(field);
    zfp_stream_close(stream);
    return decompressedBytes == 0 ? ncclInternalError : ncclSuccess;
  }
};

}  // namespace

COCCL_REGISTER_COMPRESSOR("zfp", ZfpCompressor);
