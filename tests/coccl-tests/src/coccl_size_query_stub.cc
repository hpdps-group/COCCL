#include "coccl_size_query_stub.h"

#include "core/compression/compress.h"

namespace {

struct SizeQueryState {
  size_t compressNumerator = 1;
  size_t compressDenominator = 1;
  size_t drcNumerator = 1;
  size_t drcDenominator = 1;
  bool supportsDrc = false;
  bool framed = false;
  bool fusedSwizzle = false;
  bool sameCompressor = false;
  cocclSizeQueryObservation compress = {};
  cocclSizeQueryObservation drc = {};
};

SizeQueryState state;

size_t datatypeBytes(ncclDataType_t datatype) {
  switch (datatype) {
    case ncclInt8:
    case ncclUint8:
      return 1;
    case ncclFloat16:
    case ncclBfloat16:
      return 2;
    case ncclInt32:
    case ncclUint32:
    case ncclFloat32:
      return 4;
    case ncclInt64:
    case ncclUint64:
    case ncclFloat64:
      return 8;
    default:
      return 0;
  }
}

}  // namespace

void cocclResetSizeQueryStub() {
  state = {};
}

void cocclConfigureSizeQueryStub(
    size_t compressNumerator, size_t compressDenominator,
    size_t drcNumerator, size_t drcDenominator, bool supportsDrc) {
  state = {};
  state.compressNumerator = compressNumerator;
  state.compressDenominator = compressDenominator;
  state.drcNumerator = drcNumerator;
  state.drcDenominator = drcDenominator;
  state.supportsDrc = supportsDrc;
}

void cocclConfigureFramedSizeQueryStub(bool framed) {
  state.framed = framed;
}

void cocclConfigureFusedSwizzleStub(bool fusedSwizzle) {
  state.fusedSwizzle = fusedSwizzle;
}

void cocclConfigureSameCompressorStub(bool sameCompressor) {
  state.sameCompressor = sameCompressor;
}

const cocclSizeQueryObservation& cocclCompressQueryObservation() {
  return state.compress;
}

const cocclSizeQueryObservation& cocclDrcQueryObservation() {
  return state.drc;
}

static ncclResult_t encodedSizeBound(
    cocclCompressorOperation operation,
    size_t elements, size_t chunks, ncclDataType_t datatype,
    size_t* encodedBytes) {
  const bool drc =
      operation == cocclCompressorOperationDecompressReduceCompress;
  cocclSizeQueryObservation& observation = drc ? state.drc : state.compress;
  ++observation.calls;
  observation.elements = elements;
  observation.chunks = chunks;
  observation.datatype = datatype;

  const size_t typeBytes = datatypeBytes(datatype);
  const size_t numerator = drc ? state.drcNumerator : state.compressNumerator;
  const size_t denominator =
      drc ? state.drcDenominator : state.compressDenominator;
  if (elements == 0 || chunks == 0 || elements % chunks != 0 ||
      typeBytes == 0 || denominator == 0) {
    return ncclInvalidArgument;
  }
  const size_t elementsPerChunk = elements / chunks;
  if (elementsPerChunk > SIZE_MAX / typeBytes) return ncclInvalidArgument;
  const size_t rawBytesPerChunk = elementsPerChunk * typeBytes;
  if (rawBytesPerChunk > SIZE_MAX / numerator) return ncclInvalidArgument;
  const size_t encodedBytesPerChunk =
      rawBytesPerChunk * numerator / denominator;
  if (encodedBytesPerChunk == 0 ||
      encodedBytesPerChunk > SIZE_MAX / chunks) {
    return ncclInvalidArgument;
  }
  *encodedBytes = encodedBytesPerChunk * chunks;
  return ncclSuccess;
}

ncclResult_t cocclGetCompressorEncodedSizeBound(
    void*, cocclCompressorOperation operation,
    size_t elements, size_t chunks, ncclDataType_t datatype,
    size_t* encodedBytes) {
  return encodedSizeBound(
      operation, elements, chunks, datatype, encodedBytes);
}

bool cocclCompressorSupports(
    void*, cocclCompressorCapability capability) {
  if (capability == cocclCompressorCapabilityFramed) return state.framed;
  if (capability ==
      cocclCompressorCapabilityFusedHierarchicalSwizzle) {
    return state.fusedSwizzle;
  }
  return capability ==
             cocclCompressorCapabilityDecompressReduceCompress &&
      state.supportsDrc;
}

const cocclCompressorPlugin* cocclCompressorDescriptor(void* compressor) {
  static cocclCompressorPlugin descriptor = {};
  return state.sameCompressor
      ? &descriptor
      : reinterpret_cast<const cocclCompressorPlugin*>(compressor);
}

ncclResult_t cocclResolveCompressorPolicy(
    cocclTrainingRole, cocclPolicyKey,
    cocclResolvedCompressorPolicy* resolved) {
  static int compressor;
  resolved->compressor = &compressor;
  resolved->thresholdBytes = 0;
  return ncclSuccess;
}
