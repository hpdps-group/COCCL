#include "coccl_m11_size_query_stub.h"

#include "compress.h"

namespace {

struct SizeQueryState {
  size_t compressNumerator = 1;
  size_t compressDenominator = 1;
  size_t drcNumerator = 1;
  size_t drcDenominator = 1;
  bool supportsDrc = false;
  bool framed = false;
  bool fusedSwizzle = false;
  cocclM11SizeQueryObservation compress = {};
  cocclM11SizeQueryObservation drc = {};
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

void cocclM11ResetSizeQueryStub() {
  state = {};
}

void cocclM11ConfigureSizeQueryStub(
    size_t compressNumerator, size_t compressDenominator,
    size_t drcNumerator, size_t drcDenominator, bool supportsDrc) {
  state = {};
  state.compressNumerator = compressNumerator;
  state.compressDenominator = compressDenominator;
  state.drcNumerator = drcNumerator;
  state.drcDenominator = drcDenominator;
  state.supportsDrc = supportsDrc;
}

void cocclM14ConfigureFramedSizeQueryStub(bool framed) {
  state.framed = framed;
}

void cocclM15ConfigureFusedSwizzleStub(bool fusedSwizzle) {
  state.fusedSwizzle = fusedSwizzle;
}

const cocclM11SizeQueryObservation& cocclM11CompressQueryObservation() {
  return state.compress;
}

const cocclM11SizeQueryObservation& cocclM11DrcQueryObservation() {
  return state.drc;
}

ncclResult_t cocclGetCompressorEncodedSizeBound(
    cocclPolicyKey, cocclCompressorOperation operation,
    size_t elements, size_t chunks, ncclDataType_t datatype,
    size_t* encodedBytes) {
  const bool drc =
      operation == cocclCompressorOperationDecompressReduceCompress;
  cocclM11SizeQueryObservation& observation = drc ? state.drc : state.compress;
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

bool cocclCompressorPolicySupports(
    cocclPolicyKey, cocclCompressorCapability capability) {
  if (capability == cocclCompressorCapabilityFramed) return state.framed;
  if (capability ==
      cocclCompressorCapabilityFusedHierarchicalSwizzle) {
    return state.fusedSwizzle;
  }
  return capability ==
             cocclCompressorCapabilityDecompressReduceCompress &&
      state.supportsDrc;
}

bool cocclCompressorSupports(
    void*, cocclCompressorCapability capability) {
  return cocclCompressorPolicySupports({}, capability);
}

ncclResult_t cocclResolveCompressorPolicy(
    cocclPolicyKey, cocclResolvedCompressorPolicy* resolved) {
  static int compressor;
  resolved->compressor = &compressor;
  resolved->thresholdBytes = 0;
  return ncclSuccess;
}
