#include "core/compression/compress.h"

#include "core/memory/coccl_buffer_management.h"
#include "core/compression/reduce_extend.h"

#include <limits>

namespace {

bool checkedMultiply(size_t lhs, size_t rhs, size_t* result) {
  if (result == nullptr || (lhs != 0 && rhs > SIZE_MAX / lhs)) return false;
  *result = lhs * rhs;
  return true;
}

ncclResult_t typedBytes(size_t elements, ncclDataType_t datatype,
                        size_t* bytes) {
  const int typeBytes = ncclTypeSize(datatype);
  return typeBytes <= 0 ||
                 !checkedMultiply(elements, (size_t)typeBytes, bytes)
      ? ncclInvalidArgument
      : ncclSuccess;
}

bool isRawPassthrough(const cocclCompressorView& input) {
  return input.datatype == COCCL_COMPRESSOR_RAW_PASSTHROUGH;
}

ncclResult_t copyInputAsRawPassthrough(
    const cocclCompressorView& input,
    cocclCompressorView* output, cudaStream_t stream) {
  if (output == nullptr || output->data == nullptr || input.data == nullptr ||
      input.chunks == 0 || input.bytes > output->capacityBytes ||
      input.bytes % input.chunks != 0) {
    return ncclInvalidArgument;
  }
  if (output->data != input.data) {
    cudaError_t result = cudaMemcpyAsync(
        output->data, input.data, input.bytes, cudaMemcpyDeviceToDevice,
        stream);
    if (result != cudaSuccess) return ncclUnhandledCudaError;
  }
  output->bytes = input.bytes;
  output->elements = input.bytes;
  output->chunks = input.chunks;
  output->datatype = COCCL_COMPRESSOR_RAW_PASSTHROUGH;
  return ncclSuccess;
}

ncclResult_t copyRawPassthroughToOutput(
    const cocclCompressorView& input,
    cocclCompressorView* output, cudaStream_t stream) {
  size_t outputBytes = 0;
  if (!isRawPassthrough(input) || output == nullptr ||
      input.data == nullptr || output->data == nullptr ||
      input.chunks == 0 ||
      output->chunks != input.chunks || input.elements != input.bytes ||
      input.bytes % input.chunks != 0 ||
      output->elements % output->chunks != 0 ||
      typedBytes(output->elements, output->datatype, &outputBytes) !=
          ncclSuccess ||
      input.bytes != outputBytes || outputBytes > output->capacityBytes) {
    return ncclInvalidArgument;
  }
  if (output->data != input.data) {
    cudaError_t result = cudaMemcpyAsync(
        output->data, input.data, outputBytes, cudaMemcpyDeviceToDevice,
        stream);
    if (result != cudaSuccess) return ncclUnhandledCudaError;
  }
  output->bytes = outputBytes;
  return ncclSuccess;
}

ncclResult_t releaseWorkspace(cocclBufferHandle* buffer,
                              cudaStream_t stream, ncclResult_t ret) {
  ncclResult_t releaseResult = cocclReleaseBuffer(buffer, stream);
  return ret == ncclSuccess ? releaseResult : ret;
}

}  // namespace

ncclResult_t ncclCompress(
    const cocclCompressorHandle& compressor,
    const cocclCompressorView& input,
    cocclCompressorView* output, int rank, cudaStream_t stream) {
  NCCLCHECK(cocclExecuteCompressor(
      compressor, cocclCompressorOperationCompress, input, output, rank, 0,
      ncclInt8, 0, stream));
  return !cocclCompressorSupports(
             compressor, cocclCompressorCapabilityFramed) &&
          output->bytes > input.bytes
      ? copyInputAsRawPassthrough(input, output, stream)
      : ncclSuccess;
}

ncclResult_t ncclDecompress(
    const cocclCompressorHandle& compressor,
    const cocclCompressorView& input,
    cocclCompressorView* output, cudaStream_t stream) {
  if (isRawPassthrough(input)) {
    return copyRawPassthroughToOutput(input, output, stream);
  }
  return cocclExecuteCompressor(
      compressor, cocclCompressorOperationDecompress, input, output, 0, 0,
      output == nullptr ? ncclInt8 : output->datatype,
      output == nullptr ? 0 : output->elements, stream);
}

ncclResult_t ncclDecompressReduce(
    const cocclCompressorHandle& compressor, ncclComm_t ownerComm,
    const cocclCompressorView& input,
    cocclCompressorView* output, size_t reduceChunks,
    cudaStream_t stream) {
  if (ownerComm == nullptr || output == nullptr || reduceChunks == 0) {
    return ncclInvalidArgument;
  }
  if (!compressor) return ncclInvalidArgument;
  if (!isRawPassthrough(input) && cocclCompressorSupports(
          compressor, cocclCompressorCapabilityDecompressReduce)) {
    return cocclExecuteCompressor(
        compressor, cocclCompressorOperationDecompressReduce, input, output,
        0, reduceChunks, output->datatype, output->elements, stream);
  }

  size_t decompressedElements = 0;
  size_t decompressedBytes = 0;
  if (!checkedMultiply(output->elements, reduceChunks,
                       &decompressedElements) ||
      typedBytes(decompressedElements, output->datatype,
                 &decompressedBytes) != ncclSuccess) {
    return ncclInvalidArgument;
  }

  ncclResult_t ret = ncclSuccess;
  cocclBufferHandle workspace = {};
  NCCLCHECKGOTO(cocclGetUnregisteredBuffer(
                    ownerComm, decompressedBytes, &workspace),
                ret, exit);
  {
    cocclCompressorView decompressed = {
        workspace.ptr, workspace.bytes, decompressedBytes,
        decompressedElements, input.chunks, output->datatype};
    NCCLCHECKGOTO(ncclDecompress(
                      compressor, input, &decompressed, stream),
                  ret, exit);
  }
  NCCLCHECKGOTO(
      ncclReduceChunk(workspace.ptr, output->elements, output->data,
                      output->datatype, reduceChunks, stream),
      ret, exit);
  output->bytes = output->elements * (size_t)ncclTypeSize(output->datatype);
  output->chunks = input.chunks / reduceChunks;

exit:
  return workspace.ptr == nullptr
      ? ret
      : releaseWorkspace(&workspace, stream, ret);
}

ncclResult_t ncclDecompReduceComp(
    const cocclCompressorHandle& compressor, ncclComm_t ownerComm,
    const cocclCompressorView& input,
    cocclCompressorView* output, size_t reduceChunks,
    ncclDataType_t originalDatatype, size_t originalElements,
    cudaStream_t stream) {
  if (ownerComm == nullptr || output == nullptr || reduceChunks == 0 ||
      originalElements == 0 || input.chunks == 0 ||
      input.chunks % reduceChunks != 0) {
    return ncclInvalidArgument;
  }
  if (!compressor) return ncclInvalidArgument;
  const size_t outputChunks = input.chunks / reduceChunks;
  if (output->chunks != outputChunks ||
      originalElements % outputChunks != 0) {
    return ncclInvalidArgument;
  }
  size_t reducedBytes = 0;
  NCCLCHECK(typedBytes(originalElements, originalDatatype, &reducedBytes));
  if (!isRawPassthrough(input) && cocclCompressorSupports(
          compressor,
          cocclCompressorCapabilityDecompressReduceCompress)) {
    size_t fusedBound = 0;
    NCCLCHECK(cocclGetCompressorEncodedSizeBound(
        compressor, cocclCompressorOperationDecompressReduceCompress,
        originalElements, outputChunks, originalDatatype, &fusedBound));
    if (fusedBound <= reducedBytes) {
      if (fusedBound > output->capacityBytes) return ncclInvalidUsage;
      ncclResult_t result = cocclExecuteCompressor(
          compressor, cocclCompressorOperationDecompressReduceCompress,
          input, output, 0, reduceChunks, originalDatatype,
          originalElements, stream);
      return result == ncclSuccess && output->chunks != outputChunks
          ? ncclInvalidUsage
          : result;
    }
  }

  size_t decompressedElements = 0;
  size_t decompressedBytes = 0;
  if (!checkedMultiply(originalElements, reduceChunks,
                       &decompressedElements) ||
      typedBytes(decompressedElements, originalDatatype,
                 &decompressedBytes) != ncclSuccess) {
    return ncclInvalidArgument;
  }

  ncclResult_t ret = ncclSuccess;
  cocclBufferHandle workspace = {};
  NCCLCHECKGOTO(cocclGetUnregisteredBuffer(
                    ownerComm, decompressedBytes, &workspace),
                ret, exit);
  {
    cocclCompressorView decompressed = {
        workspace.ptr, workspace.bytes, decompressedBytes,
        decompressedElements, input.chunks, originalDatatype};
    NCCLCHECKGOTO(ncclDecompress(
                      compressor, input, &decompressed, stream),
                  ret, exit);
  }
  NCCLCHECKGOTO(
      ncclReduceChunk(workspace.ptr, originalElements, workspace.ptr,
                      originalDatatype, reduceChunks, stream),
      ret, exit);
  {
    const cocclCompressorView reduced = {
        workspace.ptr, reducedBytes, reducedBytes, originalElements,
        outputChunks,
        originalDatatype};
    NCCLCHECKGOTO(ncclCompress(
                      compressor, reduced, output, 0, stream),
                  ret, exit);
    if (output->chunks != outputChunks) ret = ncclInvalidUsage;
  }

exit:
  return workspace.ptr == nullptr
      ? ret
      : releaseWorkspace(&workspace, stream, ret);
}
