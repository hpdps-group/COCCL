#include "compression/compress.h"

#include "buffer/coccl_buffer_management.h"
#include "runtime/coccl_comm.h"
#include "compression/reduce_extend.h"

#include <limits>

namespace {

ncclResult_t resolveCompressorForExecution(
    ncclComm_t comm, cocclPolicyKey policy,
    cocclCompressorHandle* compressor) {
  return cocclCommGetCompressor(comm, policy, compressor);
}

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

bool isRawPassthrough(const cocclCompressorDataView& input) {
  return input.datatype == COCCL_COMPRESSOR_RAW_PASSTHROUGH;
}

ncclResult_t copyInputAsRawPassthrough(
    const cocclCompressorDataView& input,
    cocclCompressorOutputView* output, cudaStream_t stream) {
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
    const cocclCompressorDataView& input,
    cocclCompressorOutputView* output, cudaStream_t stream) {
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
    const cocclCompressorDataView& input,
    cocclCompressorOutputView* output, int rank, cudaStream_t stream) {
  NCCLCHECK(cocclExecuteCompressor(
      compressor, cocclCompressorOperationCompress, input, output, rank, 0,
      ncclInt8, 0, stream));
  return output->bytes > input.bytes
      ? copyInputAsRawPassthrough(input, output, stream)
      : ncclSuccess;
}

ncclResult_t ncclDecompress(
    const cocclCompressorHandle& compressor,
    const cocclCompressorDataView& input,
    cocclCompressorOutputView* output, cudaStream_t stream) {
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
    const cocclCompressorDataView& input,
    cocclCompressorOutputView* output, size_t reduceChunks,
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
    cocclCompressorOutputView decompressed = {
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
    const cocclCompressorDataView& input,
    cocclCompressorOutputView* output, size_t reduceChunks,
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
  if (!isRawPassthrough(input) && cocclCompressorSupports(
          compressor,
          cocclCompressorCapabilityDecompressReduceCompress)) {
    ncclResult_t result = cocclExecuteCompressor(
        compressor, cocclCompressorOperationDecompressReduceCompress, input,
        output, 0, reduceChunks, originalDatatype, originalElements, stream);
    return result == ncclSuccess && output->chunks != outputChunks
        ? ncclInvalidUsage
        : result;
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
    cocclCompressorOutputView decompressed = {
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
    size_t reducedBytes = 0;
    NCCLCHECKGOTO(
        typedBytes(originalElements, originalDatatype, &reducedBytes),
        ret, exit);
    const cocclCompressorDataView reduced = {
        workspace.ptr, reducedBytes, originalElements, outputChunks,
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

ncclResult_t ncclCompress(
    const void* orgbuff, void** compbuff, const size_t orgChunkCount,
    ncclDataType_t orgDatatype, size_t* compChunkCount,
    ncclDataType_t* compDatatype, const size_t numChunks, const int rank,
    ncclComm_t comm, cocclPolicyKey policy, cudaStream_t stream) {
  if (orgbuff == nullptr || compbuff == nullptr || *compbuff == nullptr ||
      compChunkCount == nullptr || compDatatype == nullptr ||
      numChunks == 0) {
    return ncclInvalidArgument;
  }
  cocclCompressorHandle compressor;
  NCCLCHECK(resolveCompressorForExecution(comm, policy, &compressor));

  size_t totalElements = 0;
  size_t totalBytes = 0;
  if (!checkedMultiply(orgChunkCount, numChunks, &totalElements)) {
    return ncclInvalidArgument;
  }
  NCCLCHECK(typedBytes(totalElements, orgDatatype, &totalBytes));
  const cocclCompressorDataView input = {
      orgbuff, totalBytes, totalElements, numChunks, orgDatatype};
  cocclCompressorOutputView output = {
      *compbuff, totalBytes, 0, 0, numChunks, ncclInt8};
  NCCLCHECK(ncclCompress(compressor, input, &output, rank, stream));
  if (output.elements % output.chunks != 0) return ncclInvalidArgument;
  *compbuff = output.data;
  *compChunkCount = output.elements / output.chunks;
  *compDatatype = output.datatype;
  return ncclSuccess;
}

ncclResult_t ncclDecompress(
    void* decompbuff, const void* compbuff,
    const size_t decompChunkCount, ncclDataType_t decompDatatype,
    const size_t compChunkCount, ncclDataType_t compDatatype,
    const size_t numChunks, ncclComm_t comm, cocclPolicyKey policy,
    cudaStream_t stream) {
  if (decompbuff == nullptr || compbuff == nullptr || numChunks == 0) {
    return ncclInvalidArgument;
  }
  cocclCompressorHandle compressor;
  NCCLCHECK(resolveCompressorForExecution(comm, policy, &compressor));

  size_t inputElements = 0;
  size_t inputBytes = 0;
  size_t outputElements = 0;
  size_t outputBytes = 0;
  if (!checkedMultiply(compChunkCount, numChunks, &inputElements) ||
      !checkedMultiply(decompChunkCount, numChunks, &outputElements)) {
    return ncclInvalidArgument;
  }
  NCCLCHECK(typedBytes(inputElements, compDatatype, &inputBytes));
  NCCLCHECK(typedBytes(outputElements, decompDatatype, &outputBytes));
  const cocclCompressorDataView input = {
      compbuff, inputBytes, inputElements, numChunks, compDatatype};
  cocclCompressorOutputView output = {
      decompbuff, outputBytes, 0, outputElements, numChunks,
      decompDatatype};
  return ncclDecompress(compressor, input, &output, stream);
}

ncclResult_t ncclDecompressReduce(
    void* reducebuff, const void* compbuff,
    const size_t compChunkCount, ncclDataType_t compDatatype,
    const size_t reduceChunkCount, ncclDataType_t reduceDataType,
    const size_t numChunks, ncclComm_t comm, cocclPolicyKey policy,
    cudaStream_t stream) {
  cocclCompressorHandle compressor;
  NCCLCHECK(resolveCompressorForExecution(comm, policy, &compressor));

  size_t inputElements = 0;
  size_t inputBytes = 0;
  size_t outputBytes = 0;
  if (!checkedMultiply(compChunkCount, numChunks, &inputElements)) {
    return ncclInvalidArgument;
  }
  NCCLCHECK(typedBytes(inputElements, compDatatype, &inputBytes));
  NCCLCHECK(typedBytes(reduceChunkCount, reduceDataType, &outputBytes));
  const cocclCompressorDataView input = {
      compbuff, inputBytes, inputElements, numChunks, compDatatype};
  cocclCompressorOutputView output = {
      reducebuff, outputBytes, 0, reduceChunkCount, 1, reduceDataType};
  return ncclDecompressReduce(
      compressor, comm, input, &output, numChunks, stream);
}

ncclResult_t ncclDecompReduceComp(
    const void* compbuff, void** recompbuff, const size_t orgChunkCount,
    ncclDataType_t orgDatatype, const size_t compChunkCount,
    ncclDataType_t compDatatype, size_t* reCompChunkCount,
    ncclDataType_t* reCompDatatype, const size_t numChunks,
    ncclComm_t comm, cocclPolicyKey policy, cudaStream_t stream) {
  if (recompbuff == nullptr || *recompbuff == nullptr ||
      reCompChunkCount == nullptr || reCompDatatype == nullptr) {
    return ncclInvalidArgument;
  }
  cocclCompressorHandle compressor;
  NCCLCHECK(resolveCompressorForExecution(comm, policy, &compressor));

  size_t inputElements = 0;
  size_t inputBytes = 0;
  size_t outputCapacity = 0;
  if (!checkedMultiply(compChunkCount, numChunks, &inputElements)) {
    return ncclInvalidArgument;
  }
  NCCLCHECK(typedBytes(inputElements, compDatatype, &inputBytes));
  NCCLCHECK(typedBytes(orgChunkCount, orgDatatype, &outputCapacity));
  const cocclCompressorDataView input = {
      compbuff, inputBytes, inputElements, numChunks, compDatatype};
  cocclCompressorOutputView output = {
      *recompbuff, outputCapacity, 0, 0, 1, ncclInt8};
  NCCLCHECK(ncclDecompReduceComp(
      compressor, comm, input, &output, numChunks, orgDatatype,
      orgChunkCount, stream));
  *recompbuff = output.data;
  *reCompChunkCount = output.elements;
  *reCompDatatype = output.datatype;
  return ncclSuccess;
}
