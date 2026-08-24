#include "core/compression/compress.h"

#include "comm.h"
#include "core/compression/reduce_extend.h"
#include "core/memory/coccl_buffer_management.h"

ncclResult_t ncclCompress(
    void* compressor, const cocclCompressorView& input,
    cocclCompressorView* output, int rank, cudaStream_t stream) {
  return cocclExecuteCompressor(
      compressor, cocclCompressorOperationCompress, input, output, rank, 0,
      input.datatype, input.elements, stream);
}

ncclResult_t ncclDecompress(
    void* compressor, const cocclCompressorView& input,
    cocclCompressorView* output, cudaStream_t stream) {
  if (input.datatype == COCCL_COMPRESSOR_RAW_PASSTHROUGH) {
    if (input.bytes > output->capacityBytes) return ncclInvalidUsage;
    CUDACHECK(cudaMemcpyAsync(output->data, input.data, input.bytes,
                              cudaMemcpyDeviceToDevice, stream));
    output->bytes = input.bytes;
    return ncclSuccess;
  }
  return cocclExecuteCompressor(
      compressor, cocclCompressorOperationDecompress, input, output, -1, 0,
      output->datatype, output->elements, stream);
}

ncclResult_t ncclDecompressReduce(
    void* compressor, ncclComm_t ownerComm,
    const cocclCompressorView& input, cocclCompressorView* output,
    size_t reduceChunks, cudaStream_t stream) {
  if (input.frameMetadata == nullptr &&
      input.datatype != COCCL_COMPRESSOR_RAW_PASSTHROUGH &&
      cocclCompressorSupports(
          compressor, cocclCompressorCapabilityDecompressReduce)) {
    return cocclExecuteCompressor(
        compressor, cocclCompressorOperationDecompressReduce, input, output,
        -1, reduceChunks, output->datatype, output->elements, stream);
  }

  const size_t decompressedElements = output->elements * reduceChunks;
  const size_t decompressedBytes =
      decompressedElements * (size_t)ncclTypeSize(output->datatype);
  cocclBufferHandle workspace = {};
  ncclResult_t result = cocclGetUnregisteredBuffer(
      ownerComm, decompressedBytes, stream, &workspace);
  if (result == ncclSuccess) {
    cocclCompressorView decompressed = {
        workspace.ptr, workspace.bytes, decompressedBytes,
        decompressedElements, input.chunks, output->datatype, nullptr, 0};
    result = ncclDecompress(compressor, input, &decompressed, stream);
  }
  if (result == ncclSuccess) {
    result = ncclReduceChunk(
        workspace.ptr, output->elements, output->data, output->datatype,
        reduceChunks, stream);
  }
  if (result == ncclSuccess) {
    output->bytes =
        output->elements * (size_t)ncclTypeSize(output->datatype);
    output->chunks = input.chunks / reduceChunks;
  }
  const ncclResult_t releaseResult = cocclReleaseBuffer(&workspace, stream);
  return result == ncclSuccess ? releaseResult : result;
}

ncclResult_t ncclDecompReduceComp(
    void* decoder, void* encoder, ncclComm_t ownerComm,
    const cocclCompressorView& input, cocclCompressorView* output,
    size_t reduceChunks, ncclDataType_t originalDatatype,
    size_t originalElements, cudaStream_t stream) {
  if (input.frameMetadata == nullptr &&
      input.datatype != COCCL_COMPRESSOR_RAW_PASSTHROUGH &&
      decoder == encoder &&
      cocclCompressorSupports(
          decoder, cocclCompressorCapabilityDecompressReduceCompress)) {
    return cocclExecuteCompressor(
        decoder, cocclCompressorOperationDecompressReduceCompress, input,
        output, -1, reduceChunks, originalDatatype, originalElements,
        stream);
  }

  const size_t decompressedElements = originalElements * reduceChunks;
  const size_t decompressedBytes =
      decompressedElements * (size_t)ncclTypeSize(originalDatatype);
  const size_t reducedBytes =
      originalElements * (size_t)ncclTypeSize(originalDatatype);
  cocclBufferHandle workspace = {};
  ncclResult_t result = cocclGetUnregisteredBuffer(
      ownerComm, decompressedBytes, stream, &workspace);
  if (result == ncclSuccess) {
    cocclCompressorView decompressed = {
        workspace.ptr, workspace.bytes, decompressedBytes,
        decompressedElements, input.chunks, originalDatatype, nullptr, 0};
    result = ncclDecompress(decoder, input, &decompressed, stream);
  }
  if (result == ncclSuccess) {
    result = ncclReduceChunk(
        workspace.ptr, originalElements, workspace.ptr, originalDatatype,
        reduceChunks, stream);
  }
  if (result == ncclSuccess) {
    const cocclCompressorView reduced = {
        workspace.ptr, reducedBytes, reducedBytes, originalElements,
        input.chunks / reduceChunks, originalDatatype, nullptr, 0};
    result = ncclCompress(
        encoder, reduced, output, ownerComm->rank, stream);
  }
  const ncclResult_t releaseResult = cocclReleaseBuffer(&workspace, stream);
  return result == ncclSuccess ? releaseResult : result;
}
