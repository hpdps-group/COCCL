#include "pipeline/coccl_pipeline_stage.h"

#include "pipeline/coccl_pipeline_layout.h"
#include "primitives/coccl_primitives_internal.h"

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>

// Host-side dispatch for one non-in-place pipeline stage.
namespace {

bool cocclStageCheckedMultiply(size_t lhs, size_t rhs, size_t* result) {
  if (result == nullptr || (lhs != 0 && rhs > SIZE_MAX / lhs)) return false;
  *result = lhs * rhs;
  return true;
}

bool cocclStageCheckedAdd(size_t lhs, size_t rhs, size_t* result) {
  if (result == nullptr || rhs > SIZE_MAX - lhs) return false;
  *result = lhs + rhs;
  return true;
}

bool cocclStageBuffersOverlap(const void* input, size_t inputBytes,
                              const void* output, size_t outputBytes) {
  const uintptr_t inputBegin = (uintptr_t)input;
  const uintptr_t outputBegin = (uintptr_t)output;
  if (inputBytes > UINTPTR_MAX - inputBegin ||
      outputBytes > UINTPTR_MAX - outputBegin) {
    return true;
  }
  const uintptr_t inputEnd = inputBegin + inputBytes;
  const uintptr_t outputEnd = outputBegin + outputBytes;
  return inputBegin < outputEnd && outputBegin < inputEnd;
}

using cocclPipelineStageFn = ncclResult_t (*)(
    const cocclPipelineStageContext* context,
    const cocclPipelineStage* stage, cocclPipelineEdge* edge,
    const cocclPipelineStageOutput* output, cudaStream_t stream);

ncclResult_t cocclEnsureFrameMetadataCapacity(
    cocclPipelineFrameResources* resources, size_t frames) {
  if (resources == nullptr || frames == 0) return ncclInvalidArgument;
  if (resources->metadataCapacity >= frames) return ncclSuccess;

  size_t bytes = 0;
  if (!cocclStageCheckedMultiply(
          frames, sizeof(cocclCompressorFrameMetadata), &bytes)) {
    return ncclInvalidArgument;
  }
  cocclCompressorFrameMetadata* sendMetadata = nullptr;
  cocclCompressorFrameMetadata* recvMetadata = nullptr;
  cudaError_t error = cudaHostAlloc(
      (void**)&sendMetadata, bytes, cudaHostAllocPortable);
  if (error != cudaSuccess) return ncclUnhandledCudaError;
  error = cudaHostAlloc((void**)&recvMetadata, bytes, cudaHostAllocPortable);
  if (error != cudaSuccess) {
    cudaFreeHost(sendMetadata);
    return ncclUnhandledCudaError;
  }
  if (resources->sendMetadata != nullptr) {
    cudaFreeHost(resources->sendMetadata);
  }
  if (resources->recvMetadata != nullptr) {
    cudaFreeHost(resources->recvMetadata);
  }
  resources->sendMetadata = sendMetadata;
  resources->recvMetadata = recvMetadata;
  resources->metadataCapacity = frames;
  return ncclSuccess;
}

ncclResult_t cocclEnsureFrameExchangeCapacity(
    cocclPipelineFrameResources* resources, size_t exchanges) {
  if (resources == nullptr || exchanges == 0) return ncclInvalidArgument;
  if (resources->exchangeCapacity >= exchanges) return ncclSuccess;
  if (exchanges > SIZE_MAX / sizeof(cocclFrameExchange)) {
    return ncclInvalidArgument;
  }
  void* storage = realloc(
      resources->exchanges, exchanges * sizeof(cocclFrameExchange));
  if (storage == nullptr) return ncclSystemError;
  resources->exchanges = static_cast<cocclFrameExchange*>(storage);
  resources->exchangeCapacity = exchanges;
  return ncclSuccess;
}

ncclResult_t cocclReadFrameMetadata(
    const cocclPipelineStageContext* context,
    const cocclPipelineEdge* edge,
    const cocclPipelineStageOutput* output, size_t outputFrames,
    cudaStream_t stream) {
  if (context->frameResources == nullptr || edge->frameMetadata == nullptr ||
      output->frameMetadata == nullptr || edge->logicalChunks == 0 ||
      outputFrames == 0) {
    return ncclInvalidArgument;
  }
  const size_t capacity = edge->logicalChunks > outputFrames
      ? edge->logicalChunks : outputFrames;
  NCCLCHECK(cocclEnsureFrameMetadataCapacity(
      context->frameResources, capacity));
  size_t sendBytes = 0;
  size_t recvBytes = 0;
  if (!cocclStageCheckedMultiply(
          edge->logicalChunks, sizeof(cocclCompressorFrameMetadata),
          &sendBytes) ||
      !cocclStageCheckedMultiply(
          outputFrames, sizeof(cocclCompressorFrameMetadata), &recvBytes)) {
    return ncclInvalidArgument;
  }
  CUDACHECK(cudaMemcpyAsync(
      context->frameResources->sendMetadata, edge->frameMetadata, sendBytes,
      cudaMemcpyDeviceToHost, stream));
  CUDACHECK(cudaMemcpyAsync(
      context->frameResources->recvMetadata, output->frameMetadata, recvBytes,
      cudaMemcpyDeviceToHost, stream));
  CUDACHECK(cudaStreamSynchronize(stream));
  return ncclSuccess;
}

ncclResult_t cocclRunCompressStage(
    const cocclPipelineStageContext* context,
    const cocclPipelineStage* /* stage */, cocclPipelineEdge* edge,
    const cocclPipelineStageOutput* stageOutput, cudaStream_t stream) {
  const cocclCompressorDataView input = {
      edge->ptr, edge->bytes, edge->totalElements, edge->logicalChunks,
      edge->datatype, edge->frameMetadata, edge->frameStrideBytes};
  cocclCompressorOutputView output = {
      stageOutput->ptr, stageOutput->capacityBytes, 0, 0,
      edge->logicalChunks, ncclInt8, stageOutput->frameMetadata,
      stageOutput->frameStrideBytes};
  NCCLCHECK(ncclCompress(
      context->compressor, input, &output, context->ownerComm->rank, stream));
  edge->ptr = output.data;
  edge->bytes = output.bytes;
  edge->totalElements = output.elements;
  edge->datatype = output.datatype;
  edge->logicalChunks = output.chunks;
  edge->frameMetadata = output.frameMetadata;
  edge->frameStrideBytes = output.frameStrideBytes;
  return ncclSuccess;
}

ncclResult_t cocclRunAllToAllStage(
    const cocclPipelineStageContext* context,
    const cocclPipelineStage* stage, cocclPipelineEdge* edge,
    const cocclPipelineStageOutput* output, cudaStream_t stream) {
  if (stage->comm->nRanks <= 0 ||
      edge->bytes % (size_t)stage->comm->nRanks != 0 ||
      edge->bytes > output->capacityBytes) {
    return ncclInvalidArgument;
  }
  if (edge->frameMetadata != nullptr) {
    NCCLCHECK(cocclPreparePipelineFrameExchange(
        context, stage, edge, output, stream));
    return cocclCommitPipelineFrameExchange(
        context, stage, edge, output, stream);
  }
  if (output->frameMetadata != nullptr || output->frameStrideBytes != 0) {
    return ncclInvalidArgument;
  }
  const size_t sendBytes =
      edge->bytes / (size_t)stage->comm->nRanks;
  NCCLCHECK(ncclAllToAllNaive(edge->ptr, output->ptr, sendBytes, ncclUint8,
                              stage->comm, stream));
  edge->ptr = output->ptr;
  return ncclSuccess;
}

ncclResult_t cocclRunAllGatherStage(
    const cocclPipelineStageContext* context,
    const cocclPipelineStage* stage, cocclPipelineEdge* edge,
    const cocclPipelineStageOutput* output, cudaStream_t stream) {
  size_t gatheredBytes = 0;
  size_t gatheredElements = 0;
  size_t gatheredChunks = 0;
  if (stage->comm->nRanks <= 0 ||
      !cocclStageCheckedMultiply(edge->bytes,
                                 (size_t)stage->comm->nRanks,
                                 &gatheredBytes) ||
      !cocclStageCheckedMultiply(edge->totalElements,
                                 (size_t)stage->comm->nRanks,
                                 &gatheredElements) ||
      !cocclStageCheckedMultiply(edge->logicalChunks,
                                 (size_t)stage->comm->nRanks,
                                 &gatheredChunks) ||
      gatheredBytes > output->capacityBytes) {
    return ncclInvalidArgument;
  }

  if (edge->frameMetadata != nullptr) {
    NCCLCHECK(cocclPreparePipelineFrameExchange(
        context, stage, edge, output, stream));
    return cocclCommitPipelineFrameExchange(
        context, stage, edge, output, stream);
  }
  if (output->frameMetadata != nullptr || output->frameStrideBytes != 0) {
    return ncclInvalidArgument;
  }

  struct ncclInfo info = {
      ncclFuncAllGather, "AllGather", edge->ptr, output->ptr,
      edge->bytes, ncclUint8, ncclSum, 0, stage->comm, stream,
      ALLGATHER_CHUNKSTEPS, ALLGATHER_SLICESTEPS};
  NCCLCHECK(ncclEnqueueCheck(&info));

  edge->ptr = output->ptr;
  edge->bytes = gatheredBytes;
  edge->totalElements = gatheredElements;
  edge->logicalChunks = gatheredChunks;
  return ncclSuccess;
}

ncclResult_t cocclRunPackStage(
    const cocclPipelineStageContext* context,
    const cocclPipelineStage* /* stage */, cocclPipelineEdge* edge,
    const cocclPipelineStageOutput* output, cudaStream_t stream) {
  if (edge->frameMetadata != nullptr || output->frameMetadata != nullptr ||
      edge->bytes > output->capacityBytes) {
    return ncclInvalidArgument;
  }
  const size_t sliceBytes = edge->bytes / edge->logicalChunks;
  NCCLCHECK(cocclLaunchPackSlice(
      edge->ptr, context->rawChunkBytes, output->ptr, sliceBytes,
      edge->logicalChunks, stream));
  edge->ptr = output->ptr;
  return ncclSuccess;
}

ncclResult_t cocclRunUnpackStage(
    const cocclPipelineStageContext* context,
    const cocclPipelineStage* /* stage */, cocclPipelineEdge* edge,
    const cocclPipelineStageOutput* output, cudaStream_t stream) {
  if (edge->frameMetadata != nullptr || output->frameMetadata != nullptr ||
      edge->bytes > output->capacityBytes) {
    return ncclInvalidArgument;
  }
  const size_t sliceBytes = edge->bytes / edge->logicalChunks;
  NCCLCHECK(cocclLaunchUnpackSlice(
      edge->ptr, output->ptr, context->rawChunkBytes, sliceBytes,
      edge->logicalChunks, stream));
  edge->ptr = output->ptr;
  return ncclSuccess;
}

ncclResult_t cocclRunDecompReduceCompStage(
    const cocclPipelineStageContext* context,
    const cocclPipelineStage* stage, cocclPipelineEdge* edge,
    const cocclPipelineStageOutput* stageOutput, cudaStream_t stream) {
  if (stage->reduceChunks == 0 ||
      edge->logicalChunks % stage->reduceChunks != 0 ||
      edge->totalElements % stage->reduceChunks != 0 ||
      edge->bytes % stage->reduceChunks != 0) {
    return ncclInvalidArgument;
  }

  const size_t outputChunks =
      edge->logicalChunks / stage->reduceChunks;
  size_t reductionElements = 0;
  if (!cocclStageCheckedMultiply(context->rawSliceCount, outputChunks,
                                 &reductionElements)) {
    return ncclInvalidArgument;
  }

  const cocclCompressorDataView input = {
      edge->ptr, edge->bytes, edge->totalElements, edge->logicalChunks,
      edge->datatype, edge->frameMetadata, edge->frameStrideBytes};
  cocclCompressorOutputView output = {
      stageOutput->ptr, stageOutput->capacityBytes, 0, 0, outputChunks,
      ncclInt8, stageOutput->frameMetadata,
      stageOutput->frameStrideBytes};
  NCCLCHECK(ncclDecompReduceComp(
      context->compressor, context->ownerComm, input, &output,
      stage->reduceChunks, context->rawDatatype, reductionElements,
      stream));
  edge->ptr = output.data;
  edge->bytes = output.bytes;
  edge->totalElements = output.elements;
  edge->datatype = output.datatype;
  edge->logicalChunks = output.chunks;
  edge->frameMetadata = output.frameMetadata;
  edge->frameStrideBytes = output.frameStrideBytes;
  return ncclSuccess;
}

ncclResult_t cocclRunDecompressReduceStage(
    const cocclPipelineStageContext* context,
    const cocclPipelineStage* stage, cocclPipelineEdge* edge,
    const cocclPipelineStageOutput* stageOutput, cudaStream_t stream) {
  if (stage->reduceChunks == 0 ||
      edge->logicalChunks % stage->reduceChunks != 0 ||
      edge->totalElements % stage->reduceChunks != 0 ||
      edge->bytes % stage->reduceChunks != 0) {
    return ncclInvalidArgument;
  }

  const size_t remainingChunks =
      edge->logicalChunks / stage->reduceChunks;
  size_t reduceChunkCount = 0;
  if (!cocclStageCheckedMultiply(context->rawSliceCount, remainingChunks,
                                 &reduceChunkCount)) {
    return ncclInvalidArgument;
  }
  const cocclCompressorDataView input = {
      edge->ptr, edge->bytes, edge->totalElements, edge->logicalChunks,
      edge->datatype, edge->frameMetadata, edge->frameStrideBytes};
  cocclCompressorOutputView output = {
      stageOutput->ptr, stageOutput->capacityBytes, 0, reduceChunkCount,
      remainingChunks, context->rawDatatype, nullptr, 0};
  NCCLCHECK(ncclDecompressReduce(
      context->compressor, context->ownerComm, input, &output,
      stage->reduceChunks, stream));
  edge->ptr = output.data;
  edge->bytes = output.bytes;
  edge->totalElements = output.elements;
  edge->datatype = output.datatype;
  edge->logicalChunks = output.chunks;
  edge->frameMetadata = nullptr;
  edge->frameStrideBytes = 0;
  return ncclSuccess;
}

ncclResult_t cocclRunDecompressStage(
    const cocclPipelineStageContext* context,
    const cocclPipelineStage* /* stage */, cocclPipelineEdge* edge,
    const cocclPipelineStageOutput* stageOutput, cudaStream_t stream) {
  if (edge->logicalChunks == 0 ||
      edge->totalElements % edge->logicalChunks != 0) {
    return ncclInvalidArgument;
  }
  size_t outputElements = 0;
  size_t outputBytes = 0;
  if (!cocclStageCheckedMultiply(context->rawSliceCount,
                                 edge->logicalChunks, &outputElements) ||
      !cocclStageCheckedMultiply(
          outputElements, (size_t)ncclTypeSize(context->rawDatatype),
          &outputBytes)) {
    return ncclInvalidArgument;
  }
  const cocclCompressorDataView input = {
      edge->ptr, edge->bytes, edge->totalElements, edge->logicalChunks,
      edge->datatype, edge->frameMetadata, edge->frameStrideBytes};
  cocclCompressorOutputView output = {
      stageOutput->ptr, stageOutput->capacityBytes, 0, outputElements,
      edge->logicalChunks, context->rawDatatype, nullptr, 0};
  NCCLCHECK(ncclDecompress(
      context->compressor, input, &output, stream));
  edge->ptr = output.data;
  edge->bytes = output.bytes;
  edge->totalElements = output.elements;
  edge->datatype = output.datatype;
  edge->logicalChunks = output.chunks;
  edge->frameMetadata = nullptr;
  edge->frameStrideBytes = 0;
  return ncclSuccess;
}

constexpr int kPipelineStageKindCount = cocclPipelineStageUnpack + 1;
static_assert(cocclPipelineStageCompress == 0 &&
                  cocclPipelineStageAllToAll == 1 &&
                  cocclPipelineStageAllGather == 2 &&
                  cocclPipelineStageDecompReduceComp == 3 &&
                  cocclPipelineStageDecompressReduce == 4 &&
                  cocclPipelineStageDecompress == 5 &&
                  cocclPipelineStagePack == 6 &&
                  cocclPipelineStageUnpack == 7,
              "pipeline stage handlers require contiguous stage kinds");

const cocclPipelineStageFn
    cocclPipelineStageHandlers[kPipelineStageKindCount] = {
        cocclRunCompressStage,
        cocclRunAllToAllStage,
        cocclRunAllGatherStage,
        cocclRunDecompReduceCompStage,
        cocclRunDecompressReduceStage,
        cocclRunDecompressStage,
        cocclRunPackStage,
        cocclRunUnpackStage,
};

}  // namespace

ncclResult_t cocclPipelineStageLayoutSpans(
    const cocclPipelineStageContext* context,
    const cocclPipelineEdge* edge, size_t* contiguousBytes,
    size_t* pitchedBytes) {
  if (context == nullptr || edge == nullptr || contiguousBytes == nullptr ||
      pitchedBytes == nullptr || context->rawSliceCount == 0 ||
      context->rawChunkBytes == 0 || edge->bytes == 0 ||
      edge->totalElements == 0 || edge->logicalChunks == 0 ||
      edge->datatype != context->rawDatatype ||
      edge->bytes % edge->logicalChunks != 0) {
    return ncclInvalidArgument;
  }

  const int typeBytes = ncclTypeSize(context->rawDatatype);
  size_t expectedElements = 0;
  size_t expectedBytes = 0;
  if (typeBytes <= 0 ||
      !cocclStageCheckedMultiply(context->rawSliceCount,
                                 edge->logicalChunks, &expectedElements) ||
      !cocclStageCheckedMultiply(expectedElements, (size_t)typeBytes,
                                 &expectedBytes) ||
      edge->totalElements != expectedElements ||
      edge->bytes != expectedBytes) {
    return ncclInvalidArgument;
  }

  const size_t sliceBytes = edge->bytes / edge->logicalChunks;
  size_t pitchedPrefixBytes = 0;
  size_t spanBytes = 0;
  if (sliceBytes == 0 || context->rawChunkBytes < sliceBytes ||
      !cocclStageCheckedMultiply(edge->logicalChunks - 1,
                                 context->rawChunkBytes,
                                 &pitchedPrefixBytes) ||
      !cocclStageCheckedAdd(pitchedPrefixBytes, sliceBytes, &spanBytes)) {
    return ncclInvalidArgument;
  }
  *contiguousBytes = edge->bytes;
  *pitchedBytes = spanBytes;
  return ncclSuccess;
}

ncclResult_t cocclPipelineStageOutputChunks(
    const cocclPipelineStage& stage, size_t inputChunks,
    size_t* outputChunks) {
  if (outputChunks == nullptr || inputChunks == 0) {
    return ncclInvalidArgument;
  }

  switch (stage.kind) {
    case cocclPipelineStageCompress:
    case cocclPipelineStageAllToAll:
    case cocclPipelineStageDecompress:
    case cocclPipelineStagePack:
    case cocclPipelineStageUnpack:
      *outputChunks = inputChunks;
      return ncclSuccess;
    case cocclPipelineStageAllGather:
      if (stage.comm == nullptr || stage.comm->nRanks <= 0 ||
          !cocclStageCheckedMultiply(inputChunks,
                                     (size_t)stage.comm->nRanks,
                                     outputChunks)) {
        return ncclInvalidArgument;
      }
      return ncclSuccess;
    case cocclPipelineStageDecompReduceComp:
    case cocclPipelineStageDecompressReduce:
      if (stage.reduceChunks == 0 ||
          inputChunks % stage.reduceChunks != 0) {
        return ncclInvalidArgument;
      }
      *outputChunks = inputChunks / stage.reduceChunks;
      return ncclSuccess;
    default:
      return ncclInvalidArgument;
  }
}

bool cocclPipelineStageUsesFrameExchange(
    const cocclPipelineStage& stage, const cocclPipelineEdge& edge) {
  return edge.frameMetadata != nullptr &&
      (stage.kind == cocclPipelineStageAllToAll ||
       stage.kind == cocclPipelineStageAllGather);
}

ncclResult_t cocclPreparePipelineFrameExchange(
    const cocclPipelineStageContext* context,
    const cocclPipelineStage* stage, const cocclPipelineEdge* edge,
    const cocclPipelineStageOutput* output, cudaStream_t stream) {
  if (context == nullptr || context->frameResources == nullptr ||
      stage == nullptr || stage->comm == nullptr || edge == nullptr ||
      output == nullptr || edge->frameMetadata == nullptr ||
      output->frameMetadata == nullptr || edge->frameStrideBytes == 0 ||
      output->frameStrideBytes != edge->frameStrideBytes ||
      stage->comm->nRanks <= 0) {
    return ncclInvalidArgument;
  }
  if (cocclStageBuffersOverlap(
          edge->ptr, edge->bytes, output->ptr, output->capacityBytes)) {
    return ncclInvalidUsage;
  }

  if (stage->kind == cocclPipelineStageAllToAll) {
    const size_t ranks = (size_t)stage->comm->nRanks;
    if (edge->logicalChunks % ranks != 0 ||
        edge->bytes % ranks != 0 ||
        edge->bytes > output->capacityBytes) {
      return ncclInvalidArgument;
    }
    size_t metadataBytes = 0;
    if (!cocclStageCheckedMultiply(
            edge->logicalChunks / ranks,
            sizeof(cocclCompressorFrameMetadata), &metadataBytes)) {
      return ncclInvalidArgument;
    }
    return ncclAllToAllNaive(
        edge->frameMetadata, output->frameMetadata, metadataBytes,
        ncclUint8, stage->comm, stream);
  }

  if (stage->kind == cocclPipelineStageAllGather) {
    size_t gatheredBytes = 0;
    size_t gatheredElements = 0;
    size_t gatheredChunks = 0;
    if (!cocclStageCheckedMultiply(
            edge->bytes, (size_t)stage->comm->nRanks, &gatheredBytes) ||
        !cocclStageCheckedMultiply(
            edge->totalElements, (size_t)stage->comm->nRanks,
            &gatheredElements) ||
        !cocclStageCheckedMultiply(
            edge->logicalChunks, (size_t)stage->comm->nRanks,
            &gatheredChunks) ||
        gatheredBytes > output->capacityBytes) {
      return ncclInvalidArgument;
    }
    size_t metadataBytes = 0;
    if (!cocclStageCheckedMultiply(
            edge->logicalChunks, sizeof(cocclCompressorFrameMetadata),
            &metadataBytes)) {
      return ncclInvalidArgument;
    }
    struct ncclInfo metadataInfo = {
        ncclFuncAllGather, "AllGather", edge->frameMetadata,
        output->frameMetadata, metadataBytes, ncclUint8, ncclSum, 0,
        stage->comm, stream, ALLGATHER_CHUNKSTEPS, ALLGATHER_SLICESTEPS};
    return ncclEnqueueCheck(&metadataInfo);
  }
  return ncclInvalidArgument;
}

ncclResult_t cocclCommitPipelineFrameExchange(
    const cocclPipelineStageContext* context,
    const cocclPipelineStage* stage, cocclPipelineEdge* edge,
    const cocclPipelineStageOutput* output, cudaStream_t stream) {
  if (context == nullptr || context->frameResources == nullptr ||
      stage == nullptr || stage->comm == nullptr || edge == nullptr ||
      output == nullptr || edge->frameMetadata == nullptr ||
      output->frameMetadata == nullptr || edge->frameStrideBytes == 0 ||
      output->frameStrideBytes != edge->frameStrideBytes ||
      stage->comm->nRanks <= 0) {
    return ncclInvalidArgument;
  }

  size_t outputFrames = edge->logicalChunks;
  if (stage->kind == cocclPipelineStageAllGather &&
      !cocclStageCheckedMultiply(
          outputFrames, (size_t)stage->comm->nRanks, &outputFrames)) {
    return ncclInvalidArgument;
  }
  if (stage->kind != cocclPipelineStageAllToAll &&
      stage->kind != cocclPipelineStageAllGather) {
    return ncclInvalidArgument;
  }
  NCCLCHECK(cocclReadFrameMetadata(
      context, edge, output, outputFrames, stream));
  NCCLCHECK(cocclEnsureFrameExchangeCapacity(
      context->frameResources, outputFrames));

  size_t exchangeCount = 0;
  if (stage->kind == cocclPipelineStageAllToAll) {
    NCCLCHECK(cocclBuildAllToAllFrameExchanges(
        edge->ptr, output->ptr, edge->logicalChunks,
        edge->frameStrideBytes, stage->comm->nRanks,
        context->frameResources->sendMetadata,
        context->frameResources->recvMetadata,
        context->frameResources->exchanges,
        context->frameResources->exchangeCapacity, &exchangeCount));
  } else {
    NCCLCHECK(cocclBuildAllGatherFrameExchanges(
        edge->ptr, output->ptr, edge->logicalChunks,
        edge->frameStrideBytes, stage->comm->nRanks,
        context->frameResources->sendMetadata,
        context->frameResources->recvMetadata,
        context->frameResources->exchanges,
        context->frameResources->exchangeCapacity, &exchangeCount));
  }
  NCCLCHECK(cocclCommitFrameExchange(
      context->frameResources->exchanges, exchangeCount,
      stage->comm, stream));

  if (stage->kind == cocclPipelineStageAllGather) {
    if (!cocclStageCheckedMultiply(
            edge->bytes, (size_t)stage->comm->nRanks, &edge->bytes) ||
        !cocclStageCheckedMultiply(
            edge->totalElements, (size_t)stage->comm->nRanks,
            &edge->totalElements)) {
      return ncclInvalidArgument;
    }
    edge->logicalChunks = outputFrames;
  }
  edge->ptr = output->ptr;
  edge->frameMetadata = output->frameMetadata;
  edge->frameStrideBytes = output->frameStrideBytes;
  return ncclSuccess;
}

ncclResult_t cocclExecutePipelineStage(
    const cocclPipelineStageContext* context,
    const cocclPipelineStage* stage, cocclPipelineEdge* edge,
    const cocclPipelineStageOutput* output, cudaStream_t stream) {
  if (context == nullptr || context->ownerComm == nullptr ||
      stage == nullptr || edge == nullptr || edge->ptr == nullptr ||
      edge->bytes == 0 || output == nullptr || output->ptr == nullptr ||
      output->capacityBytes == 0 ||
      (edge->frameMetadata == nullptr) != (edge->frameStrideBytes == 0) ||
      (output->frameMetadata == nullptr) !=
          (output->frameStrideBytes == 0)) {
    return ncclInvalidArgument;
  }
  const int stageKind = (int)stage->kind;
  if (stageKind < 0 || stageKind >= kPipelineStageKindCount) {
    return ncclInvalidArgument;
  }
  size_t inputSpanBytes = edge->bytes;
  size_t outputSpanBytes = output->capacityBytes;
  if (stage->kind == cocclPipelineStagePack ||
      stage->kind == cocclPipelineStageUnpack) {
    size_t contiguousBytes = 0;
    size_t pitchedBytes = 0;
    NCCLCHECK(cocclPipelineStageLayoutSpans(
        context, edge, &contiguousBytes, &pitchedBytes));
    if (stage->kind == cocclPipelineStagePack) {
      inputSpanBytes = pitchedBytes;
      outputSpanBytes = contiguousBytes;
    } else {
      inputSpanBytes = contiguousBytes;
      outputSpanBytes = pitchedBytes;
    }
  }
  if (cocclStageBuffersOverlap(edge->ptr, inputSpanBytes, output->ptr,
                               outputSpanBytes)) {
    return ncclInvalidUsage;
  }

  const cocclPipelineStageFn handler =
      cocclPipelineStageHandlers[stageKind];
  return handler == nullptr
      ? ncclInvalidArgument
      : handler(context, stage, edge, output, stream);
}
