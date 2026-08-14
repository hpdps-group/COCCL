#include "core/pipeline/coccl_pipeline_stage.h"

#include "core/compression/compress.h"
#include "core/pipeline/coccl_pipeline_layout.h"
#include "checks.h"
#include "collectives.h"
#include "comm.h"
#include "enqueue.h"

#include <stdint.h>
#include <stdlib.h>

// Host-side dispatch for one non-in-place pipeline stage.
namespace {

bool cocclStageCheckedMultiply(size_t lhs, size_t rhs, size_t* result) {
  if (result == nullptr || (lhs != 0 && rhs > SIZE_MAX / lhs)) return false;
  *result = lhs * rhs;
  return true;
}

bool cocclStageBuffersOverlap(const void* input, size_t inputBytes,
                              const void* output, size_t outputBytes) {
  const uintptr_t inputBegin = (uintptr_t)input;
  const uintptr_t outputBegin = (uintptr_t)output;
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
  const size_t capacity = edge->logicalChunks > outputFrames
      ? edge->logicalChunks : outputFrames;
  NCCLCHECK(cocclEnsureFrameMetadataCapacity(
      context->frameResources, capacity));
  const size_t sendBytes =
      edge->logicalChunks * sizeof(cocclCompressorFrameMetadata);
  const size_t recvBytes =
      outputFrames * sizeof(cocclCompressorFrameMetadata);
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
  const cocclCompressorView input = {
      edge->ptr, edge->bytes, edge->bytes, edge->totalElements,
      edge->logicalChunks, edge->datatype, edge->frameMetadata,
      edge->frameStrideBytes};
  cocclCompressorView output = {
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
  if (edge->frameMetadata != nullptr) {
    NCCLCHECK(cocclPreparePipelineFrameExchange(
        context, stage, edge, output, stream));
    return cocclCommitPipelineFrameExchange(
        context, stage, edge, output, stream);
  }
  const size_t sendBytes =
      edge->bytes / (size_t)stage->comm->nRanks;
  NCCLCHECK(ncclAllToAll(edge->ptr, output->ptr, sendBytes, ncclUint8,
                         stage->comm, stream));
  edge->ptr = output->ptr;
  return ncclSuccess;
}

ncclResult_t cocclRunAllGatherStage(
    const cocclPipelineStageContext* context,
    const cocclPipelineStage* stage, cocclPipelineEdge* edge,
    const cocclPipelineStageOutput* output, cudaStream_t stream) {
  if (edge->frameMetadata != nullptr) {
    NCCLCHECK(cocclPreparePipelineFrameExchange(
        context, stage, edge, output, stream));
    return cocclCommitPipelineFrameExchange(
        context, stage, edge, output, stream);
  }
  struct ncclInfo info = {
      ncclFuncAllGather, "AllGather", edge->ptr, output->ptr,
      edge->bytes, ncclUint8, ncclSum, 0, stage->comm, stream,
      ALLGATHER_CHUNKSTEPS, ALLGATHER_SLICESTEPS};
  NCCLCHECK(ncclEnqueueCheck(&info));

  edge->ptr = output->ptr;
  edge->bytes *= (size_t)stage->comm->nRanks;
  edge->totalElements *= (size_t)stage->comm->nRanks;
  edge->logicalChunks *= (size_t)stage->comm->nRanks;
  return ncclSuccess;
}

ncclResult_t cocclRunPackStage(
    const cocclPipelineStageContext* context,
    const cocclPipelineStage* /* stage */, cocclPipelineEdge* edge,
    const cocclPipelineStageOutput* output, cudaStream_t stream) {
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
  const size_t outputChunks =
      edge->logicalChunks / stage->reduceChunks;
  const size_t reductionElements = context->rawSliceCount * outputChunks;

  const cocclCompressorView input = {
      edge->ptr, edge->bytes, edge->bytes, edge->totalElements,
      edge->logicalChunks, edge->datatype, edge->frameMetadata,
      edge->frameStrideBytes};
  cocclCompressorView output = {
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
  const size_t remainingChunks =
      edge->logicalChunks / stage->reduceChunks;
  const size_t reduceChunkCount =
      context->rawSliceCount * remainingChunks;
  const cocclCompressorView input = {
      edge->ptr, edge->bytes, edge->bytes, edge->totalElements,
      edge->logicalChunks, edge->datatype, edge->frameMetadata,
      edge->frameStrideBytes};
  cocclCompressorView output = {
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
  const size_t outputElements =
      context->rawSliceCount * edge->logicalChunks;
  const cocclCompressorView input = {
      edge->ptr, edge->bytes, edge->bytes, edge->totalElements,
      edge->logicalChunks, edge->datatype, edge->frameMetadata,
      edge->frameStrideBytes};
  cocclCompressorView output = {
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

void cocclPipelineStageLayoutSpans(
    const cocclPipelineStageContext* context,
    const cocclPipelineEdge* edge, size_t* contiguousBytes,
    size_t* pitchedBytes) {
  const size_t sliceBytes = edge->bytes / edge->logicalChunks;
  *contiguousBytes = edge->bytes;
  *pitchedBytes =
      (edge->logicalChunks - 1) * context->rawChunkBytes + sliceBytes;
}

ncclResult_t cocclPipelineStageOutputChunks(
    const cocclPipelineStage& stage, size_t inputChunks,
    size_t* outputChunks) {
  switch (stage.kind) {
    case cocclPipelineStageCompress:
    case cocclPipelineStageAllToAll:
    case cocclPipelineStageDecompress:
    case cocclPipelineStagePack:
    case cocclPipelineStageUnpack:
      *outputChunks = inputChunks;
      return ncclSuccess;
    case cocclPipelineStageAllGather:
      if (!cocclStageCheckedMultiply(
              inputChunks, (size_t)stage.comm->nRanks, outputChunks)) {
        return ncclInvalidArgument;
      }
      return ncclSuccess;
    case cocclPipelineStageDecompReduceComp:
    case cocclPipelineStageDecompressReduce:
      *outputChunks = inputChunks / stage.reduceChunks;
      return ncclSuccess;
  }
  __builtin_unreachable();
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
  if (cocclStageBuffersOverlap(
          edge->ptr, edge->bytes, output->ptr, output->capacityBytes)) {
    return ncclInvalidUsage;
  }

  if (stage->kind == cocclPipelineStageAllToAll) {
    const size_t ranks = (size_t)stage->comm->nRanks;
    const size_t metadataBytes = edge->logicalChunks / ranks *
        sizeof(cocclCompressorFrameMetadata);
    return ncclAllToAll(
        edge->frameMetadata, output->frameMetadata, metadataBytes,
        ncclUint8, stage->comm, stream);
  }

  const size_t metadataBytes =
      edge->logicalChunks * sizeof(cocclCompressorFrameMetadata);
  struct ncclInfo metadataInfo = {
      ncclFuncAllGather, "AllGather", edge->frameMetadata,
      output->frameMetadata, metadataBytes, ncclUint8, ncclSum, 0,
      stage->comm, stream, ALLGATHER_CHUNKSTEPS, ALLGATHER_SLICESTEPS};
  return ncclEnqueueCheck(&metadataInfo);
}

ncclResult_t cocclCommitPipelineFrameExchange(
    const cocclPipelineStageContext* context,
    const cocclPipelineStage* stage, cocclPipelineEdge* edge,
    const cocclPipelineStageOutput* output, cudaStream_t stream) {
  size_t outputFrames = edge->logicalChunks;
  if (stage->kind == cocclPipelineStageAllGather) {
    outputFrames *= (size_t)stage->comm->nRanks;
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
    edge->bytes *= (size_t)stage->comm->nRanks;
    edge->totalElements *= (size_t)stage->comm->nRanks;
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
  const int stageKind = (int)stage->kind;
  size_t inputSpanBytes = edge->bytes;
  size_t outputSpanBytes = output->capacityBytes;
  if (stage->kind == cocclPipelineStagePack ||
      stage->kind == cocclPipelineStageUnpack) {
    size_t contiguousBytes = 0;
    size_t pitchedBytes = 0;
    cocclPipelineStageLayoutSpans(
        context, edge, &contiguousBytes, &pitchedBytes);
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

  return cocclPipelineStageHandlers[stageKind](
      context, stage, edge, output, stream);
}
