#include "core/pipeline/coccl_pipeline_internal.h"

#include "checks.h"
#include "collectives.h"
#include "comm.h"
#include "core/compression/compress.h"
#include "core/pipeline/coccl_pipeline_layout.h"
#include "debug.h"
#include "enqueue.h"

#include <stdlib.h>

namespace {

bool buffersOverlap(const void* first, size_t firstBytes,
                    const void* second, size_t secondBytes) {
  const uintptr_t firstBegin = reinterpret_cast<uintptr_t>(first);
  const uintptr_t secondBegin = reinterpret_cast<uintptr_t>(second);
  return firstBegin < secondBegin + secondBytes &&
      secondBegin < firstBegin + firstBytes;
}

bool useFramedAllGatherV(
    const cocclPipelineStage* stage,
    const cocclFrameExchange* exchanges, size_t exchangeCount) {
  if (stage->kind != cocclPipelineStageAllGather ||
      ncclParamAllgathervEnable() == 0 ||
      ncclParamEnqueueRearchEnable() != 0 || stage->comm->ccEnable) {
    return false;
  }
  if (stage->comm->nNodes > 1) return true;

  constexpr size_t kSingleNodeP2PFrameBytes = size_t{1} << 30;
  for (size_t i = 0; i < exchangeCount; ++i) {
    if (exchanges[i].recvBytes >= kSingleNodeP2PFrameBytes) return false;
  }
  return true;
}

ncclCollConfig_t communicationConfig(
    const cocclPipelineStageContext* context,
    const cocclPipelineStage* stage) {
  ncclCollConfig_t config = NCCL_COLLCONFIG_INITIALIZER;
  if (stage->config != nullptr) config = *stage->config;
  config.userProfilerTag = context->profilerTag;
  return config;
}

ncclResult_t ensureFrameMetadataCapacity(
    cocclPipelineFrameResources* resources, size_t frames) {
  if (resources->metadataCapacity >= frames) return ncclSuccess;
  const size_t bytes =
      frames * sizeof(cocclCompressorFrameMetadata);
  cocclCompressorFrameMetadata* sendMetadata = nullptr;
  cocclCompressorFrameMetadata* recvMetadata = nullptr;
  CUDACHECK(cudaHostAlloc(
      reinterpret_cast<void**>(&sendMetadata), bytes,
      cudaHostAllocPortable));
  const cudaError_t recvResult = cudaHostAlloc(
      reinterpret_cast<void**>(&recvMetadata), bytes,
      cudaHostAllocPortable);
  if (recvResult != cudaSuccess) {
    (void)cudaFreeHost(sendMetadata);
    return ncclUnhandledCudaError;
  }
  if (resources->sendMetadata != nullptr) {
    (void)cudaFreeHost(resources->sendMetadata);
  }
  if (resources->recvMetadata != nullptr) {
    (void)cudaFreeHost(resources->recvMetadata);
  }
  resources->sendMetadata = sendMetadata;
  resources->recvMetadata = recvMetadata;
  resources->metadataCapacity = frames;
  return ncclSuccess;
}

ncclResult_t ensureFrameExchangeCapacity(
    cocclPipelineFrameResources* resources, size_t count) {
  if (resources->exchangeCapacity >= count) return ncclSuccess;
  void* storage =
      realloc(resources->exchanges, count * sizeof(cocclFrameExchange));
  if (storage == nullptr) return ncclSystemError;
  resources->exchanges = static_cast<cocclFrameExchange*>(storage);
  resources->exchangeCapacity = count;
  return ncclSuccess;
}

ncclResult_t readFrameMetadata(
    const cocclPipelineStageContext* context,
    const cocclPipelineEdge* edge,
    const cocclPipelineStageOutput* output, size_t outputFrames,
    cudaStream_t stream) {
  const size_t capacity = edge->logicalChunks > outputFrames
      ? edge->logicalChunks : outputFrames;
  NCCLCHECK(ensureFrameMetadataCapacity(
      context->frameResources, capacity));
  CUDACHECK(cudaMemcpyAsync(
      context->frameResources->sendMetadata, edge->frameMetadata,
      edge->logicalChunks * sizeof(cocclCompressorFrameMetadata),
      cudaMemcpyDeviceToHost, stream));
  CUDACHECK(cudaMemcpyAsync(
      context->frameResources->recvMetadata, output->frameMetadata,
      outputFrames * sizeof(cocclCompressorFrameMetadata),
      cudaMemcpyDeviceToHost, stream));
  CUDACHECK(cudaStreamSynchronize(stream));
  return ncclSuccess;
}

ncclResult_t runCompress(const cocclPipelineStageContext* context,
                         const cocclPipelineStage* stage,
                         cocclPipelineEdge* edge,
                         const cocclPipelineStageOutput* output,
                         cudaStream_t stream) {
  const cocclCompressorView input = {
      edge->ptr, edge->bytes, edge->bytes, edge->totalElements,
      edge->logicalChunks, edge->datatype, edge->frameMetadata,
      edge->frameStrideBytes};
  cocclCompressorView encoded = {
      output->ptr, output->capacityBytes, 0, 0, edge->logicalChunks,
      ncclInt8, output->frameMetadata, output->frameStrideBytes};
  NCCLCHECK(ncclCompress(
      stage->compressor, input, &encoded, context->ownerComm->rank,
      stream));
  edge->ptr = encoded.data;
  edge->bytes = encoded.bytes;
  edge->totalElements = encoded.elements;
  edge->datatype = encoded.datatype;
  edge->logicalChunks = encoded.chunks;
  edge->compressor = stage->compressor;
  edge->frameMetadata = encoded.frameMetadata;
  edge->frameStrideBytes = encoded.frameStrideBytes;
  return ncclSuccess;
}

ncclResult_t runAllToAll(const cocclPipelineStageContext* context,
                         const cocclPipelineStage* stage,
                         cocclPipelineEdge* edge,
                         const cocclPipelineStageOutput* output,
                         cudaStream_t stream) {
  if (edge->frameMetadata != nullptr) {
    NCCLCHECK(cocclPreparePipelineFrameExchange(
        context, stage, edge, output, stream));
    return cocclCommitPipelineFrameExchange(
        context, stage, edge, output, stream);
  }
  const size_t sendBytes = edge->bytes / (size_t)stage->comm->nRanks;
  const ncclCollConfig_t config = communicationConfig(context, stage);
  NCCLCHECK(ncclAlltoAllConfig(
      edge->ptr, output->ptr, sendBytes, ncclUint8, stage->comm, stream,
      &config));
  edge->ptr = output->ptr;
  return ncclSuccess;
}

ncclResult_t runAllGather(const cocclPipelineStageContext* context,
                          const cocclPipelineStage* stage,
                          cocclPipelineEdge* edge,
                          const cocclPipelineStageOutput* output,
                          cudaStream_t stream) {
  if (edge->frameMetadata != nullptr) {
    NCCLCHECK(cocclPreparePipelineFrameExchange(
        context, stage, edge, output, stream));
    return cocclCommitPipelineFrameExchange(
        context, stage, edge, output, stream);
  }
  const ncclCollConfig_t config = communicationConfig(context, stage);
  NCCLCHECK(ncclAllGatherConfig(
      edge->ptr, output->ptr, edge->bytes, ncclUint8, stage->comm, stream,
      &config));
  const size_t ranks = (size_t)stage->comm->nRanks;
  edge->ptr = output->ptr;
  edge->bytes *= ranks;
  edge->totalElements *= ranks;
  edge->logicalChunks *= ranks;
  return ncclSuccess;
}

ncclResult_t runDecompress(const cocclPipelineStageContext* context,
                           const cocclPipelineStage*,
                           cocclPipelineEdge* edge,
                           const cocclPipelineStageOutput* output,
                           cudaStream_t stream) {
  const cocclCompressorView input = {
      edge->ptr, edge->bytes, edge->bytes, edge->totalElements,
      edge->logicalChunks, edge->datatype, edge->frameMetadata,
      edge->frameStrideBytes};
  cocclCompressorView decoded = {
      output->ptr, output->capacityBytes, 0,
      context->rawSliceCount * edge->logicalChunks,
      edge->logicalChunks, context->rawDatatype, nullptr, 0};
  NCCLCHECK(ncclDecompress(
      edge->compressor, input, &decoded, stream));
  edge->ptr = decoded.data;
  edge->bytes = decoded.bytes;
  edge->totalElements = decoded.elements;
  edge->datatype = decoded.datatype;
  edge->logicalChunks = decoded.chunks;
  edge->compressor = nullptr;
  edge->frameMetadata = nullptr;
  edge->frameStrideBytes = 0;
  return ncclSuccess;
}

ncclResult_t runDecompReduceComp(
    const cocclPipelineStageContext* context,
    const cocclPipelineStage* stage, cocclPipelineEdge* edge,
    const cocclPipelineStageOutput* output, cudaStream_t stream) {
  const size_t outputChunks = edge->logicalChunks / stage->reduceChunks;
  size_t reductionElements = 0;
  if (!cocclPipelineCheckedMultiply(context->rawSliceCount, outputChunks,
                                    &reductionElements)) {
    return ncclInvalidArgument;
  }

  const cocclCompressorView input = {
      edge->ptr, edge->bytes, edge->bytes, edge->totalElements,
      edge->logicalChunks, edge->datatype, edge->frameMetadata,
      edge->frameStrideBytes};
  cocclCompressorView encoded = {
      output->ptr, output->capacityBytes, 0, 0, outputChunks,
      ncclInt8, output->frameMetadata, output->frameStrideBytes};
  NCCLCHECK(ncclDecompReduceComp(
      edge->compressor, stage->compressor, context->ownerComm,
      input, &encoded,
      stage->reduceChunks, context->rawDatatype, reductionElements,
      stream));
  edge->ptr = encoded.data;
  edge->bytes = encoded.bytes;
  edge->totalElements = encoded.elements;
  edge->datatype = encoded.datatype;
  edge->logicalChunks = encoded.chunks;
  edge->compressor = stage->compressor;
  edge->frameMetadata = encoded.frameMetadata;
  edge->frameStrideBytes = encoded.frameStrideBytes;
  return ncclSuccess;
}

ncclResult_t runDecompressReduce(
    const cocclPipelineStageContext* context,
    const cocclPipelineStage* stage, cocclPipelineEdge* edge,
    const cocclPipelineStageOutput* output, cudaStream_t stream) {
  const size_t outputChunks = edge->logicalChunks / stage->reduceChunks;
  size_t reductionElements = 0;
  if (!cocclPipelineCheckedMultiply(context->rawSliceCount, outputChunks,
                                    &reductionElements)) {
    return ncclInvalidArgument;
  }
  const cocclCompressorView input = {
      edge->ptr, edge->bytes, edge->bytes, edge->totalElements,
      edge->logicalChunks, edge->datatype, edge->frameMetadata,
      edge->frameStrideBytes};
  cocclCompressorView reduced = {
      output->ptr, output->capacityBytes, 0, reductionElements,
      outputChunks, context->rawDatatype, nullptr, 0};
  NCCLCHECK(ncclDecompressReduce(
      edge->compressor, context->ownerComm, input, &reduced,
      stage->reduceChunks, stream));
  edge->ptr = reduced.data;
  edge->bytes = reduced.bytes;
  edge->totalElements = reduced.elements;
  edge->datatype = reduced.datatype;
  edge->logicalChunks = reduced.chunks;
  edge->compressor = nullptr;
  edge->frameMetadata = nullptr;
  edge->frameStrideBytes = 0;
  return ncclSuccess;
}

ncclResult_t runReduceScatter(
    const cocclPipelineStageContext* context,
    const cocclPipelineStage* stage, cocclPipelineEdge* edge,
    const cocclPipelineStageOutput* output, cudaStream_t stream) {
  const size_t outputChunks =
      edge->logicalChunks / (size_t)stage->comm->nRanks;
  const size_t recvcount = context->rawSliceCount * outputChunks;
  const ncclCollConfig_t config = communicationConfig(context, stage);
  NCCLCHECK(ncclReduceScatterConfig(
      edge->ptr, output->ptr, recvcount, context->rawDatatype, ncclSum,
      stage->comm, stream, &config));
  edge->ptr = output->ptr;
  edge->bytes = recvcount * (size_t)ncclTypeSize(context->rawDatatype);
  edge->totalElements = recvcount;
  edge->datatype = context->rawDatatype;
  edge->logicalChunks = outputChunks;
  edge->compressor = nullptr;
  edge->frameMetadata = nullptr;
  edge->frameStrideBytes = 0;
  return ncclSuccess;
}

ncclResult_t runPack(const cocclPipelineStageContext* context,
                     const cocclPipelineStage*, cocclPipelineEdge* edge,
                     const cocclPipelineStageOutput* output,
                     cudaStream_t stream) {
  NCCLCHECK(cocclLaunchPackSlice(
      edge->ptr, context->rawChunkBytes, output->ptr,
      context->rawSliceBytes, edge->logicalChunks, context->inputLayout,
      context->nNodes, context->ranksPerNode, stream));
  edge->ptr = output->ptr;
  return ncclSuccess;
}

ncclResult_t runUnpack(const cocclPipelineStageContext* context,
                       const cocclPipelineStage*, cocclPipelineEdge* edge,
                       const cocclPipelineStageOutput* output,
                       cudaStream_t stream) {
  NCCLCHECK(cocclLaunchUnpackSlice(
      edge->ptr, output->ptr, context->rawChunkBytes,
      context->rawSliceBytes, edge->logicalChunks, stream));
  edge->ptr = output->ptr;
  return ncclSuccess;
}

typedef ncclResult_t (*StageHandler)(
    const cocclPipelineStageContext*, const cocclPipelineStage*,
    cocclPipelineEdge*, const cocclPipelineStageOutput*, cudaStream_t);

static_assert(cocclPipelineStageCompress == 0 &&
                  cocclPipelineStageAllToAll == 1 &&
                  cocclPipelineStageAllGather == 2 &&
                  cocclPipelineStageDecompReduceComp == 3 &&
                  cocclPipelineStageDecompressReduce == 4 &&
                  cocclPipelineStageDecompress == 5 &&
                  cocclPipelineStageReduceScatter == 6 &&
                  cocclPipelineStagePack == 7 &&
                  cocclPipelineStageUnpack == 8,
              "pipeline stage kinds must match the handler table");

const StageHandler handlers[kCocclPipelineStageKindCount] = {
    runCompress, runAllToAll, runAllGather,
    runDecompReduceComp, runDecompressReduce, runDecompress,
    runReduceScatter, runPack, runUnpack};

}  // namespace

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
  if (buffersOverlap(
          edge->ptr, edge->bytes, output->ptr, output->capacityBytes)) {
    return ncclInvalidUsage;
  }
  if (stage->kind == cocclPipelineStageAllToAll) {
    const size_t metadataBytes =
        edge->logicalChunks / (size_t)stage->comm->nRanks *
        sizeof(cocclCompressorFrameMetadata);
    const ncclCollConfig_t config = communicationConfig(context, stage);
    return ncclAlltoAllConfig(
        edge->frameMetadata, output->frameMetadata, metadataBytes,
        ncclUint8, stage->comm, stream, &config);
  }
  const size_t metadataBytes =
      edge->logicalChunks * sizeof(cocclCompressorFrameMetadata);
  const ncclCollConfig_t config = communicationConfig(context, stage);
  return ncclAllGatherConfig(
      edge->frameMetadata, output->frameMetadata, metadataBytes,
      ncclUint8, stage->comm, stream, &config);
}

ncclResult_t cocclCommitPipelineFrameExchange(
    const cocclPipelineStageContext* context,
    const cocclPipelineStage* stage, cocclPipelineEdge* edge,
    const cocclPipelineStageOutput* output, cudaStream_t stream) {
  size_t outputFrames = edge->logicalChunks;
  if (stage->kind == cocclPipelineStageAllGather) {
    outputFrames *= (size_t)stage->comm->nRanks;
  }
  NCCLCHECK(readFrameMetadata(
      context, edge, output, outputFrames, stream));
  NCCLCHECK(ensureFrameExchangeCapacity(
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
  if (useFramedAllGatherV(
          stage, context->frameResources->exchanges, exchangeCount)) {
    NCCLCHECK(cocclCommitAllGatherVFrameExchange(
        context->frameResources->exchanges, edge->logicalChunks,
        stage->comm, stream));
  } else {
    NCCLCHECK(cocclCommitFrameExchange(
        context->frameResources->exchanges, exchangeCount,
        stage->comm, stream));
  }

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
  size_t inputSpan = edge->bytes;
  size_t outputSpan = output->capacityBytes;
  if (stage->kind == cocclPipelineStagePack ||
      stage->kind == cocclPipelineStageUnpack) {
    const size_t pitchedSpan =
        (edge->logicalChunks - 1) * context->rawChunkBytes +
        context->rawSliceBytes;
    if (stage->kind == cocclPipelineStagePack) {
      inputSpan = pitchedSpan;
      outputSpan = edge->bytes;
    } else {
      outputSpan = pitchedSpan;
    }
  }
  if (buffersOverlap(edge->ptr, inputSpan, output->ptr, outputSpan)) {
    return ncclInvalidUsage;
  }
  const ncclResult_t result =
      handlers[(int)stage->kind](context, stage, edge, output, stream);
  if (result != ncclSuccess) {
    WARN("COCCL pipeline stage %d failed with result %d bytes %zu "
         "elements %zu datatype %d chunks %zu",
         (int)stage->kind, (int)result, edge->bytes,
         edge->totalElements, (int)edge->datatype, edge->logicalChunks);
  }
  return result;
}
