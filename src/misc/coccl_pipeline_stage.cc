#include "coccl_pipeline_internal.h"

#include "checks.h"
#include "collectives.h"
#include "comm.h"
#include "compress.h"
#include "coccl_pipeline_layout.h"
#include "debug.h"

namespace {

ncclCommOp_t compressorOperation(const cocclPolicyKey& policy) {
  const bool hierarchical =
      policy.variant == cocclPolicyVariant::Hierarchical;
  switch (policy.operation) {
    case cocclOperation::AllToAll:
      return hierarchical ? AlltoAll_Inter : AlltoAll;
    case cocclOperation::AllGather:
      return hierarchical ? AllGather_Inter : AllGather;
    case cocclOperation::ReduceScatter:
      return hierarchical ? ReduceScatter_Inter : ReduceScatter;
    case cocclOperation::AllReduce:
      return hierarchical ? AllReduce_Inter : AllReduce;
    case cocclOperation::SendRecv:
      return SendRecv;
    case cocclOperation::Count:
      break;
  }
  __builtin_unreachable();
}

bool buffersOverlap(const void* first, size_t firstBytes,
                    const void* second, size_t secondBytes) {
  const uintptr_t firstBegin = reinterpret_cast<uintptr_t>(first);
  const uintptr_t secondBegin = reinterpret_cast<uintptr_t>(second);
  return firstBegin < secondBegin + secondBytes &&
      secondBegin < firstBegin + firstBytes;
}

ncclResult_t runCompress(const cocclPipelineStageContext* context,
                         const cocclPipelineStage*, cocclPipelineEdge* edge,
                         const cocclPipelineStageOutput* output,
                         cudaStream_t stream) {
  void* encoded = output->ptr;
  size_t encodedChunkCount = 0;
  ncclDataType_t encodedDatatype = ncclNumTypes;
  NCCLCHECK(ncclCompress(
      edge->ptr, &encoded, context->rawSliceCount, context->rawDatatype,
      &encodedChunkCount, &encodedDatatype, edge->logicalChunks,
      context->ownerComm->rank,
      compressorOperation(context->compressorPolicy), stream));

  size_t encodedElements = 0;
  size_t encodedBytes = 0;
  const int typeBytes = ncclTypeSize(encodedDatatype);
  if (typeBytes <= 0 ||
      !cocclPipelineCheckedMultiply(encodedChunkCount, edge->logicalChunks,
                                    &encodedElements) ||
      !cocclPipelineCheckedMultiply(encodedElements, (size_t)typeBytes,
                                    &encodedBytes) ||
      encodedBytes > output->capacityBytes) {
    return ncclInvalidUsage;
  }
  edge->ptr = encoded;
  edge->bytes = encodedBytes;
  edge->totalElements = encodedElements;
  edge->datatype = encodedDatatype;
  return ncclSuccess;
}

ncclResult_t runAllToAll(const cocclPipelineStageContext*,
                         const cocclPipelineStage* stage,
                         cocclPipelineEdge* edge,
                         const cocclPipelineStageOutput* output,
                         cudaStream_t stream) {
  const size_t sendCount = edge->totalElements / edge->logicalChunks;
  NCCLCHECK(ncclAllToAll(edge->ptr, output->ptr, sendCount, edge->datatype,
                         stage->comm, stream));
  edge->ptr = output->ptr;
  return ncclSuccess;
}

ncclResult_t runAllGather(const cocclPipelineStageContext*,
                          const cocclPipelineStage* stage,
                          cocclPipelineEdge* edge,
                          const cocclPipelineStageOutput* output,
                          cudaStream_t stream) {
  NCCLCHECK(ncclAllGather(edge->ptr, output->ptr, edge->bytes, ncclUint8,
                          stage->comm, stream));
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
  const size_t encodedChunkCount =
      edge->totalElements / edge->logicalChunks;
  NCCLCHECK(ncclDecompress(
      output->ptr, edge->ptr, context->rawSliceCount,
      context->rawDatatype, encodedChunkCount, edge->datatype,
      edge->logicalChunks, compressorOperation(context->compressorPolicy),
      stream));
  edge->ptr = output->ptr;
  edge->bytes = context->rawSliceBytes * edge->logicalChunks;
  edge->totalElements = context->rawSliceCount * edge->logicalChunks;
  edge->datatype = context->rawDatatype;
  return ncclSuccess;
}

ncclResult_t runPack(const cocclPipelineStageContext* context,
                     const cocclPipelineStage*, cocclPipelineEdge* edge,
                     const cocclPipelineStageOutput* output,
                     cudaStream_t stream) {
  NCCLCHECK(cocclLaunchPackSlice(
      edge->ptr, context->rawChunkBytes, output->ptr,
      context->rawSliceBytes, edge->logicalChunks, stream));
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
                  cocclPipelineStageDecompress == 3 &&
                  cocclPipelineStagePack == 4 &&
                  cocclPipelineStageUnpack == 5,
              "pipeline stage kinds must match the handler table");

const StageHandler handlers[kCocclPipelineStageKindCount] = {
    runCompress, runAllToAll, runAllGather,
    runDecompress, runPack, runUnpack};

}  // namespace

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
