#include "coccl_pipeline_stage.h"

#include "coccl_primitives_internal.h"

#include <limits.h>

namespace {

bool cocclStageCheckedMultiply(size_t lhs, size_t rhs, size_t* result) {
  if (result == nullptr || (lhs != 0 && rhs > SIZE_MAX / lhs)) return false;
  *result = lhs * rhs;
  return true;
}

using cocclPipelineStageFn = ncclResult_t (*)(
    const cocclPipelineStageContext* context,
    const cocclPipelineStage* stage, cocclPipelineEdge* edge,
    void* outputPtr, cudaStream_t stream);

ncclResult_t cocclRunCompressStage(
    const cocclPipelineStageContext* context,
    const cocclPipelineStage* /* stage */, cocclPipelineEdge* edge,
    void* outputPtr, cudaStream_t stream) {
  const size_t logicalChunks = edge->logicalChunks;
  size_t compElementsPerChunk = 0;
  ncclDataType_t compDatatype = ncclInt8;
  void* compressOutput = outputPtr;
  NCCLCHECK(ncclCompress(
      edge->ptr, &compressOutput, context->rawSliceCount,
      context->rawDatatype, &compElementsPerChunk, &compDatatype,
      logicalChunks, context->ownerComm->rank, context->ownerComm,
      context->commOp, stream));
  if (!cocclStageCheckedMultiply(compElementsPerChunk, logicalChunks,
                                 &edge->totalElements)) {
    return ncclInvalidArgument;
  }
  edge->ptr = compressOutput;
  edge->datatype = compDatatype;
  return ncclSuccess;
}

ncclResult_t cocclRunAllToAllStage(
    const cocclPipelineStageContext* /* context */,
    const cocclPipelineStage* stage, cocclPipelineEdge* edge,
    void* outputPtr, cudaStream_t stream) {
  if (stage->comm->nRanks <= 0 ||
      edge->totalElements % (size_t)stage->comm->nRanks != 0) {
    return ncclInvalidArgument;
  }
  const size_t sendCount =
      edge->totalElements / (size_t)stage->comm->nRanks;
  NCCLCHECK(ncclAllToAll(edge->ptr, outputPtr, sendCount, edge->datatype,
                         stage->comm, stream));
  edge->ptr = outputPtr;
  return ncclSuccess;
}

ncclResult_t cocclRunAllGatherStage(
    const cocclPipelineStageContext* /* context */,
    const cocclPipelineStage* stage, cocclPipelineEdge* edge,
    void* /* outputPtr */, cudaStream_t stream) {
  struct ncclInfo info = {
      ncclFuncAllGather, "AllGather", edge->ptr, edge->ptr,
      edge->totalElements, edge->datatype, ncclSum, 0, stage->comm, stream,
      ALLGATHER_CHUNKSTEPS, ALLGATHER_SLICESTEPS};
  NCCLCHECK(ncclEnqueueCheck(&info));

  size_t gatheredElements = 0;
  if (!cocclStageCheckedMultiply(edge->totalElements,
                                 (size_t)stage->comm->nRanks,
                                 &gatheredElements)) {
    return ncclInvalidArgument;
  }
  edge->totalElements = gatheredElements;
  return ncclSuccess;
}

ncclResult_t cocclRunDecompReduceCompStage(
    const cocclPipelineStageContext* context,
    const cocclPipelineStage* stage, cocclPipelineEdge* edge,
    void* outputPtr, cudaStream_t stream) {
  if (edge->logicalChunks % stage->reduceChunks != 0 ||
      edge->totalElements % stage->reduceChunks != 0) {
    return ncclInvalidArgument;
  }

  const size_t remainingChunks =
      edge->logicalChunks / stage->reduceChunks;
  size_t originalChunkCount = 0;
  if (!cocclStageCheckedMultiply(context->rawSliceCount, remainingChunks,
                                 &originalChunkCount)) {
    return ncclInvalidArgument;
  }

  const size_t compChunkCount =
      edge->totalElements / stage->reduceChunks;
  size_t recompressedCount = 0;
  ncclDataType_t recompressedDatatype = ncclInt8;
  void* recompressedOutput = outputPtr;
  NCCLCHECK(ncclDecompReduceComp(
      edge->ptr, &recompressedOutput, originalChunkCount,
      context->rawDatatype, compChunkCount, edge->datatype,
      &recompressedCount, &recompressedDatatype, stage->reduceChunks,
      context->ownerComm, context->commOp, stream));
  edge->ptr = recompressedOutput;
  edge->totalElements = recompressedCount;
  edge->datatype = recompressedDatatype;
  return ncclSuccess;
}

ncclResult_t cocclRunDecompressReduceStage(
    const cocclPipelineStageContext* context,
    const cocclPipelineStage* stage, cocclPipelineEdge* edge,
    void* outputPtr, cudaStream_t stream) {
  if (edge->logicalChunks % stage->reduceChunks != 0 ||
      edge->totalElements % stage->reduceChunks != 0) {
    return ncclInvalidArgument;
  }

  const size_t remainingChunks =
      edge->logicalChunks / stage->reduceChunks;
  size_t reduceChunkCount = 0;
  if (!cocclStageCheckedMultiply(context->rawSliceCount, remainingChunks,
                                 &reduceChunkCount)) {
    return ncclInvalidArgument;
  }
  NCCLCHECK(ncclDecompressReduce(
      outputPtr, edge->ptr, edge->totalElements / stage->reduceChunks,
      edge->datatype, reduceChunkCount, context->rawDatatype,
      stage->reduceChunks, context->ownerComm, context->commOp, stream));
  edge->ptr = outputPtr;
  edge->totalElements = reduceChunkCount;
  edge->datatype = context->rawDatatype;
  return ncclSuccess;
}

ncclResult_t cocclRunDecompressStage(
    const cocclPipelineStageContext* context,
    const cocclPipelineStage* /* stage */, cocclPipelineEdge* edge,
    void* outputPtr, cudaStream_t stream) {
  if (edge->logicalChunks == 0 ||
      edge->totalElements % edge->logicalChunks != 0) {
    return ncclInvalidArgument;
  }
  NCCLCHECK(ncclDecompress(
      outputPtr, edge->ptr, context->rawSliceCount, context->rawDatatype,
      edge->totalElements / edge->logicalChunks, edge->datatype,
      edge->logicalChunks, context->ownerComm, context->commOp, stream));
  if (!cocclStageCheckedMultiply(context->rawSliceCount,
                                 edge->logicalChunks,
                                 &edge->totalElements)) {
    return ncclInvalidArgument;
  }
  edge->ptr = outputPtr;
  edge->datatype = context->rawDatatype;
  return ncclSuccess;
}

constexpr int kPipelineStageKindCount = cocclPipelineStageDecompress + 1;
static_assert(cocclPipelineStageCompress == 0 &&
                  cocclPipelineStageAllToAll == 1 &&
                  cocclPipelineStageAllGather == 2 &&
                  cocclPipelineStageDecompReduceComp == 3 &&
                  cocclPipelineStageDecompressReduce == 4 &&
                  cocclPipelineStageDecompress == 5,
              "pipeline stage handlers require contiguous stage kinds");

const cocclPipelineStageFn
    cocclPipelineStageHandlers[kPipelineStageKindCount] = {
        cocclRunCompressStage,
        cocclRunAllToAllStage,
        cocclRunAllGatherStage,
        cocclRunDecompReduceCompStage,
        cocclRunDecompressReduceStage,
        cocclRunDecompressStage,
};

}  // namespace

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

ncclResult_t cocclExecutePipelineStage(
    const cocclPipelineStageContext* context,
    const cocclPipelineStage* stage, cocclPipelineEdge* edge,
    void* outputPtr, cudaStream_t stream) {
  if (context == nullptr || context->ownerComm == nullptr ||
      stage == nullptr || edge == nullptr || outputPtr == nullptr) {
    return ncclInvalidArgument;
  }

  const int stageKind = (int)stage->kind;
  if (stageKind < 0 || stageKind >= kPipelineStageKindCount) {
    return ncclInvalidArgument;
  }
  const cocclPipelineStageFn handler =
      cocclPipelineStageHandlers[stageKind];
  return handler == nullptr
      ? ncclInvalidArgument
      : handler(context, stage, edge, outputPtr, stream);
}
