#ifndef COCCL_PIPELINE_H_
#define COCCL_PIPELINE_H_

#include <stddef.h>

#include "runtime/coccl_compressor_runtime.h"
#include "nccl.h"

// COCCL overlap primitives are linear flows assembled from these common
// operations. Collective identity is carried by the communicator supplied to
// a stage rather than by separate intra/inter stage kinds.
enum cocclPipelineStageKind {
  cocclPipelineStageCompress = 0,
  cocclPipelineStageAllToAll = 1,
  cocclPipelineStageAllGather = 2,
  cocclPipelineStageDecompReduceComp = 3,
  cocclPipelineStageDecompressReduce = 4,
  cocclPipelineStageDecompress = 5,
  // Layout stages are injected at pipeline boundaries and are not part of a
  // primitive's explicit stage list.
  cocclPipelineStagePack = 6,
  cocclPipelineStageUnpack = 7,
};

// Declares the only user-buffer alias that may retain sliced overlap. The
// planner validates the exact pointer and chunk relationship before using it.
enum cocclPipelineInPlaceLayout {
  cocclPipelineInPlaceNone = 0,
  cocclPipelineInPlaceSameBuffer = 1,
  cocclPipelineInPlaceInputRankChunk = 2,
  cocclPipelineInPlaceOutputRankChunk = 3,
};

struct cocclPipelineStage {
  cocclPipelineStageKind kind;
  ncclComm_t comm;
  size_t reduceChunks;
};

static inline cocclPipelineStage cocclPipelineCompress() {
  return {cocclPipelineStageCompress, nullptr, 0};
}

static inline cocclPipelineStage cocclPipelineAllToAll(ncclComm_t comm) {
  return {cocclPipelineStageAllToAll, comm, 0};
}

static inline cocclPipelineStage cocclPipelineAllGather(ncclComm_t comm) {
  return {cocclPipelineStageAllGather, comm, 0};
}

static inline cocclPipelineStage cocclPipelineDecompReduceComp(
    size_t reduceChunks) {
  return {cocclPipelineStageDecompReduceComp, nullptr, reduceChunks};
}

static inline cocclPipelineStage cocclPipelineDecompressReduce(
    size_t reduceChunks) {
  return {cocclPipelineStageDecompressReduce, nullptr, reduceChunks};
}

static inline cocclPipelineStage cocclPipelineDecompress() {
  return {cocclPipelineStageDecompress, nullptr, 0};
}

static inline cocclPipelineStage cocclPipelinePack() {
  return {cocclPipelineStagePack, nullptr, 0};
}

static inline cocclPipelineStage cocclPipelineUnpack() {
  return {cocclPipelineStageUnpack, nullptr, 0};
}

// rawChunkCount is the uncompressed element count in one logical chunk before
// slicing. inputChunks describes how many such chunks enter the flow.
struct cocclPipelineSpec {
  const char* name;
  const void* input;
  void* output;
  size_t rawChunkCount;
  size_t inputChunks;
  ncclDataType_t datatype;
  ncclComm_t ownerComm;
  cocclCompressorHandle compressor;
  cudaStream_t stream;
  const cocclPipelineStage* stages;
  int stageCount;
  cocclPipelineInPlaceLayout inPlaceLayout;
};

ncclResult_t cocclRunPipeline(const cocclPipelineSpec* spec);

// Executes the same stage graph and workspace plan on the caller's stream,
// without slicing or creating overlap streams.
ncclResult_t cocclRunPipelineSerial(const cocclPipelineSpec* spec);

#endif
