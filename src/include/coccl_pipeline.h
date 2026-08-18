#ifndef COCCL_PIPELINE_H_
#define COCCL_PIPELINE_H_

#include <stddef.h>

#include "coccl_operation.h"
#include "nccl.h"

enum cocclPipelineStageKind {
  cocclPipelineStageCompress = 0,
  cocclPipelineStageAllToAll = 1,
  cocclPipelineStageAllGather = 2,
  cocclPipelineStageDecompReduceComp = 3,
  cocclPipelineStageDecompressReduce = 4,
  cocclPipelineStageDecompress = 5,
  // Pack and Unpack are automatic boundary stages. Primitives do not list
  // them in their explicit stage arrays.
  cocclPipelineStagePack = 6,
  cocclPipelineStageUnpack = 7,
};

// The planner preserves overlap only when user buffers match the primitive's
// declared NCCL in-place layout exactly.
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

// rawChunkCount is the unsliced element count in one rank chunk.
// inputChunks is the number of rank chunks entering the pipeline.
struct cocclPipelineSpec {
  const char* name;
  const void* input;
  void* output;
  size_t rawChunkCount;
  size_t inputChunks;
  ncclDataType_t datatype;
  cocclPolicyKey compressorPolicy;
  ncclComm_t ownerComm;
  cudaStream_t stream;
  const cocclPipelineStage* stages;
  int stageCount;
  cocclPipelineInPlaceLayout inPlaceLayout;
};

ncclResult_t cocclRunPipeline(const cocclPipelineSpec* spec);
ncclResult_t cocclRunPipelineSerial(const cocclPipelineSpec* spec);
ncclResult_t cocclPipelineCommDestroy(ncclComm_t comm);

#endif
