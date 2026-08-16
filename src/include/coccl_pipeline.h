#ifndef COCCL_PIPELINE_H_
#define COCCL_PIPELINE_H_

#include <stddef.h>

#include "coccl_operation.h"
#include "nccl.h"

enum cocclPipelineStageKind {
  cocclPipelineStageCompress = 0,
  cocclPipelineStageAllToAll = 1,
  cocclPipelineStageAllGather = 2,
  cocclPipelineStageDecompress = 3,
  // Pack and Unpack are automatic boundary stages. Primitives do not list
  // them in their explicit stage arrays.
  cocclPipelineStagePack = 4,
  cocclPipelineStageUnpack = 5,
};

struct cocclPipelineStage {
  cocclPipelineStageKind kind;
  ncclComm_t comm;
};

static inline cocclPipelineStage cocclPipelineCompress() {
  return {cocclPipelineStageCompress, nullptr};
}

static inline cocclPipelineStage cocclPipelineAllToAll(ncclComm_t comm) {
  return {cocclPipelineStageAllToAll, comm};
}

static inline cocclPipelineStage cocclPipelineAllGather(ncclComm_t comm) {
  return {cocclPipelineStageAllGather, comm};
}

static inline cocclPipelineStage cocclPipelineDecompress() {
  return {cocclPipelineStageDecompress, nullptr};
}

static inline cocclPipelineStage cocclPipelinePack() {
  return {cocclPipelineStagePack, nullptr};
}

static inline cocclPipelineStage cocclPipelineUnpack() {
  return {cocclPipelineStageUnpack, nullptr};
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
};

ncclResult_t cocclRunPipeline(const cocclPipelineSpec* spec);
ncclResult_t cocclPipelineCommDestroy(ncclComm_t comm);

#endif
