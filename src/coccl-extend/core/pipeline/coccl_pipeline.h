#ifndef COCCL_PIPELINE_H_
#define COCCL_PIPELINE_H_

#include <stddef.h>

#include "runtime/coccl_operation.h"
#include "nccl.h"

enum cocclPipelineStageKind {
  cocclPipelineStageCompress = 0,
  cocclPipelineStageAllToAll = 1,
  cocclPipelineStageAllGather = 2,
  cocclPipelineStageDecompReduceComp = 3,
  cocclPipelineStageDecompressReduce = 4,
  cocclPipelineStageDecompress = 5,
  cocclPipelineStageReduceScatter = 6,
  // Pack and Unpack are automatic boundary stages. Primitives do not list
  // them in their explicit stage arrays.
  cocclPipelineStagePack = 7,
  cocclPipelineStageUnpack = 8,
};

// The planner preserves overlap only when user buffers match the primitive's
// declared NCCL in-place layout exactly.
enum cocclPipelineInPlaceLayout {
  cocclPipelineInPlaceNone = 0,
  cocclPipelineInPlaceSameBuffer = 1,
  cocclPipelineInPlaceInputRankChunk = 2,
  cocclPipelineInPlaceOutputRankChunk = 3,
};

enum cocclPipelineInputLayout {
  cocclPipelineInputContiguous = 0,
  cocclPipelineInputHierarchicalSwizzle = 1,
};

struct cocclPipelineStage {
  cocclPipelineStageKind kind;
  ncclComm_t comm;
  size_t reduceChunks;
  void* compressor;
  const ncclCollConfig_t* config;
};

static inline cocclPipelineStage cocclPipelineCompress(void* compressor) {
  return {cocclPipelineStageCompress, nullptr, 0, compressor, nullptr};
}

static inline cocclPipelineStage cocclPipelineAllToAll(
    ncclComm_t comm, const ncclCollConfig_t* config = nullptr) {
  return {cocclPipelineStageAllToAll, comm, 0, nullptr, config};
}

static inline cocclPipelineStage cocclPipelineAllGather(
    ncclComm_t comm, const ncclCollConfig_t* config = nullptr) {
  return {cocclPipelineStageAllGather, comm, 0, nullptr, config};
}

static inline cocclPipelineStage cocclPipelineDecompReduceComp(
    size_t reduceChunks, void* compressor) {
  return {cocclPipelineStageDecompReduceComp, nullptr, reduceChunks,
          compressor, nullptr};
}

static inline cocclPipelineStage cocclPipelineDecompressReduce(
    size_t reduceChunks) {
  return {cocclPipelineStageDecompressReduce, nullptr, reduceChunks,
          nullptr, nullptr};
}

static inline cocclPipelineStage cocclPipelineDecompress() {
  return {cocclPipelineStageDecompress, nullptr, 0, nullptr, nullptr};
}

static inline cocclPipelineStage cocclPipelineReduceScatter(
    ncclComm_t comm, const ncclCollConfig_t* config = nullptr) {
  return {cocclPipelineStageReduceScatter, comm, 0, nullptr, config};
}

static inline cocclPipelineStage cocclPipelinePack() {
  return {cocclPipelineStagePack, nullptr, 0, nullptr, nullptr};
}

static inline cocclPipelineStage cocclPipelineUnpack() {
  return {cocclPipelineStageUnpack, nullptr, 0, nullptr, nullptr};
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
  ncclComm_t ownerComm;
  cudaStream_t stream;
  const cocclPipelineStage* stages;
  int stageCount;
  cocclPipelineInPlaceLayout inPlaceLayout;
  cocclPipelineInputLayout inputLayout;
  uint64_t profilerTag;
};

ncclResult_t cocclRunPipeline(const cocclPipelineSpec* spec);
ncclResult_t cocclRunPipelineSerial(const cocclPipelineSpec* spec);
ncclResult_t cocclPipelineCommDestroy(ncclComm_t comm);

#endif
