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
  cocclPipelineStageSendRecv = 9,
};

enum cocclPipelineSendRecvDirection {
  cocclPipelineSend = 0,
  cocclPipelineRecv = 1,
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
  int peer = -1;
  cocclPipelineSendRecvDirection direction = cocclPipelineSend;
};

static inline cocclPipelineStage cocclPipelineCompress(void* compressor) {
  return {cocclPipelineStageCompress, nullptr, 0, compressor};
}

static inline cocclPipelineStage cocclPipelineAllToAll(ncclComm_t comm) {
  return {cocclPipelineStageAllToAll, comm, 0, nullptr};
}

static inline cocclPipelineStage cocclPipelineAllGather(ncclComm_t comm) {
  return {cocclPipelineStageAllGather, comm, 0, nullptr};
}

static inline cocclPipelineStage cocclPipelineDecompReduceComp(
    size_t reduceChunks, void* compressor) {
  return {cocclPipelineStageDecompReduceComp, nullptr, reduceChunks,
          compressor};
}

static inline cocclPipelineStage cocclPipelineDecompressReduce(
    size_t reduceChunks) {
  return {cocclPipelineStageDecompressReduce, nullptr, reduceChunks,
          nullptr};
}

static inline cocclPipelineStage cocclPipelineDecompress() {
  return {cocclPipelineStageDecompress, nullptr, 0, nullptr};
}

static inline cocclPipelineStage cocclPipelineReduceScatter(
    ncclComm_t comm) {
  return {cocclPipelineStageReduceScatter, comm, 0, nullptr};
}

static inline cocclPipelineStage cocclPipelinePack() {
  return {cocclPipelineStagePack, nullptr, 0, nullptr};
}

static inline cocclPipelineStage cocclPipelineUnpack() {
  return {cocclPipelineStageUnpack, nullptr, 0, nullptr};
}

static inline cocclPipelineStage cocclPipelineSendRecv(
    ncclComm_t comm, int peer, cocclPipelineSendRecvDirection direction,
    void* compressor) {
  return {cocclPipelineStageSendRecv, comm, 0, compressor, peer, direction};
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
};

ncclResult_t cocclRunPipeline(const cocclPipelineSpec* spec);
ncclResult_t cocclRunPipelineBatch(
    const cocclPipelineSpec* specs, size_t count);
ncclResult_t cocclRunPipelineSerial(const cocclPipelineSpec* spec);
ncclResult_t cocclPipelineCommDestroy(ncclComm_t comm);

#endif
