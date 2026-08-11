#ifndef COCCL_PIPELINE_INTERNAL_H_
#define COCCL_PIPELINE_INTERNAL_H_

#include "pipeline/coccl_pipeline_stage.h"

#include <stdint.h>

constexpr int kMaxPipelineStages = 8;
constexpr int kMaxPipelineTemps = kMaxPipelineStages + 1;
constexpr int kPipelineRawRingSlots = 2;
constexpr size_t kPipelineBufferAlignment = 256;

enum cocclPipelineWorkspaceKind {
  cocclPipelineWorkspaceUnified,
  cocclPipelineWorkspaceSplit,
};

enum cocclPipelineTempStorage {
  cocclPipelineRegisteredArena,
  cocclPipelineRawRing,
};

struct cocclPipelineTempPlan {
  size_t offset;
  size_t logicalBytes;
  size_t bytes;
  cocclPipelineTempStorage storage;
};

struct cocclPipelinePlan {
  cocclPipelineWorkspaceKind workspaceKind;
  int tempCount;
  int inputStagingTemp;
  int outputStagingTemp;
  int stageOutputTemp[kMaxPipelineStages];
  size_t stageOutputCapacityBytes[kMaxPipelineStages];
  cocclPipelineTempPlan temps[kMaxPipelineTemps];
  size_t finalChunks;
  size_t registeredSliceBytes;
  size_t registeredBytes;
  size_t rawBytes;
  size_t totalBytes;
};

struct cocclPipelineContext {
  const cocclPipelineSpec* spec;
  int depth;
  size_t rawSliceCount;
  size_t rawSliceBytes;
  cocclPipelineStageContext stageContext;
  cocclPipelinePlan plan;
};

inline bool cocclPipelineCheckedMultiply(
    size_t lhs, size_t rhs, size_t* result) {
  if (result == nullptr || (lhs != 0 && rhs > SIZE_MAX / lhs)) return false;
  *result = lhs * rhs;
  return true;
}

inline bool cocclPipelineCheckedAdd(size_t lhs, size_t rhs, size_t* result) {
  if (result == nullptr || rhs > SIZE_MAX - lhs) return false;
  *result = lhs + rhs;
  return true;
}

inline bool cocclAlignPipelineBytes(size_t bytes, size_t* aligned) {
  size_t padded = 0;
  if (!cocclPipelineCheckedAdd(
          bytes, kPipelineBufferAlignment - 1, &padded)) {
    return false;
  }
  *aligned = padded / kPipelineBufferAlignment * kPipelineBufferAlignment;
  return true;
}

// Maps one already-shaped logical buffer chain to both physical layouts and
// keeps the smaller valid result. Equal sizes retain the simpler Unified path.
ncclResult_t cocclPlanPipelineWorkspace(cocclPipelinePlan* plan, int depth);

// Reports whether overlapping user buffers require the pipeline to run as one
// unsliced operation. Exact declared in-place layouts are slice-safe.
ncclResult_t cocclPipelineUserBuffersRequireSerial(
    const void* input, size_t inputChunks, void* output, size_t outputChunks,
    size_t rawChunkBytes, int rank,
    cocclPipelineInPlaceLayout inPlaceLayout, bool* requireSerial);

ncclResult_t cocclValidatePipelineSpec(const cocclPipelineSpec* spec);
ncclResult_t cocclPreparePipeline(cocclPipelineContext* context);

#endif
