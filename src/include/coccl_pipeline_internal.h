#ifndef COCCL_PIPELINE_INTERNAL_H_
#define COCCL_PIPELINE_INTERNAL_H_

#include <stdint.h>

#include "coccl_pipeline.h"

constexpr int kCocclPipelineExplicitStages = 7;
constexpr int kCocclPipelinePhysicalStages = 9;
constexpr int kCocclPipelineStageKindCount = 8;
constexpr int kCocclPipelineMaxTemps = 8;
constexpr int kCocclPipelineMaxDepth = 16;
constexpr size_t kCocclPipelineAlignment = 256;

struct cocclPipelineEdge {
  void* ptr;
  size_t bytes;
  size_t totalElements;
  ncclDataType_t datatype;
  size_t logicalChunks;
};

struct cocclPipelineStageOutput {
  void* ptr;
  size_t capacityBytes;
};

struct cocclPipelineStageContext {
  size_t rawSliceCount;
  size_t rawSliceBytes;
  size_t rawChunkBytes;
  ncclDataType_t rawDatatype;
  cocclPolicyKey compressorPolicy;
  ncclComm_t ownerComm;
};

enum cocclPipelineTempRole {
  cocclPipelineTempInputStaging,
  cocclPipelineTempCompressOutput,
  cocclPipelineTempAllToAllOutput,
  cocclPipelineTempAllGatherOutput,
  cocclPipelineTempDecompReduceCompOutput,
  cocclPipelineTempDecompressReduceOutput,
  cocclPipelineTempOutputStaging,
};

struct cocclPipelineTempPlan {
  cocclPipelineTempRole role;
  size_t offset;
  size_t logicalBytes;
  size_t alignedBytes;
};

struct cocclPipelinePlan {
  int tempCount;
  int inputStagingTemp;
  int outputStagingTemp;
  int stageOutputTemp[kCocclPipelineExplicitStages];
  size_t stageOutputCapacityBytes[kCocclPipelineExplicitStages];
  cocclPipelineTempPlan temps[kCocclPipelineMaxTemps];
  size_t finalChunks;
  size_t sliceWorkspaceBytes;
  size_t workspaceBytes;
};

struct cocclPipelineContext {
  const cocclPipelineSpec* spec;
  int depth;
  cocclPipelineStageContext stageContext;
  cocclPipelinePlan plan;
};

inline bool cocclPipelineCheckedMultiply(size_t lhs, size_t rhs,
                                         size_t* result) {
  if (lhs != 0 && rhs > SIZE_MAX / lhs) return false;
  *result = lhs * rhs;
  return true;
}

inline bool cocclPipelineCheckedAdd(size_t lhs, size_t rhs,
                                    size_t* result) {
  if (rhs > SIZE_MAX - lhs) return false;
  *result = lhs + rhs;
  return true;
}

inline bool cocclAlignPipelineBytes(size_t bytes, size_t* aligned) {
  size_t padded = 0;
  if (!cocclPipelineCheckedAdd(bytes, kCocclPipelineAlignment - 1,
                               &padded)) {
    return false;
  }
  *aligned = padded / kCocclPipelineAlignment * kCocclPipelineAlignment;
  return true;
}

ncclResult_t cocclPreparePipeline(const cocclPipelineSpec* spec,
                                  int requestedDepth,
                                  cocclPipelineContext* context);
ncclResult_t cocclPlanUnifiedWorkspace(cocclPipelinePlan* plan, int depth);
ncclResult_t cocclPipelineStageOutputChunks(
    const cocclPipelineStage& stage, size_t inputChunks,
    size_t* outputChunks);
ncclResult_t cocclExecutePipelineStage(
    const cocclPipelineStageContext* context,
    const cocclPipelineStage* stage, cocclPipelineEdge* edge,
    const cocclPipelineStageOutput* output, cudaStream_t stream);

#endif
