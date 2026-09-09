#ifndef COCCL_PIPELINE_EXECUTION_H_
#define COCCL_PIPELINE_EXECUTION_H_

#include "core/pipeline/coccl_pipeline_internal.h"
#include "core/memory/coccl_buffer_management.h"
#include "checks.h"

namespace cocclPipelineInternal {

enum cocclPipelinePhase {
  cocclPipelinePhasePack = 0,
  cocclPipelinePhaseFirstStage = 1,
  cocclPipelinePhaseUnpack = kCocclPipelinePhysicalStages - 1,
};

struct cocclPipelineSliceState {
  cocclPipelineEdge edge;
  cocclPipelineStageOutput output;
};

struct cocclPipelineResources {
  int depth;
  cudaStream_t streams[kCocclPipelinePhysicalStages];
  cudaEvent_t events[kCocclPipelinePhysicalStages]
                    [kCocclPipelineMaxDepth];
  cudaEvent_t inputReady;
  cudaEvent_t inputRawConsumed[kCocclPipelineRawRingSlots];
  cudaEvent_t outputRawConsumed[kCocclPipelineRawRingSlots];
  cocclPipelineSliceState slices[kCocclPipelineMaxDepth];
  cocclPipelineFrameResources frameResources;
};

struct cocclPipelineWorkspace {
  void* registeredBase;
  void* rawBase;
};

struct cocclPipelineExecution {
  cocclPipelineContext context;
  cocclBufferHandle coreWorkspace;
  cocclBufferHandle rawWorkspace;
  cocclPipelineWorkspace workspace;
  cocclPipelineResources* resources;
};

ncclResult_t prepareExecution(
    const cocclPipelineSpec* spec, int requestedDepth,
    size_t targetSliceBytes, int maxDepth, cocclPipelineExecution* execution);
ncclResult_t releaseExecution(
    cocclPipelineExecution* execution, cudaStream_t stream);

inline void* tempPtr(const cocclPipelineContext& context,
              const cocclPipelineWorkspace& workspace,
              int slice, int tempIndex) {
  const cocclPipelineTempPlan& temp = context.plan.temps[tempIndex];
  if (temp.storage == cocclPipelineRawRing) {
    return static_cast<char*>(workspace.rawBase) + temp.offset +
        (size_t)(slice % kCocclPipelineRawRingSlots) * temp.alignedBytes;
  }
  return static_cast<char*>(workspace.registeredBase) +
      (size_t)slice * context.plan.registeredSliceBytes + temp.offset;
}

inline cocclPipelineStageContext stageContextForSlice(
    const cocclPipelineContext& context, int slice) {
  cocclPipelineStageContext stageContext = context.stageContext;
  stageContext.rawSliceCount = context.slices[slice].elementCount;
  stageContext.rawSliceBytes = context.slices[slice].bytes;
  return stageContext;
}

inline cocclPipelineEdge inputEdge(const cocclPipelineContext& context,
                            int slice) {
  const cocclPipelineSliceShape& shape = context.slices[slice];
  return {
      static_cast<char*>(const_cast<void*>(context.spec->input)) +
          shape.byteOffset,
      shape.bytes * context.spec->inputChunks,
      shape.elementCount * context.spec->inputChunks,
      context.spec->datatype,
      context.spec->inputChunks,
      nullptr,
      nullptr,
      0,
  };
}

inline cocclPipelineStageOutput stageOutput(const cocclPipelineContext& context,
                                     const cocclPipelineWorkspace& workspace,
                                     int stage, int slice) {
  const int tempIndex = context.plan.stageOutputTemp[stage];
  if (tempIndex >= 0) {
    void* payload = tempPtr(context, workspace, slice, tempIndex);
    const cocclPipelineTempPlan& temp = context.plan.temps[tempIndex];
    return {
        payload,
        context.sliceStageOutputBytes[slice][stage],
        temp.frameMetadataBytes == 0
            ? nullptr
            : reinterpret_cast<cocclCompressorFrameMetadata*>(
                  static_cast<char*>(payload) +
                  temp.frameMetadataOffset),
        context.sliceStageFrameStrideBytes[slice][stage]};
  }
  return {static_cast<char*>(context.spec->output) +
              context.slices[slice].byteOffset,
          context.sliceStageOutputBytes[slice][stage], nullptr, 0};
}

inline ncclResult_t waitForPhase(cocclPipelineResources* resources, int phase,
                          int previousPhase, int slice) {
  CUDACHECK(cudaStreamWaitEvent(resources->streams[phase],
                                resources->events[previousPhase][slice], 0));
  return ncclSuccess;
}

inline ncclResult_t recordPhase(cocclPipelineResources* resources, int phase,
                         int slice) {
  CUDACHECK(cudaEventRecord(resources->events[phase][slice],
                            resources->streams[phase]));
  return ncclSuccess;
}

}  // namespace cocclPipelineInternal

#endif
