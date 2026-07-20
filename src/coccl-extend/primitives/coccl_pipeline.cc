#include "coccl_primitives_internal.h"
#include "coccl_pipeline.h"
#include "coccl_pipeline_layout.h"
#include "coccl_pipeline_stage.h"
#include "param.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

NCCL_PARAM(PipelineDepth, "PIPELINE_DEPTH", 0);

namespace {

constexpr int kMaxPipelineStages = 8;
constexpr size_t kPipelineBufferAlignment = 256;

struct cocclPipelineResources {
  const char* name;
  int cudaDev;
  int stageCount;
  cudaStream_t* streams;
  cudaEvent_t* events;
  cudaStream_t packStream;
  cudaStream_t unpackStream;
  cudaEvent_t packReadyEvent;
  cudaEvent_t unpackDoneEvent;
  cudaEvent_t mainEvent;
  cocclPipelineResources* next;
};

struct cocclPipelineTempPlan {
  size_t offset;
  size_t bytes;
};

struct cocclPipelinePlan {
  int tempCount;
  int inputStagingTemp;
  int outputStagingTemp;
  int stageOutputTemp[kMaxPipelineStages];
  cocclPipelineTempPlan temps[kMaxPipelineStages];
  size_t finalChunks;
  size_t sliceBytes;
  size_t workspaceBytes;
};

struct cocclPipelineContext {
  const cocclPipelineSpec* spec;
  int depth;
  size_t rawSliceCount;
  size_t rawSliceBytes;
  cocclPipelineStageContext stageContext;
  cocclPipelinePlan plan;
};

__thread cocclPipelineResources* pipelineResources = nullptr;

bool cocclCheckedMultiply(size_t lhs, size_t rhs, size_t* result) {
  if (result == nullptr || (lhs != 0 && rhs > SIZE_MAX / lhs)) return false;
  *result = lhs * rhs;
  return true;
}

bool cocclCheckedAdd(size_t lhs, size_t rhs, size_t* result) {
  if (result == nullptr || rhs > SIZE_MAX - lhs) return false;
  *result = lhs + rhs;
  return true;
}

bool cocclAlignPipelineBytes(size_t bytes, size_t* aligned) {
  size_t padded = 0;
  if (!cocclCheckedAdd(bytes, kPipelineBufferAlignment - 1, &padded)) {
    return false;
  }
  *aligned = padded / kPipelineBufferAlignment * kPipelineBufferAlignment;
  return true;
}

bool cocclStageCreatesTemp(cocclPipelineStageKind kind) {
  return kind == cocclPipelineStageCompress ||
      kind == cocclPipelineStageAllToAll ||
      kind == cocclPipelineStageDecompReduceComp;
}

ncclResult_t cocclValidatePipelineSpec(const cocclPipelineSpec* spec) {
  if (spec == nullptr || spec->name == nullptr || spec->input == nullptr ||
      spec->output == nullptr || spec->ownerComm == nullptr ||
      spec->stages == nullptr || spec->stageCount < 2 ||
      spec->stageCount > kMaxPipelineStages || spec->rawChunkCount == 0 ||
      spec->inputChunks == 0) {
    return ncclInvalidArgument;
  }
  if (spec->stages[0].kind != cocclPipelineStageCompress) {
    return ncclInvalidArgument;
  }
  cocclPipelineStageKind last = spec->stages[spec->stageCount - 1].kind;
  if (last != cocclPipelineStageDecompress &&
      last != cocclPipelineStageDecompressReduce) {
    return ncclInvalidArgument;
  }
  for (int i = 0; i < spec->stageCount; ++i) {
    const auto& stage = spec->stages[i];
    switch (stage.kind) {
      case cocclPipelineStageCompress:
        if (i != 0) return ncclInvalidArgument;
        break;
      case cocclPipelineStageAllToAll:
      case cocclPipelineStageAllGather:
        if (stage.comm == nullptr) return ncclInvalidArgument;
        break;
      case cocclPipelineStageDecompReduceComp:
      case cocclPipelineStageDecompressReduce:
        if (stage.reduceChunks == 0) return ncclInvalidArgument;
        break;
      case cocclPipelineStageDecompress:
        break;
      default:
        return ncclInvalidArgument;
    }
  }
  return ncclSuccess;
}

ncclResult_t cocclBuildPipelinePlan(cocclPipelineContext* ctx) {
  const auto* spec = ctx->spec;
  auto* plan = &ctx->plan;
  size_t logicalChunks = spec->inputChunks;
  int currentTemp = -1;
  plan->inputStagingTemp = -1;
  plan->outputStagingTemp = -1;

  for (int stageIndex = 0; stageIndex < spec->stageCount; ++stageIndex) {
    const auto& stage = spec->stages[stageIndex];
    size_t outputChunks = 0;
    NCCLCHECK(cocclPipelineStageOutputChunks(stage, logicalChunks,
                                             &outputChunks));
    logicalChunks = outputChunks;

    bool finalStage = stageIndex + 1 == spec->stageCount;
    if (stage.kind == cocclPipelineStageAllGather) {
      // AllGather grows the current edge in place. A logical edge may contain
      // one or more chunks; every rank contributes the complete edge.
      if (currentTemp < 0) return ncclInvalidArgument;
      size_t capacity = 0;
      if (!cocclCheckedMultiply(ctx->rawSliceBytes, logicalChunks, &capacity) ||
          !cocclAlignPipelineBytes(capacity, &capacity)) {
        return ncclInvalidArgument;
      }
      if (capacity > plan->temps[currentTemp].bytes) {
        plan->temps[currentTemp].bytes = capacity;
      }
      plan->stageOutputTemp[stageIndex] = currentTemp;
    } else if (finalStage) {
      plan->stageOutputTemp[stageIndex] = -1;
    } else if (cocclStageCreatesTemp(stage.kind)) {
      if (plan->tempCount >= kMaxPipelineStages) return ncclInvalidArgument;
      currentTemp = plan->tempCount++;
      size_t capacity = 0;
      if (!cocclCheckedMultiply(ctx->rawSliceBytes, logicalChunks, &capacity) ||
          !cocclAlignPipelineBytes(capacity, &capacity)) {
        return ncclInvalidArgument;
      }
      plan->temps[currentTemp].bytes = capacity;
      plan->stageOutputTemp[stageIndex] = currentTemp;
    } else {
      return ncclInvalidArgument;
    }
  }

  plan->finalChunks = logicalChunks;
  // User collective buffers are chunk-major. Overlapped slices are packed
  // into a contiguous edge before Compress and scattered back after the final
  // stage, keeping this layout detail out of primitive flow descriptions.
  if (ctx->depth > 1 && spec->inputChunks > 1) {
    if (plan->tempCount >= kMaxPipelineStages) return ncclInvalidArgument;
    plan->inputStagingTemp = plan->tempCount++;
    size_t capacity = 0;
    if (!cocclCheckedMultiply(ctx->rawSliceBytes, spec->inputChunks,
                              &capacity) ||
        !cocclAlignPipelineBytes(capacity, &capacity)) {
      return ncclInvalidArgument;
    }
    plan->temps[plan->inputStagingTemp].bytes = capacity;
  }
  if (ctx->depth > 1 && plan->finalChunks > 1) {
    if (plan->tempCount >= kMaxPipelineStages) return ncclInvalidArgument;
    plan->outputStagingTemp = plan->tempCount++;
    size_t capacity = 0;
    if (!cocclCheckedMultiply(ctx->rawSliceBytes, plan->finalChunks,
                              &capacity) ||
        !cocclAlignPipelineBytes(capacity, &capacity)) {
      return ncclInvalidArgument;
    }
    plan->temps[plan->outputStagingTemp].bytes = capacity;
  }
  plan->sliceBytes = 0;
  for (int i = 0; i < plan->tempCount; ++i) {
    plan->temps[i].offset = plan->sliceBytes;
    if (!cocclCheckedAdd(plan->sliceBytes, plan->temps[i].bytes,
                         &plan->sliceBytes)) {
      return ncclInvalidArgument;
    }
  }
  if (!cocclCheckedMultiply(plan->sliceBytes, (size_t)ctx->depth,
                            &plan->workspaceBytes) ||
      plan->workspaceBytes == 0) {
    return ncclInvalidArgument;
  }
  return ncclSuccess;
}

ncclResult_t cocclPreparePipeline(cocclPipelineContext* ctx) {
  ctx->rawSliceCount = ctx->spec->rawChunkCount / (size_t)ctx->depth;
  if (ctx->rawSliceCount == 0 ||
      !cocclCheckedMultiply(ctx->rawSliceCount,
                            ncclTypeSize(ctx->spec->datatype),
                            &ctx->rawSliceBytes)) {
    return ncclInvalidArgument;
  }
  ctx->stageContext = {
      ctx->rawSliceCount,
      ctx->spec->datatype,
      ctx->spec->ownerComm,
      ctx->spec->commOp,
  };
  NCCLCHECK(cocclBuildPipelinePlan(ctx));
  return ncclSuccess;
}

ncclResult_t cocclPipelineUserBuffersOverlap(const cocclPipelineContext* ctx,
                                              bool* overlap) {
  if (overlap == nullptr) return ncclInvalidArgument;
  size_t inputElements = 0;
  size_t outputElements = 0;
  size_t inputBytes = 0;
  size_t outputBytes = 0;
  size_t typeSize = ncclTypeSize(ctx->spec->datatype);
  if (!cocclCheckedMultiply(ctx->spec->rawChunkCount,
                            ctx->spec->inputChunks, &inputElements) ||
      !cocclCheckedMultiply(ctx->spec->rawChunkCount,
                            ctx->plan.finalChunks, &outputElements) ||
      !cocclCheckedMultiply(inputElements, typeSize, &inputBytes) ||
      !cocclCheckedMultiply(outputElements, typeSize, &outputBytes)) {
    return ncclInvalidArgument;
  }

  uintptr_t inputBegin = (uintptr_t)ctx->spec->input;
  uintptr_t outputBegin = (uintptr_t)ctx->spec->output;
  if (inputBytes > UINTPTR_MAX - inputBegin ||
      outputBytes > UINTPTR_MAX - outputBegin) {
    return ncclInvalidArgument;
  }
  uintptr_t inputEnd = inputBegin + inputBytes;
  uintptr_t outputEnd = outputBegin + outputBytes;
  *overlap = inputBegin < outputEnd && outputBegin < inputEnd;
  return ncclSuccess;
}

ncclResult_t cocclFindOrCreatePipelineResources(
    const char* name, int cudaDev, int stageCount,
    cocclPipelineResources** out) {
  if (name == nullptr || cudaDev < 0 || stageCount <= 0 || out == nullptr) {
    return ncclInvalidArgument;
  }

  for (cocclPipelineResources* cur = pipelineResources; cur != nullptr;
       cur = cur->next) {
    if (cur->cudaDev == cudaDev && cur->stageCount == stageCount &&
        strcmp(cur->name, name) == 0) {
      *out = cur;
      return ncclSuccess;
    }
  }

  cocclPipelineResources* res =
      (cocclPipelineResources*)calloc(1, sizeof(cocclPipelineResources));
  if (res == nullptr) return ncclSystemError;
  res->streams = (cudaStream_t*)calloc(stageCount, sizeof(cudaStream_t));
  res->events = (cudaEvent_t*)calloc(stageCount, sizeof(cudaEvent_t));
  if (res->streams == nullptr || res->events == nullptr) {
    free(res->streams);
    free(res->events);
    free(res);
    return ncclSystemError;
  }

  res->name = name;
  res->cudaDev = cudaDev;
  res->stageCount = stageCount;
  for (int i = 0; i < stageCount; ++i) {
    CUDACHECK(cudaStreamCreateWithFlags(res->streams + i,
                                        cudaStreamNonBlocking));
    CUDACHECK(cudaEventCreateWithFlags(res->events + i, cudaEventDefault));
  }
  CUDACHECK(cudaStreamCreateWithFlags(&res->packStream,
                                      cudaStreamNonBlocking));
  CUDACHECK(cudaStreamCreateWithFlags(&res->unpackStream,
                                      cudaStreamNonBlocking));
  CUDACHECK(cudaEventCreateWithFlags(&res->packReadyEvent,
                                     cudaEventDefault));
  CUDACHECK(cudaEventCreateWithFlags(&res->unpackDoneEvent,
                                     cudaEventDefault));
  CUDACHECK(cudaEventCreateWithFlags(&res->mainEvent, cudaEventDefault));

  res->next = pipelineResources;
  pipelineResources = res;
  *out = res;
  return ncclSuccess;
}

void* cocclPipelineTempPtr(const cocclPipelineContext* ctx,
                           void* workspaceBase, int slice, int tempIndex) {
  if (tempIndex < 0 || tempIndex >= ctx->plan.tempCount) return nullptr;
  return (char*)workspaceBase + (size_t)slice * ctx->plan.sliceBytes +
      ctx->plan.temps[tempIndex].offset;
}

ncclResult_t cocclPipelineInputSlice(const cocclPipelineContext* ctx,
                                     void* workspaceBase, int slice,
                                     cudaStream_t stream, void** input) {
  if (input == nullptr) return ncclInvalidArgument;
  size_t typeSize = ncclTypeSize(ctx->spec->datatype);
  if (ctx->plan.inputStagingTemp < 0) {
    *input = (char*)const_cast<void*>(ctx->spec->input) +
        (size_t)slice * ctx->rawSliceBytes;
    return ncclSuccess;
  }

  void* packed = cocclPipelineTempPtr(
      ctx, workspaceBase, slice, ctx->plan.inputStagingTemp);
  size_t rawChunkBytes = 0;
  if (packed == nullptr ||
      !cocclCheckedMultiply(ctx->spec->rawChunkCount, typeSize,
                            &rawChunkBytes)) {
    return ncclInvalidArgument;
  }
  const void* source = (const char*)ctx->spec->input +
      (size_t)slice * ctx->rawSliceBytes;
  NCCLCHECK(cocclLaunchPackSlice(source, rawChunkBytes, packed,
                                 ctx->rawSliceBytes,
                                 ctx->spec->inputChunks, stream));
  *input = packed;
  return ncclSuccess;
}

void* cocclPipelineOutputSlice(const cocclPipelineContext* ctx,
                               void* workspaceBase, int slice) {
  if (ctx->plan.outputStagingTemp >= 0) {
    return cocclPipelineTempPtr(ctx, workspaceBase, slice,
                                ctx->plan.outputStagingTemp);
  }
  return (char*)ctx->spec->output + (size_t)slice * ctx->rawSliceBytes;
}

ncclResult_t cocclScatterPipelineOutput(const cocclPipelineContext* ctx,
                                        int slice, const void* packed,
                                        cudaStream_t stream) {
  if (ctx->plan.outputStagingTemp < 0) return ncclSuccess;
  size_t rawChunkBytes = 0;
  if (!cocclCheckedMultiply(ctx->spec->rawChunkCount,
                            ncclTypeSize(ctx->spec->datatype),
                            &rawChunkBytes)) {
    return ncclInvalidArgument;
  }
  void* destination = (char*)ctx->spec->output +
      (size_t)slice * ctx->rawSliceBytes;
  NCCLCHECK(cocclLaunchUnpackSlice(
      packed, destination, rawChunkBytes, ctx->rawSliceBytes,
      ctx->plan.finalChunks, stream));
  return ncclSuccess;
}

ncclResult_t cocclRunPipelineStage(cocclPipelineContext* ctx,
                                   int stageIndex, int slice,
                                   void* workspaceBase,
                                   cudaStream_t stream,
                                   cocclPipelineEdge* edge,
                                   void* initialInput = nullptr) {
  if (edge == nullptr) return ncclInvalidArgument;
  const auto* spec = ctx->spec;
  const auto& stage = spec->stages[stageIndex];
  if (stageIndex == 0) {
    if (initialInput == nullptr) {
      NCCLCHECK(cocclPipelineInputSlice(ctx, workspaceBase, slice, stream,
                                        &edge->ptr));
    } else {
      edge->ptr = initialInput;
    }
    edge->totalElements = ctx->rawSliceCount * spec->inputChunks;
    edge->datatype = spec->datatype;
    edge->logicalChunks = spec->inputChunks;
  }

  bool finalStage = stageIndex + 1 == spec->stageCount;
  void* outputPtr = finalStage
      ? cocclPipelineOutputSlice(ctx, workspaceBase, slice)
      : cocclPipelineTempPtr(ctx, workspaceBase, slice,
                             ctx->plan.stageOutputTemp[stageIndex]);
  if (edge->ptr == nullptr || outputPtr == nullptr) return ncclInvalidArgument;

  size_t outputChunks = 0;
  NCCLCHECK(cocclPipelineStageOutputChunks(stage, edge->logicalChunks,
                                           &outputChunks));

  NCCLCHECK(cocclExecutePipelineStage(
      &ctx->stageContext, &stage, edge, outputPtr, stream));

  edge->logicalChunks = outputChunks;
  return ncclSuccess;
}

ncclResult_t cocclRegisterPipelineCommunicators(
    const cocclPipelineSpec* spec, cocclBufferHandle* workspace) {
  ncclComm_t registered[kMaxPipelineStages + 1] = {};
  int registeredCount = 1;
  registered[0] = spec->ownerComm;

  for (int i = 0; i < spec->stageCount; ++i) {
    ncclComm_t comm = spec->stages[i].comm;
    if (comm == nullptr) continue;
    bool seen = false;
    for (int j = 0; j < registeredCount; ++j) {
      if (registered[j] == comm) {
        seen = true;
        break;
      }
    }
    if (seen) continue;
    NCCLCHECK(cocclRegisterBufferForComm(workspace, comm));
    registered[registeredCount++] = comm;
  }
  return ncclSuccess;
}

ncclResult_t cocclExecutePipelineSerial(cocclPipelineContext* ctx,
                                        void* workspaceBase) {
  cocclPipelineEdge edge = {};
  for (int stageIndex = 0; stageIndex < ctx->spec->stageCount; ++stageIndex) {
    NCCLCHECK(cocclRunPipelineStage(ctx, stageIndex, 0, workspaceBase,
                                    ctx->spec->stream, &edge));
  }
  NCCLCHECK(cocclScatterPipelineOutput(ctx, 0, edge.ptr,
                                       ctx->spec->stream));
  return ncclSuccess;
}

ncclResult_t cocclRunPipelineOverlap(cocclPipelineContext* ctx,
                                     void* workspaceBase) {
  const auto* spec = ctx->spec;
  cocclPipelineResources* resources = nullptr;
  NCCLCHECK(cocclFindOrCreatePipelineResources(
      spec->name, spec->ownerComm->cudaDev, spec->stageCount, &resources));

  CUDACHECK(cudaEventRecord(resources->mainEvent, spec->stream));
  CUDACHECK(cudaStreamWaitEvent(resources->streams[0], resources->mainEvent,
                                0));
  const bool usesPackStream = ctx->plan.inputStagingTemp >= 0;
  const bool usesUnpackStream = ctx->plan.outputStagingTemp >= 0;
  if (usesPackStream) {
    CUDACHECK(cudaStreamWaitEvent(resources->packStream, resources->mainEvent,
                                  0));
  }

  for (int slice = 0; slice < ctx->depth; ++slice) {
    cocclPipelineEdge edge = {};
    void* initialInput = nullptr;
    cudaStream_t inputStream = usesPackStream
        ? resources->packStream : resources->streams[0];
    NCCLCHECK(cocclPipelineInputSlice(ctx, workspaceBase, slice, inputStream,
                                      &initialInput));
    if (usesPackStream) {
      // Pack the next slice independently from Compress. The stage-0 stream
      // waits only for this slice's packed input, so Pack(N+1) may overlap
      // Compress(N) and later stages.
      CUDACHECK(cudaEventRecord(resources->packReadyEvent,
                                resources->packStream));
      CUDACHECK(cudaStreamWaitEvent(resources->streams[0],
                                    resources->packReadyEvent, 0));
    }

    for (int stageIndex = 0; stageIndex < spec->stageCount; ++stageIndex) {
      NCCLCHECK(cocclRunPipelineStage(ctx, stageIndex, slice, workspaceBase,
                                      resources->streams[stageIndex],
                                      &edge,
                                      stageIndex == 0 ? initialInput : nullptr));
      if (stageIndex + 1 < spec->stageCount) {
        CUDACHECK(cudaEventRecord(resources->events[stageIndex],
                                  resources->streams[stageIndex]));
        CUDACHECK(cudaStreamWaitEvent(resources->streams[stageIndex + 1],
                                      resources->events[stageIndex], 0));
      }
    }

    if (usesUnpackStream) {
      int last = spec->stageCount - 1;
      CUDACHECK(cudaEventRecord(resources->events[last],
                                resources->streams[last]));
      CUDACHECK(cudaStreamWaitEvent(resources->unpackStream,
                                    resources->events[last], 0));
      // The final stage writes one contiguous slice. Scatter it on a separate
      // stream so Unpack(N) may overlap the final stage for slice N+1.
      NCCLCHECK(cocclScatterPipelineOutput(ctx, slice, edge.ptr,
                                           resources->unpackStream));
    }
  }

  int last = spec->stageCount - 1;
  if (usesUnpackStream) {
    CUDACHECK(cudaEventRecord(resources->unpackDoneEvent,
                              resources->unpackStream));
    CUDACHECK(cudaStreamWaitEvent(spec->stream, resources->unpackDoneEvent,
                                  0));
  } else {
    CUDACHECK(cudaEventRecord(resources->events[last],
                              resources->streams[last]));
    CUDACHECK(cudaStreamWaitEvent(spec->stream, resources->events[last], 0));
  }
  return ncclSuccess;
}

ncclResult_t cocclRunPipelineWithDepth(const cocclPipelineSpec* spec,
                                       int64_t requestedDepth) {
  NCCLCHECK(cocclValidatePipelineSpec(spec));

  if (requestedDepth > INT_MAX) return ncclInvalidArgument;
  cocclPipelineContext ctx = {};
  ctx.spec = spec;
  ctx.depth = requestedDepth < 2 ? 1 : (int)requestedDepth;
  if (ctx.depth > 1 && spec->rawChunkCount % (size_t)ctx.depth != 0) {
    ctx.depth = 1;
  }

  CUDACHECK(cudaSetDevice(spec->ownerComm->cudaDev));
  NCCLCHECK(cocclPreparePipeline(&ctx));

  bool userBuffersOverlap = false;
  NCCLCHECK(cocclPipelineUserBuffersOverlap(&ctx, &userBuffersOverlap));
  if (ctx.depth > 1 && userBuffersOverlap) {
    ctx.depth = 1;
    ctx.plan = {};
    NCCLCHECK(cocclPreparePipeline(&ctx));
  }

  cocclBufferHandle workspace = {};
  NCCLCHECK(cocclGetBuffer(spec->ownerComm, ctx.plan.workspaceBytes,
                           &workspace));
  NCCLCHECK(cocclRegisterPipelineCommunicators(spec, &workspace));

  if (ctx.depth == 1) {
    NCCLCHECK(cocclExecutePipelineSerial(&ctx, workspace.ptr));
  } else {
    NCCLCHECK(cocclRunPipelineOverlap(&ctx, workspace.ptr));
  }

  NCCLCHECK(cocclReleaseBuffer(&workspace, spec->stream));
  return ncclSuccess;
}

}  // namespace

ncclResult_t cocclRunPipeline(const cocclPipelineSpec* spec) {
  return cocclRunPipelineWithDepth(spec, ncclParamPipelineDepth());
}

ncclResult_t cocclRunPipelineSerial(const cocclPipelineSpec* spec) {
  return cocclRunPipelineWithDepth(spec, 1);
}
