#include "config/coccl_config.h"
#include "pipeline/coccl_pipeline.h"
#include "coccl_pipeline_internal.h"
#include "pipeline/coccl_pipeline_stage.h"
#include "primitives/coccl_primitives_internal.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

namespace {

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
  cudaEvent_t inputRawConsumed[kPipelineRawRingSlots];
  cudaEvent_t outputRawConsumed[kPipelineRawRingSlots];
  cudaEvent_t mainEvent;
  cocclPipelineResources* next;
};

struct cocclPipelineWorkspace {
  void* registeredBase;
  void* rawBase;
};

__thread cocclPipelineResources* pipelineResources = nullptr;

ncclResult_t cocclFindOrCreatePipelineResources(
    const char* name, int cudaDev, int stageCount,
    cocclPipelineResources** out) {
  if (name == nullptr || cudaDev < 0 || stageCount <= 0 || out == nullptr) {
    return ncclInvalidArgument;
  }

  for (cocclPipelineResources* cur = pipelineResources; cur != nullptr;
       cur = cur->next) {
    if (cur->cudaDev == cudaDev && cur->stageCount == stageCount &&
        (cur->name == name || strcmp(cur->name, name) == 0)) {
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
    CUDACHECK(cudaEventCreateWithFlags(
        res->events + i, cudaEventDisableTiming));
  }
  CUDACHECK(cudaStreamCreateWithFlags(&res->packStream,
                                      cudaStreamNonBlocking));
  CUDACHECK(cudaStreamCreateWithFlags(&res->unpackStream,
                                      cudaStreamNonBlocking));
  CUDACHECK(cudaEventCreateWithFlags(&res->packReadyEvent,
                                     cudaEventDisableTiming));
  CUDACHECK(cudaEventCreateWithFlags(&res->unpackDoneEvent,
                                     cudaEventDisableTiming));
  for (int slot = 0; slot < kPipelineRawRingSlots; ++slot) {
    CUDACHECK(cudaEventCreateWithFlags(
        res->inputRawConsumed + slot, cudaEventDisableTiming));
    CUDACHECK(cudaEventCreateWithFlags(
        res->outputRawConsumed + slot, cudaEventDisableTiming));
  }
  CUDACHECK(cudaEventCreateWithFlags(
      &res->mainEvent, cudaEventDisableTiming));

  res->next = pipelineResources;
  pipelineResources = res;
  *out = res;
  return ncclSuccess;
}

void* cocclPipelineTempPtr(const cocclPipelineContext* ctx,
                           const cocclPipelineWorkspace& workspace,
                           int slice, int tempIndex) {
  if (tempIndex < 0 || tempIndex >= ctx->plan.tempCount) return nullptr;
  const auto& temp = ctx->plan.temps[tempIndex];
  if (temp.storage == cocclPipelineRawRing) {
    if (workspace.rawBase == nullptr) return nullptr;
    return (char*)workspace.rawBase + temp.offset +
        (size_t)(slice % kPipelineRawRingSlots) * temp.bytes;
  }
  if (workspace.registeredBase == nullptr) return nullptr;
  return (char*)workspace.registeredBase +
      (size_t)slice * ctx->plan.registeredSliceBytes + temp.offset;
}

ncclResult_t cocclPipelineInputSlice(const cocclPipelineContext* ctx,
                                     const cocclPipelineWorkspace& workspace,
                                     int slice,
                                     cudaStream_t stream,
                                     cocclPipelineEdge* edge) {
  if (edge == nullptr) return ncclInvalidArgument;
  edge->ptr = (char*)const_cast<void*>(ctx->spec->input) +
      (size_t)slice * ctx->rawSliceBytes;
  if (!cocclPipelineCheckedMultiply(ctx->rawSliceBytes,
                                    ctx->spec->inputChunks, &edge->bytes) ||
      !cocclPipelineCheckedMultiply(ctx->rawSliceCount,
                                    ctx->spec->inputChunks,
                                    &edge->totalElements)) {
    return ncclInvalidArgument;
  }
  edge->datatype = ctx->spec->datatype;
  edge->logicalChunks = ctx->spec->inputChunks;
  if (ctx->plan.inputStagingTemp < 0) return ncclSuccess;

  void* packed = cocclPipelineTempPtr(
      ctx, workspace, slice, ctx->plan.inputStagingTemp);
  if (packed == nullptr) return ncclInvalidArgument;
  const cocclPipelineStage packStage = cocclPipelinePack();
  return cocclExecutePipelineStage(
      &ctx->stageContext, &packStage, edge, packed, edge->bytes, stream);
}

void* cocclPipelineOutputSlice(const cocclPipelineContext* ctx,
                               const cocclPipelineWorkspace& workspace,
                               int slice) {
  if (ctx->plan.outputStagingTemp >= 0) {
    return cocclPipelineTempPtr(ctx, workspace, slice,
                                ctx->plan.outputStagingTemp);
  }
  return (char*)ctx->spec->output + (size_t)slice * ctx->rawSliceBytes;
}

ncclResult_t cocclScatterPipelineOutput(const cocclPipelineContext* ctx,
                                        int slice, cocclPipelineEdge* edge,
                                        cudaStream_t stream) {
  if (edge == nullptr) return ncclInvalidArgument;
  if (ctx->plan.outputStagingTemp < 0) return ncclSuccess;
  void* destination = (char*)ctx->spec->output +
      (size_t)slice * ctx->rawSliceBytes;
  const cocclPipelineStage unpackStage = cocclPipelineUnpack();
  return cocclExecutePipelineStage(
      &ctx->stageContext, &unpackStage, edge, destination, edge->bytes,
      stream);
}

ncclResult_t cocclRunPipelineStage(cocclPipelineContext* ctx,
                                   int stageIndex, int slice,
                                   const cocclPipelineWorkspace& workspace,
                                   cudaStream_t stream,
                                   cocclPipelineEdge* edge) {
  if (edge == nullptr) return ncclInvalidArgument;
  const auto* spec = ctx->spec;
  const auto& stage = spec->stages[stageIndex];

  bool finalStage = stageIndex + 1 == spec->stageCount;
  void* outputPtr = finalStage
      ? cocclPipelineOutputSlice(ctx, workspace, slice)
      : cocclPipelineTempPtr(ctx, workspace, slice,
                             ctx->plan.stageOutputTemp[stageIndex]);
  if (edge->ptr == nullptr || outputPtr == nullptr) return ncclInvalidArgument;

  size_t outputChunks = 0;
  NCCLCHECK(cocclPipelineStageOutputChunks(stage, edge->logicalChunks,
                                           &outputChunks));

  const size_t outputCapacityBytes =
      ctx->plan.stageOutputCapacityBytes[stageIndex];
  if (outputCapacityBytes == 0) return ncclInternalError;
  const int outputTemp = finalStage
      ? ctx->plan.outputStagingTemp
      : ctx->plan.stageOutputTemp[stageIndex];
  if (outputTemp >= 0) {
    if (outputTemp >= ctx->plan.tempCount) {
      return ncclInvalidArgument;
    }
    // Physical temp ranges include alignment padding; stages see only the
    // unaligned logical capacity derived during shape planning.
    if (outputCapacityBytes > ctx->plan.temps[outputTemp].bytes) {
      return ncclInternalError;
    }
  } else if (!finalStage) {
    return ncclInvalidArgument;
  }
  NCCLCHECK(cocclExecutePipelineStage(
      &ctx->stageContext, &stage, edge, outputPtr, outputCapacityBytes,
      stream));

  size_t edgeStorageBytes = 0;
  const int edgeTypeBytes = ncclTypeSize(edge->datatype);
  if (edge->ptr != outputPtr || edge->logicalChunks != outputChunks ||
      edge->logicalChunks == 0 || edge->bytes == 0 ||
      edge->bytes > outputCapacityBytes ||
      edge->totalElements == 0 || edgeTypeBytes <= 0 ||
      !cocclPipelineCheckedMultiply(edge->totalElements,
                                    (size_t)edgeTypeBytes,
                                    &edgeStorageBytes) ||
      edgeStorageBytes != edge->bytes ||
      edge->totalElements % edge->logicalChunks != 0 ||
      edge->bytes % edge->logicalChunks != 0) {
    return ncclInvalidUsage;
  }
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
                                        const cocclPipelineWorkspace& workspace) {
  cocclPipelineEdge edge = {};
  NCCLCHECK(cocclPipelineInputSlice(
      ctx, workspace, 0, ctx->spec->stream, &edge));
  for (int stageIndex = 0; stageIndex < ctx->spec->stageCount; ++stageIndex) {
    NCCLCHECK(cocclRunPipelineStage(ctx, stageIndex, 0, workspace,
                                    ctx->spec->stream, &edge));
  }
  NCCLCHECK(cocclScatterPipelineOutput(ctx, 0, &edge, ctx->spec->stream));
  return ncclSuccess;
}

ncclResult_t cocclRunPipelineOverlap(cocclPipelineContext* ctx,
                                     const cocclPipelineWorkspace& workspace) {
  const auto* spec = ctx->spec;
  cocclPipelineResources* resources = nullptr;
  NCCLCHECK(cocclFindOrCreatePipelineResources(
      spec->name, spec->ownerComm->cudaDev, spec->stageCount, &resources));

  CUDACHECK(cudaEventRecord(resources->mainEvent, spec->stream));
  CUDACHECK(cudaStreamWaitEvent(resources->streams[0], resources->mainEvent,
                                0));
  const bool usesPackStream = ctx->plan.inputStagingTemp >= 0;
  const bool usesUnpackStream = ctx->plan.outputStagingTemp >= 0;
  const bool reusesInputRaw =
      ctx->plan.workspaceKind == cocclPipelineWorkspaceSplit &&
      usesPackStream;
  const bool reusesOutputRaw =
      ctx->plan.workspaceKind == cocclPipelineWorkspaceSplit &&
      usesUnpackStream;
  if (usesPackStream) {
    CUDACHECK(cudaStreamWaitEvent(resources->packStream, resources->mainEvent,
                                  0));
  }

  for (int slice = 0; slice < ctx->depth; ++slice) {
    const int rawSlot = slice % kPipelineRawRingSlots;
    cocclPipelineEdge edge = {};
    cudaStream_t inputStream = usesPackStream
        ? resources->packStream : resources->streams[0];
    if (reusesInputRaw && slice >= kPipelineRawRingSlots) {
      CUDACHECK(cudaStreamWaitEvent(
          resources->packStream, resources->inputRawConsumed[rawSlot], 0));
    }
    NCCLCHECK(cocclPipelineInputSlice(ctx, workspace, slice, inputStream,
                                      &edge));
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
      const bool finalStage = stageIndex + 1 == spec->stageCount;
      if (reusesOutputRaw && finalStage &&
          slice >= kPipelineRawRingSlots) {
        CUDACHECK(cudaStreamWaitEvent(
            resources->streams[stageIndex],
            resources->outputRawConsumed[rawSlot], 0));
      }
      NCCLCHECK(cocclRunPipelineStage(ctx, stageIndex, slice, workspace,
                                      resources->streams[stageIndex],
                                      &edge));
      if (reusesInputRaw && stageIndex == 0) {
        CUDACHECK(cudaEventRecord(resources->inputRawConsumed[rawSlot],
                                  resources->streams[stageIndex]));
      }
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
      NCCLCHECK(cocclScatterPipelineOutput(ctx, slice, &edge,
                                           resources->unpackStream));
      if (reusesOutputRaw) {
        CUDACHECK(cudaEventRecord(resources->outputRawConsumed[rawSlot],
                                  resources->unpackStream));
      }
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

  bool requireSerial = false;
  NCCLCHECK(cocclPipelineUserBuffersRequireSerial(
      spec->input, spec->inputChunks, spec->output, ctx.plan.finalChunks,
      ctx.stageContext.rawChunkBytes, spec->ownerComm->rank,
      spec->inPlaceLayout, &requireSerial));
  if (ctx.depth > 1 && requireSerial) {
    ctx.depth = 1;
    ctx.plan = {};
    NCCLCHECK(cocclPreparePipeline(&ctx));
  }

  cocclBufferHandle registeredWorkspace = {};
  NCCLCHECK(cocclGetBuffer(spec->ownerComm, ctx.plan.registeredBytes,
                           &registeredWorkspace));
  ncclResult_t result =
      cocclRegisterPipelineCommunicators(spec, &registeredWorkspace);
  if (result != ncclSuccess) {
    cocclReleaseBuffer(&registeredWorkspace, spec->stream);
    return result;
  }

  cocclBufferHandle rawWorkspace = {};
  if (ctx.plan.rawBytes > 0) {
    result = cocclGetUnregisteredBuffer(
        spec->ownerComm, ctx.plan.rawBytes, &rawWorkspace);
    if (result != ncclSuccess) {
      cocclReleaseBuffer(&registeredWorkspace, spec->stream);
      return result;
    }
  }
  const cocclPipelineWorkspace workspace = {
      registeredWorkspace.ptr, rawWorkspace.ptr};

  if (ctx.depth == 1) {
    NCCLCHECK(cocclExecutePipelineSerial(&ctx, workspace));
  } else {
    NCCLCHECK(cocclRunPipelineOverlap(&ctx, workspace));
  }

  NCCLCHECK(cocclReleaseBuffer(&rawWorkspace, spec->stream));
  NCCLCHECK(cocclReleaseBuffer(&registeredWorkspace, spec->stream));
  return ncclSuccess;
}

}  // namespace

ncclResult_t cocclRunPipeline(const cocclPipelineSpec* spec) {
  return cocclRunPipelineWithDepth(spec, cocclGetConfig().pipeline.depth);
}

ncclResult_t cocclRunPipelineSerial(const cocclPipelineSpec* spec) {
  return cocclRunPipelineWithDepth(spec, 1);
}
