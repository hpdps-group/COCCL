#include "core/config/coccl_config.h"
#include "core/memory/coccl_buffer_management.h"
#include "core/pipeline/coccl_pipeline.h"
#include "coccl_pipeline_internal.h"
#include "core/pipeline/coccl_pipeline_stage.h"
#include "checks.h"
#include "comm.h"

#include <stdlib.h>
#include <string.h>

namespace {

struct cocclPipelineResources {
  const char* name;
  int cudaDev;
  int stageCount;
  int depth;
  cudaStream_t* streams;
  cudaEvent_t* events;
  cudaEvent_t* packReadyEvents;
  cocclPipelineEdge* edges;
  cocclPipelineStageOutput* stageOutputs;
  cudaStream_t packStream;
  cudaStream_t unpackStream;
  cudaEvent_t unpackDoneEvent;
  cudaEvent_t inputRawConsumed[kPipelineRawRingSlots];
  cudaEvent_t outputRawConsumed[kPipelineRawRingSlots];
  cudaEvent_t mainEvent;
  cocclPipelineFrameResources frameResources;
  cocclPipelineResources* next;
};

struct cocclPipelineWorkspace {
  void* registeredBase;
  void* rawBase;
};

__thread cocclPipelineResources* pipelineResources = nullptr;

ncclResult_t cocclFindOrCreatePipelineResources(
    const char* name, int cudaDev, int stageCount, int depth,
    cocclPipelineResources** out) {
  for (cocclPipelineResources* cur = pipelineResources; cur != nullptr;
       cur = cur->next) {
    if (cur->cudaDev == cudaDev && cur->stageCount == stageCount &&
        cur->depth == depth &&
        (cur->name == name || strcmp(cur->name, name) == 0)) {
      *out = cur;
      return ncclSuccess;
    }
  }

  cocclPipelineResources* res =
      (cocclPipelineResources*)calloc(1, sizeof(cocclPipelineResources));
  if (res == nullptr) return ncclSystemError;
  res->streams = (cudaStream_t*)calloc(stageCount, sizeof(cudaStream_t));
  res->events = (cudaEvent_t*)calloc(
      (size_t)stageCount * depth, sizeof(cudaEvent_t));
  res->packReadyEvents =
      (cudaEvent_t*)calloc(depth, sizeof(cudaEvent_t));
  res->edges =
      (cocclPipelineEdge*)calloc(depth, sizeof(cocclPipelineEdge));
  res->stageOutputs = (cocclPipelineStageOutput*)calloc(
      depth, sizeof(cocclPipelineStageOutput));
  if (res->streams == nullptr || res->events == nullptr ||
      res->packReadyEvents == nullptr || res->edges == nullptr ||
      res->stageOutputs == nullptr) {
    free(res->streams);
    free(res->events);
    free(res->packReadyEvents);
    free(res->edges);
    free(res->stageOutputs);
    free(res);
    return ncclSystemError;
  }

  res->name = name;
  res->cudaDev = cudaDev;
  res->stageCount = stageCount;
  res->depth = depth;
  for (int i = 0; i < stageCount; ++i) {
    CUDACHECK(cudaStreamCreateWithFlags(res->streams + i,
                                        cudaStreamNonBlocking));
    for (int slice = 0; slice < depth; ++slice) {
      CUDACHECK(cudaEventCreateWithFlags(
          res->events + (size_t)i * depth + slice,
          cudaEventDisableTiming));
    }
  }
  CUDACHECK(cudaStreamCreateWithFlags(&res->packStream,
                                      cudaStreamNonBlocking));
  CUDACHECK(cudaStreamCreateWithFlags(&res->unpackStream,
                                      cudaStreamNonBlocking));
  for (int slice = 0; slice < depth; ++slice) {
    CUDACHECK(cudaEventCreateWithFlags(
        res->packReadyEvents + slice, cudaEventDisableTiming));
  }
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

cudaEvent_t cocclPipelineStageEvent(
    const cocclPipelineResources* resources, int stage, int slice) {
  return resources->events[(size_t)stage * resources->depth + slice];
}

void* cocclPipelineTempPtr(const cocclPipelineContext* ctx,
                           const cocclPipelineWorkspace& workspace,
                           int slice, int tempIndex) {
  const auto& temp = ctx->plan.temps[tempIndex];
  if (temp.storage == cocclPipelineRawRing) {
    return (char*)workspace.rawBase + temp.offset +
        (size_t)(slice % kPipelineRawRingSlots) * temp.bytes;
  }
  return (char*)workspace.registeredBase +
      (size_t)slice * ctx->plan.registeredSliceBytes + temp.offset;
}

cocclPipelineStageOutput cocclPipelineTempOutput(
    const cocclPipelineContext* ctx,
    const cocclPipelineWorkspace& workspace, int slice, int tempIndex,
    size_t capacityBytes) {
  const auto& temp = ctx->plan.temps[tempIndex];
  void* payload = cocclPipelineTempPtr(ctx, workspace, slice, tempIndex);
  return {
      payload,
      capacityBytes,
      temp.frameMetadataBytes == 0
          ? nullptr
          : reinterpret_cast<cocclCompressorFrameMetadata*>(
                (char*)payload + temp.frameMetadataOffset),
      temp.frameStrideBytes,
  };
}

ncclResult_t cocclPipelineInputSlice(const cocclPipelineContext* ctx,
                                     const cocclPipelineWorkspace& workspace,
                                     int slice,
                                     cudaStream_t stream,
                                     cocclPipelineEdge* edge) {
  edge->ptr = (char*)const_cast<void*>(ctx->spec->input) +
      (size_t)slice * ctx->rawSliceBytes;
  edge->bytes = ctx->rawSliceBytes * ctx->spec->inputChunks;
  edge->totalElements = ctx->rawSliceCount * ctx->spec->inputChunks;
  edge->datatype = ctx->spec->datatype;
  edge->logicalChunks = ctx->spec->inputChunks;
  edge->frameMetadata = nullptr;
  edge->frameStrideBytes = 0;
  if (ctx->plan.inputStagingTemp < 0) return ncclSuccess;

  void* packed = cocclPipelineTempPtr(
      ctx, workspace, slice, ctx->plan.inputStagingTemp);
  const cocclPipelineStage packStage = cocclPipelinePack();
  const cocclPipelineStageOutput output = {
      packed, edge->bytes, nullptr, 0};
  return cocclExecutePipelineStage(
      &ctx->stageContext, &packStage, edge, &output, stream);
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
  if (ctx->plan.outputStagingTemp < 0) return ncclSuccess;
  void* destination = (char*)ctx->spec->output +
      (size_t)slice * ctx->rawSliceBytes;
  const cocclPipelineStage unpackStage = cocclPipelineUnpack();
  const cocclPipelineStageOutput output = {
      destination, edge->bytes, nullptr, 0};
  return cocclExecutePipelineStage(
      &ctx->stageContext, &unpackStage, edge, &output, stream);
}

cocclPipelineStageOutput cocclResolvePipelineStageOutput(
    const cocclPipelineContext* ctx, int stageIndex, int slice,
    const cocclPipelineWorkspace& workspace) {
  const auto* spec = ctx->spec;
  const size_t outputCapacityBytes =
      ctx->plan.stageOutputCapacityBytes[stageIndex];
  const bool finalStage = stageIndex + 1 == spec->stageCount;
  const int outputTemp = finalStage
      ? ctx->plan.outputStagingTemp
      : ctx->plan.stageOutputTemp[stageIndex];
  if (outputTemp >= 0) {
    return cocclPipelineTempOutput(
        ctx, workspace, slice, outputTemp, outputCapacityBytes);
  }
  return {cocclPipelineOutputSlice(ctx, workspace, slice),
          outputCapacityBytes, nullptr, 0};
}

ncclResult_t cocclRunPipelineStage(cocclPipelineContext* ctx,
                                   int stageIndex, int slice,
                                   const cocclPipelineWorkspace& workspace,
                                   cudaStream_t stream,
                                   cocclPipelineEdge* edge) {
  const cocclPipelineStageOutput output = cocclResolvePipelineStageOutput(
      ctx, stageIndex, slice, workspace);
  const auto& stage = ctx->spec->stages[stageIndex];
  return cocclExecutePipelineStage(
      &ctx->stageContext, &stage, edge, &output, stream);
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

ncclResult_t cocclFinishPipelineOverlap(
    const cocclPipelineContext* ctx, cocclPipelineResources* resources,
    bool usesUnpackStream) {
  const int last = ctx->spec->stageCount - 1;
  if (usesUnpackStream) {
    CUDACHECK(cudaEventRecord(resources->unpackDoneEvent,
                              resources->unpackStream));
    CUDACHECK(cudaStreamWaitEvent(ctx->spec->stream,
                                  resources->unpackDoneEvent, 0));
  } else {
    CUDACHECK(cudaStreamWaitEvent(
        ctx->spec->stream,
        cocclPipelineStageEvent(resources, last, ctx->depth - 1), 0));
  }
  return ncclSuccess;
}

ncclResult_t cocclRunFramedPipelineOverlap(
    cocclPipelineContext* ctx, const cocclPipelineWorkspace& workspace,
    cocclPipelineResources* resources, bool usesPackStream,
    bool usesUnpackStream, bool reusesInputRaw, bool reusesOutputRaw) {
  const auto* spec = ctx->spec;

  // Fill the codec pipeline first. This keeps Pack and Compress overlap while
  // ensuring every variable communication stage sees all slice metadata
  // collectives before the first Host length read.
  for (int slice = 0; slice < ctx->depth; ++slice) {
    const int rawSlot = slice % kPipelineRawRingSlots;
    cocclPipelineEdge* edge = resources->edges + slice;
    *edge = {};
    cudaStream_t inputStream = usesPackStream
        ? resources->packStream : resources->streams[0];
    if (reusesInputRaw && slice >= kPipelineRawRingSlots) {
      CUDACHECK(cudaStreamWaitEvent(
          resources->packStream, resources->inputRawConsumed[rawSlot], 0));
    }
    NCCLCHECK(cocclPipelineInputSlice(
        ctx, workspace, slice, inputStream, edge));
    if (usesPackStream) {
      CUDACHECK(cudaEventRecord(resources->packReadyEvents[slice],
                                resources->packStream));
      CUDACHECK(cudaStreamWaitEvent(resources->streams[0],
                                    resources->packReadyEvents[slice], 0));
    }
    NCCLCHECK(cocclRunPipelineStage(
        ctx, 0, slice, workspace, resources->streams[0], edge));
    if (reusesInputRaw) {
      CUDACHECK(cudaEventRecord(resources->inputRawConsumed[rawSlot],
                                resources->streams[0]));
    }
    CUDACHECK(cudaEventRecord(
        cocclPipelineStageEvent(resources, 0, slice),
        resources->streams[0]));
  }

  for (int stageIndex = 1; stageIndex < spec->stageCount; ++stageIndex) {
    const auto& stage = spec->stages[stageIndex];
    const bool variable = cocclPipelineStageUsesFrameExchange(
        stage, resources->edges[0]);
    const bool finalStage = stageIndex + 1 == spec->stageCount;

    if (variable) {
      for (int slice = 0; slice < ctx->depth; ++slice) {
        cocclPipelineEdge* edge = resources->edges + slice;
        CUDACHECK(cudaStreamWaitEvent(
            resources->streams[stageIndex],
            cocclPipelineStageEvent(resources, stageIndex - 1, slice), 0));
        resources->stageOutputs[slice] = cocclResolvePipelineStageOutput(
            ctx, stageIndex, slice, workspace);
        NCCLCHECK(cocclPreparePipelineFrameExchange(
            &ctx->stageContext, &stage, edge,
            resources->stageOutputs + slice,
            resources->streams[stageIndex]));
      }
      for (int slice = 0; slice < ctx->depth; ++slice) {
        cocclPipelineEdge* edge = resources->edges + slice;
        NCCLCHECK(cocclCommitPipelineFrameExchange(
            &ctx->stageContext, &stage, edge,
            resources->stageOutputs + slice,
            resources->streams[stageIndex]));
        CUDACHECK(cudaEventRecord(
            cocclPipelineStageEvent(resources, stageIndex, slice),
            resources->streams[stageIndex]));
      }
      continue;
    }

    for (int slice = 0; slice < ctx->depth; ++slice) {
      const int rawSlot = slice % kPipelineRawRingSlots;
      CUDACHECK(cudaStreamWaitEvent(
          resources->streams[stageIndex],
          cocclPipelineStageEvent(resources, stageIndex - 1, slice), 0));
      if (reusesOutputRaw && finalStage &&
          slice >= kPipelineRawRingSlots) {
        CUDACHECK(cudaStreamWaitEvent(
            resources->streams[stageIndex],
            resources->outputRawConsumed[rawSlot], 0));
      }
      cocclPipelineEdge* edge = resources->edges + slice;
      NCCLCHECK(cocclRunPipelineStage(
          ctx, stageIndex, slice, workspace,
          resources->streams[stageIndex], edge));
      CUDACHECK(cudaEventRecord(
          cocclPipelineStageEvent(resources, stageIndex, slice),
          resources->streams[stageIndex]));

      if (finalStage && usesUnpackStream) {
        CUDACHECK(cudaStreamWaitEvent(
            resources->unpackStream,
            cocclPipelineStageEvent(resources, stageIndex, slice), 0));
        NCCLCHECK(cocclScatterPipelineOutput(
            ctx, slice, edge, resources->unpackStream));
        if (reusesOutputRaw) {
          CUDACHECK(cudaEventRecord(
              resources->outputRawConsumed[rawSlot],
              resources->unpackStream));
        }
      }
    }
  }
  return cocclFinishPipelineOverlap(ctx, resources, usesUnpackStream);
}

ncclResult_t cocclRunPipelineOverlap(cocclPipelineContext* ctx,
                                     const cocclPipelineWorkspace& workspace,
                                     cocclPipelineResources* resources) {
  const auto* spec = ctx->spec;
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
  if (cocclCompressorSupports(
          spec->compressor, cocclCompressorCapabilityFramed)) {
    return cocclRunFramedPipelineOverlap(
        ctx, workspace, resources, usesPackStream, usesUnpackStream,
        reusesInputRaw, reusesOutputRaw);
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
      CUDACHECK(cudaEventRecord(resources->packReadyEvents[slice],
                                resources->packStream));
      CUDACHECK(cudaStreamWaitEvent(resources->streams[0],
                                    resources->packReadyEvents[slice], 0));
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
      CUDACHECK(cudaEventRecord(
                                cocclPipelineStageEvent(
                                    resources, stageIndex, slice),
                                resources->streams[stageIndex]));
      if (stageIndex + 1 < spec->stageCount) {
        CUDACHECK(cudaStreamWaitEvent(resources->streams[stageIndex + 1],
                                      cocclPipelineStageEvent(
                                          resources, stageIndex, slice),
                                      0));
      }
    }

    if (usesUnpackStream) {
      int last = spec->stageCount - 1;
      CUDACHECK(cudaStreamWaitEvent(resources->unpackStream,
                                    cocclPipelineStageEvent(
                                        resources, last, slice),
                                    0));
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

  return cocclFinishPipelineOverlap(ctx, resources, usesUnpackStream);
}

ncclResult_t cocclRunPipelineWithDepth(const cocclPipelineSpec* spec,
                                       int64_t requestedDepth) {
  NCCLCHECK(cocclValidatePipelineSpec(spec));

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

  cocclPipelineResources* resources = nullptr;
  result = cocclFindOrCreatePipelineResources(
      spec->name, spec->ownerComm->cudaDev, spec->stageCount, ctx.depth,
      &resources);
  if (result != ncclSuccess) {
    cocclReleaseBuffer(&rawWorkspace, spec->stream);
    cocclReleaseBuffer(&registeredWorkspace, spec->stream);
    return result;
  }
  ctx.stageContext.frameResources = &resources->frameResources;

  if (ctx.depth == 1) {
    result = cocclExecutePipelineSerial(&ctx, workspace);
  } else {
    result = cocclRunPipelineOverlap(&ctx, workspace, resources);
  }

  const ncclResult_t rawRelease =
      cocclReleaseBuffer(&rawWorkspace, spec->stream);
  const ncclResult_t registeredRelease =
      cocclReleaseBuffer(&registeredWorkspace, spec->stream);
  if (result == ncclSuccess) result = rawRelease;
  if (result == ncclSuccess) result = registeredRelease;
  return result;
}

}  // namespace

ncclResult_t cocclRunPipeline(const cocclPipelineSpec* spec) {
  return cocclRunPipelineWithDepth(spec, cocclGetConfig().pipeline.depth);
}

ncclResult_t cocclRunPipelineSerial(const cocclPipelineSpec* spec) {
  return cocclRunPipelineWithDepth(spec, 1);
}
