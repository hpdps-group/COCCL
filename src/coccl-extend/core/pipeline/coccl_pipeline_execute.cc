#include "core/pipeline/coccl_pipeline.h"

#include "checks.h"
#include "core/memory/coccl_buffer_management.h"
#include "core/config/coccl_config.h"
#include "core/pipeline/coccl_pipeline_internal.h"
#include "core/tuning/coccl_autotune_pipeline.h"
#include "comm.h"
#include "core/compression/compress.h"

#include <stdlib.h>

#include <algorithm>
#include <map>
#include <new>
#include <vector>

namespace {

enum cocclPipelinePhase {
  cocclPipelinePhasePack = 0,
  cocclPipelinePhaseFirstStage = 1,
  cocclPipelinePhaseUnpack = kCocclPipelinePhysicalStages - 1,
};

struct cocclPipelineResources {
  int depth;
  cudaStream_t streams[kCocclPipelinePhysicalStages];
  cudaEvent_t events[kCocclPipelinePhysicalStages]
                    [kCocclPipelineMaxDepth];
  cudaEvent_t inputReady;
  cudaEvent_t inputRawConsumed[kCocclPipelineRawRingSlots];
  cudaEvent_t outputRawConsumed[kCocclPipelineRawRingSlots];
  cocclPipelineEdge edges[kCocclPipelineMaxDepth];
  cocclPipelineStageOutput stageOutputs[kCocclPipelineMaxDepth];
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

struct cocclPipelineBatchState {
  cocclPipelineExecution execution;
  cocclPipelineEdge edges[kCocclPipelineMaxDepth];
  cocclPipelineStageOutput outputs[kCocclPipelineMaxDepth];
  int communicationStage;
};

struct alignas(16) cocclPipelineBatchPlan {
  uint64_t targetSliceBytes;
  uint32_t maxDepth;
  uint32_t reserved;
};

static_assert(sizeof(cocclPipelineBatchPlan) == 16,
              "Pipeline batch plan is a fixed control message");

thread_local std::map<ncclComm_t, cocclPipelineResources*> resourcesByComm;

void destroyResources(cocclPipelineResources* resources) {
  for (int phase = 0; phase < kCocclPipelinePhysicalStages; ++phase) {
    for (int slice = 0; slice < resources->depth; ++slice) {
      if (resources->events[phase][slice] != nullptr) {
        (void)cudaEventDestroy(resources->events[phase][slice]);
      }
    }
    if (resources->streams[phase] != nullptr) {
      (void)cudaStreamDestroy(resources->streams[phase]);
    }
  }
  if (resources->inputReady != nullptr) {
    (void)cudaEventDestroy(resources->inputReady);
  }
  for (int slot = 0; slot < kCocclPipelineRawRingSlots; ++slot) {
    if (resources->inputRawConsumed[slot] != nullptr) {
      (void)cudaEventDestroy(resources->inputRawConsumed[slot]);
    }
    if (resources->outputRawConsumed[slot] != nullptr) {
      (void)cudaEventDestroy(resources->outputRawConsumed[slot]);
    }
  }
  if (resources->frameResources.sendMetadata != nullptr) {
    (void)cudaFreeHost(resources->frameResources.sendMetadata);
  }
  if (resources->frameResources.recvMetadata != nullptr) {
    (void)cudaFreeHost(resources->frameResources.recvMetadata);
  }
  free(resources->frameResources.exchanges);
  delete resources;
}

ncclResult_t createResources(cocclPipelineResources** output) {
  cocclPipelineResources* resources =
      new (std::nothrow) cocclPipelineResources();
  if (resources == nullptr) return ncclSystemError;
  *resources = {};
  resources->depth = kCocclPipelineMaxDepth;

  int leastPriority = 0;
  int greatestPriority = 0;
  CUDACHECK(cudaDeviceGetStreamPriorityRange(&leastPriority,
                                              &greatestPriority));
  for (int phase = 0; phase < kCocclPipelinePhysicalStages; ++phase) {
    const int priority = phase == cocclPipelinePhasePack ||
            phase == cocclPipelinePhaseUnpack
        ? leastPriority
        : greatestPriority;
    cudaError_t cudaResult = cudaStreamCreateWithPriority(
        &resources->streams[phase], cudaStreamNonBlocking, priority);
    if (cudaResult != cudaSuccess) {
      destroyResources(resources);
      return ncclUnhandledCudaError;
    }
    for (int slice = 0; slice < resources->depth; ++slice) {
      cudaResult = cudaEventCreateWithFlags(
          &resources->events[phase][slice], cudaEventDisableTiming);
      if (cudaResult != cudaSuccess) {
        destroyResources(resources);
        return ncclUnhandledCudaError;
      }
    }
  }
  if (cudaEventCreateWithFlags(&resources->inputReady,
                               cudaEventDisableTiming) != cudaSuccess) {
    destroyResources(resources);
    return ncclUnhandledCudaError;
  }
  for (int slot = 0; slot < kCocclPipelineRawRingSlots; ++slot) {
    if (cudaEventCreateWithFlags(resources->inputRawConsumed + slot,
                                 cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(resources->outputRawConsumed + slot,
                                 cudaEventDisableTiming) != cudaSuccess) {
      destroyResources(resources);
      return ncclUnhandledCudaError;
    }
  }
  *output = resources;
  return ncclSuccess;
}

ncclResult_t findOrCreateResources(ncclComm_t comm,
                                   cocclPipelineResources** output) {
  auto found = resourcesByComm.find(comm);
  if (found != resourcesByComm.end()) {
    *output = found->second;
    return ncclSuccess;
  }

  cocclPipelineResources* resources = nullptr;
  // Auto slice selection may choose a different depth for each message.
  ncclResult_t result = createResources(&resources);
  if (result == ncclSuccess) {
    resourcesByComm.emplace(comm, resources);
    *output = resources;
  }
  return result;
}

void* tempPtr(const cocclPipelineContext& context,
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

cocclPipelineStageContext stageContextForSlice(
    const cocclPipelineContext& context, int slice) {
  cocclPipelineStageContext stageContext = context.stageContext;
  stageContext.rawSliceCount = context.slices[slice].elementCount;
  stageContext.rawSliceBytes = context.slices[slice].bytes;
  return stageContext;
}

cocclPipelineEdge inputEdge(const cocclPipelineContext& context,
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

bool pipelineUsesFramedCompressor(const cocclPipelineSpec* spec) {
  for (int stage = 0; stage < spec->stageCount; ++stage) {
    if (spec->stages[stage].kind == cocclPipelineStageSendRecv) {
      return true;
    }
    if (spec->stages[stage].compressor != nullptr &&
        cocclCompressorSupports(
            spec->stages[stage].compressor,
            cocclCompressorCapabilityFramed)) {
      return true;
    }
  }
  return false;
}

struct cocclPipelineCommunicationComm {
  ncclComm_t comm;
  cocclBufferRegistrationKind registration;
};

int collectCommunicationComms(
    const cocclPipelineSpec* spec, bool framed,
    cocclPipelineCommunicationComm* comms) {
  int count = 0;
  for (int stage = 0; stage < spec->stageCount; ++stage) {
    const cocclPipelineStage& pipelineStage = spec->stages[stage];
    ncclComm_t comm = pipelineStage.comm;
    if (comm == nullptr) continue;

    const bool symmetric = !framed && comm->nNodes == 1 &&
        (pipelineStage.kind == cocclPipelineStageAllGather ||
         pipelineStage.kind == cocclPipelineStageReduceScatter);
    const cocclBufferRegistrationKind registration = symmetric
        ? cocclBufferRegistrationKind::Symmetric
        : cocclBufferRegistrationKind::Ordinary;

    int existing = 0;
    while (existing < count && comms[existing].comm != comm) ++existing;
    if (existing == count) {
      comms[count++] = {comm, registration};
    } else if (symmetric) {
      comms[existing].registration = registration;
    }
  }
  return count;
}

ncclResult_t releaseExecution(cocclPipelineExecution* execution,
                              cudaStream_t stream) {
  const ncclResult_t raw =
      cocclReleaseBuffer(&execution->rawWorkspace, stream);
  const ncclResult_t core =
      cocclReleaseBuffer(&execution->coreWorkspace, stream);
  return raw == ncclSuccess ? core : raw;
}

ncclResult_t prepareExecution(
    const cocclPipelineSpec* spec, int requestedDepth,
    size_t targetSliceBytes, int maxDepth,
    cocclPipelineExecution* execution) {
  *execution = {};
  if (targetSliceBytes == 0) {
    NCCLCHECK(cocclPreparePipeline(
        spec, requestedDepth, &execution->context));
  } else {
    NCCLCHECK(cocclPreparePipelineForSlice(
        spec, targetSliceBytes, maxDepth, &execution->context));
  }

  cocclPipelineCommunicationComm
      communicationComms[kCocclPipelineExplicitStages] = {};
  const bool framed = pipelineUsesFramedCompressor(spec);
  const int communicationCommCount = collectCommunicationComms(
      spec, framed, communicationComms);
  ncclResult_t result = cocclGetBufferForComm(
      spec->ownerComm, communicationComms[0].comm,
      execution->context.plan.registeredBytes,
      communicationComms[0].registration, spec->stream,
      &execution->coreWorkspace);
  if (result != ncclSuccess) return result;
  for (int i = 1; i < communicationCommCount; ++i) {
    result = cocclRegisterBufferForComm(
        &execution->coreWorkspace, communicationComms[i].comm,
        communicationComms[i].registration);
    if (result != ncclSuccess) {
      (void)releaseExecution(execution, spec->stream);
      return result;
    }
  }

  if (execution->context.plan.rawBytes != 0) {
    result = cocclGetUnregisteredBuffer(
        spec->ownerComm, execution->context.plan.rawBytes,
        spec->stream, &execution->rawWorkspace);
    if (result != ncclSuccess) {
      (void)releaseExecution(execution, spec->stream);
      return result;
    }
  }
  execution->workspace = {
      execution->coreWorkspace.ptr, execution->rawWorkspace.ptr};

  if (execution->context.depth > 1 || framed) {
    result = findOrCreateResources(
        spec->ownerComm, &execution->resources);
    if (result != ncclSuccess) {
      (void)releaseExecution(execution, spec->stream);
      return result;
    }
    execution->context.stageContext.frameResources =
        &execution->resources->frameResources;
  }
  return ncclSuccess;
}

cocclPipelineStageOutput stageOutput(const cocclPipelineContext& context,
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

ncclResult_t runSerial(const cocclPipelineContext& context,
                       const cocclPipelineWorkspace& workspace) {
  cocclPipelineEdge edge = inputEdge(context, 0);
  const cocclPipelineStageContext stageContext =
      stageContextForSlice(context, 0);
  if (context.plan.inputStagingTemp >= 0) {
    const cocclPipelineStage pack = cocclPipelinePack();
    const cocclPipelineStageOutput output = {
        tempPtr(context, workspace, 0, context.plan.inputStagingTemp),
        edge.bytes, nullptr, 0};
    NCCLCHECK(cocclExecutePipelineStage(
        &stageContext, &pack, &edge, &output,
        context.spec->stream));
  }
  for (int stage = 0; stage < context.spec->stageCount; ++stage) {
    const cocclPipelineStageOutput output =
        stageOutput(context, workspace, stage, 0);
    NCCLCHECK(cocclExecutePipelineStage(
        &stageContext, context.spec->stages + stage, &edge,
        &output, context.spec->stream));
  }
  if (context.plan.outputStagingTemp >= 0) {
    const cocclPipelineStage unpack = cocclPipelineUnpack();
    const cocclPipelineStageOutput output = {
        static_cast<char*>(context.spec->output) +
            context.slices[0].byteOffset,
        edge.bytes, nullptr, 0};
    NCCLCHECK(cocclExecutePipelineStage(
        &stageContext, &unpack, &edge, &output,
        context.spec->stream));
  }
  return ncclSuccess;
}

ncclResult_t waitForPhase(cocclPipelineResources* resources, int phase,
                          int previousPhase, int slice) {
  CUDACHECK(cudaStreamWaitEvent(resources->streams[phase],
                                resources->events[previousPhase][slice], 0));
  return ncclSuccess;
}

ncclResult_t recordPhase(cocclPipelineResources* resources, int phase,
                         int slice) {
  CUDACHECK(cudaEventRecord(resources->events[phase][slice],
                            resources->streams[phase]));
  return ncclSuccess;
}

ncclResult_t runFramedOverlap(
    const cocclPipelineContext& context,
    const cocclPipelineWorkspace& workspace,
    cocclPipelineResources* resources) {
  const int finalStagePhase =
      cocclPipelinePhaseFirstStage + context.spec->stageCount - 1;
  const bool stagesInput = context.plan.inputStagingTemp >= 0;
  CUDACHECK(cudaEventRecord(resources->inputReady, context.spec->stream));
  CUDACHECK(cudaStreamWaitEvent(
      resources->streams[stagesInput ? cocclPipelinePhasePack
                                    : cocclPipelinePhaseFirstStage],
      resources->inputReady, 0));
  const bool reusesInputRaw =
      context.plan.workspaceKind == cocclPipelineWorkspaceSplit &&
      context.plan.inputStagingTemp >= 0;
  const bool reusesOutputRaw =
      context.plan.workspaceKind == cocclPipelineWorkspaceSplit &&
      context.plan.outputStagingTemp >= 0;

  for (int slice = 0; slice < context.depth; ++slice) {
    const cocclPipelineStageContext stageContext =
        stageContextForSlice(context, slice);
    const int rawSlot = slice % kCocclPipelineRawRingSlots;
    cocclPipelineEdge& edge = resources->edges[slice];
    edge = inputEdge(context, slice);
    if (stagesInput) {
      if (reusesInputRaw && slice >= kCocclPipelineRawRingSlots) {
        CUDACHECK(cudaStreamWaitEvent(
            resources->streams[cocclPipelinePhasePack],
            resources->inputRawConsumed[rawSlot], 0));
      }
      const cocclPipelineStage pack = cocclPipelinePack();
      const cocclPipelineStageOutput packOutput = {
          tempPtr(context, workspace, slice,
                  context.plan.inputStagingTemp),
          edge.bytes, nullptr, 0};
      NCCLCHECK(cocclExecutePipelineStage(
          &stageContext, &pack, &edge, &packOutput,
          resources->streams[cocclPipelinePhasePack]));
      NCCLCHECK(recordPhase(
          resources, cocclPipelinePhasePack, slice));
    }

    const int phase = cocclPipelinePhaseFirstStage;
    if (stagesInput) {
      NCCLCHECK(waitForPhase(
          resources, phase, cocclPipelinePhasePack, slice));
    }
    const cocclPipelineStageOutput output =
        stageOutput(context, workspace, 0, slice);
    NCCLCHECK(cocclExecutePipelineStage(
        &stageContext, context.spec->stages, &edge, &output,
        resources->streams[phase]));
    if (reusesInputRaw) {
      CUDACHECK(cudaEventRecord(
          resources->inputRawConsumed[rawSlot],
          resources->streams[phase]));
    }
    NCCLCHECK(recordPhase(resources, phase, slice));
  }

  for (int stage = 1; stage < context.spec->stageCount; ++stage) {
    const int phase = cocclPipelinePhaseFirstStage + stage;
    const int previousPhase = phase - 1;
    const bool variable = cocclPipelineStageUsesFrameExchange(
        context.spec->stages[stage], resources->edges[0]);
    const bool finalStage = stage + 1 == context.spec->stageCount;

    if (variable) {
      for (int slice = 0; slice < context.depth; ++slice) {
        const cocclPipelineStageContext stageContext =
            stageContextForSlice(context, slice);
        NCCLCHECK(waitForPhase(
            resources, phase, previousPhase, slice));
        resources->stageOutputs[slice] =
            stageOutput(context, workspace, stage, slice);
        NCCLCHECK(cocclPreparePipelineFrameExchange(
            &stageContext, context.spec->stages + stage,
            resources->edges + slice, resources->stageOutputs + slice,
            resources->streams[phase]));
      }
      for (int slice = 0; slice < context.depth; ++slice) {
        const cocclPipelineStageContext stageContext =
            stageContextForSlice(context, slice);
        NCCLCHECK(cocclCommitPipelineFrameExchange(
            &stageContext, context.spec->stages + stage,
            resources->edges + slice, resources->stageOutputs + slice,
            resources->streams[phase]));
        NCCLCHECK(recordPhase(resources, phase, slice));
      }
      continue;
    }

    for (int slice = 0; slice < context.depth; ++slice) {
      const cocclPipelineStageContext stageContext =
          stageContextForSlice(context, slice);
      const int rawSlot = slice % kCocclPipelineRawRingSlots;
      NCCLCHECK(waitForPhase(
          resources, phase, previousPhase, slice));
      if (reusesOutputRaw && finalStage &&
          slice >= kCocclPipelineRawRingSlots) {
        CUDACHECK(cudaStreamWaitEvent(
            resources->streams[phase],
            resources->outputRawConsumed[rawSlot], 0));
      }
      cocclPipelineEdge& edge = resources->edges[slice];
      const cocclPipelineStageOutput output =
          stageOutput(context, workspace, stage, slice);
      NCCLCHECK(cocclExecutePipelineStage(
          &stageContext, context.spec->stages + stage,
          &edge, &output, resources->streams[phase]));
      NCCLCHECK(recordPhase(resources, phase, slice));

      if (finalStage && context.plan.outputStagingTemp >= 0) {
        NCCLCHECK(waitForPhase(
            resources, cocclPipelinePhaseUnpack, phase, slice));
        const cocclPipelineStage unpack = cocclPipelineUnpack();
        const cocclPipelineStageOutput unpackOutput = {
            static_cast<char*>(context.spec->output) +
                context.slices[slice].byteOffset,
            edge.bytes, nullptr, 0};
        NCCLCHECK(cocclExecutePipelineStage(
            &stageContext, &unpack, &edge, &unpackOutput,
            resources->streams[cocclPipelinePhaseUnpack]));
        if (reusesOutputRaw) {
          CUDACHECK(cudaEventRecord(
              resources->outputRawConsumed[rawSlot],
              resources->streams[cocclPipelinePhaseUnpack]));
        }
        if (slice == context.depth - 1) {
          NCCLCHECK(recordPhase(
              resources, cocclPipelinePhaseUnpack, slice));
        }
      }
    }
  }

  const int completionPhase = context.plan.outputStagingTemp >= 0
      ? cocclPipelinePhaseUnpack : finalStagePhase;
  CUDACHECK(cudaStreamWaitEvent(
      context.spec->stream,
      resources->events[completionPhase][context.depth - 1], 0));
  return ncclSuccess;
}

ncclResult_t runOverlap(const cocclPipelineContext& context,
                        const cocclPipelineWorkspace& workspace,
                        cocclPipelineResources* resources) {
  const int finalStagePhase =
      cocclPipelinePhaseFirstStage + context.spec->stageCount - 1;
  const bool stagesInput = context.plan.inputStagingTemp >= 0;
  CUDACHECK(cudaEventRecord(resources->inputReady, context.spec->stream));
  CUDACHECK(cudaStreamWaitEvent(
      resources->streams[stagesInput ? cocclPipelinePhasePack
                                    : cocclPipelinePhaseFirstStage],
      resources->inputReady, 0));
  const bool reusesInputRaw =
      context.plan.workspaceKind == cocclPipelineWorkspaceSplit &&
      context.plan.inputStagingTemp >= 0;
  const bool reusesOutputRaw =
      context.plan.workspaceKind == cocclPipelineWorkspaceSplit &&
      context.plan.outputStagingTemp >= 0;

  for (int slice = 0; slice < context.depth; ++slice) {
    const cocclPipelineStageContext stageContext =
        stageContextForSlice(context, slice);
    const int rawSlot = slice % kCocclPipelineRawRingSlots;
    cocclPipelineEdge edge = inputEdge(context, slice);

    if (stagesInput) {
      if (reusesInputRaw && slice >= kCocclPipelineRawRingSlots) {
        CUDACHECK(cudaStreamWaitEvent(
            resources->streams[cocclPipelinePhasePack],
            resources->inputRawConsumed[rawSlot], 0));
      }
      const cocclPipelineStage pack = cocclPipelinePack();
      const cocclPipelineStageOutput packOutput = {
          tempPtr(context, workspace, slice,
                  context.plan.inputStagingTemp),
          edge.bytes, nullptr, 0};
      NCCLCHECK(cocclExecutePipelineStage(
          &stageContext, &pack, &edge, &packOutput,
          resources->streams[cocclPipelinePhasePack]));
      NCCLCHECK(recordPhase(resources, cocclPipelinePhasePack, slice));
    }

    for (int stage = 0; stage < context.spec->stageCount; ++stage) {
      const int phase = cocclPipelinePhaseFirstStage + stage;
      if (stage != 0 || stagesInput) {
        NCCLCHECK(waitForPhase(resources, phase, phase - 1, slice));
      }
      const bool finalStage = stage + 1 == context.spec->stageCount;
      if (reusesOutputRaw && finalStage &&
          slice >= kCocclPipelineRawRingSlots) {
        CUDACHECK(cudaStreamWaitEvent(
            resources->streams[phase],
            resources->outputRawConsumed[rawSlot], 0));
      }
      const cocclPipelineStageOutput output =
          stageOutput(context, workspace, stage, slice);
      NCCLCHECK(cocclExecutePipelineStage(
          &stageContext, context.spec->stages + stage, &edge,
          &output, resources->streams[phase]));
      if (reusesInputRaw && stage == 0) {
        CUDACHECK(cudaEventRecord(
            resources->inputRawConsumed[rawSlot],
            resources->streams[phase]));
      }
      NCCLCHECK(recordPhase(resources, phase, slice));
    }

    if (context.plan.outputStagingTemp >= 0) {
      NCCLCHECK(waitForPhase(resources, cocclPipelinePhaseUnpack,
                              finalStagePhase, slice));
      const cocclPipelineStage unpack = cocclPipelineUnpack();
      const cocclPipelineStageOutput unpackOutput = {
          static_cast<char*>(context.spec->output) +
              context.slices[slice].byteOffset,
          edge.bytes, nullptr, 0};
      NCCLCHECK(cocclExecutePipelineStage(
          &stageContext, &unpack, &edge, &unpackOutput,
          resources->streams[cocclPipelinePhaseUnpack]));
      if (reusesOutputRaw) {
        CUDACHECK(cudaEventRecord(
            resources->outputRawConsumed[rawSlot],
            resources->streams[cocclPipelinePhaseUnpack]));
      }
      if (slice == context.depth - 1) {
        NCCLCHECK(recordPhase(resources, cocclPipelinePhaseUnpack, slice));
      }
    }
  }

  const int completionPhase = context.plan.outputStagingTemp >= 0
      ? cocclPipelinePhaseUnpack : finalStagePhase;
  CUDACHECK(cudaStreamWaitEvent(
      context.spec->stream,
      resources->events[completionPhase][context.depth - 1], 0));
  return ncclSuccess;
}

int sendRecvStage(const cocclPipelineSpec* spec) {
  for (int stage = 0; stage < spec->stageCount; ++stage) {
    if (spec->stages[stage].kind == cocclPipelineStageSendRecv) {
      return stage;
    }
  }
  return -1;
}

ncclResult_t exchangeBatchPlans(
    const cocclPipelineSpec* specs, size_t count,
    std::vector<cocclPipelineBatchPlan>* plans) {
  ncclResult_t result = ncclSuccess;
  std::vector<cocclBufferHandle> buffers(count);
  std::vector<cocclFrameExchange> exchanges(count);
  const cocclPipelineConfig& config = cocclGetConfig().pipeline;
  for (size_t i = 0; i < count; ++i) {
    const int stageIndex = sendRecvStage(specs + i);
    const cocclPipelineStage& stage = specs[i].stages[stageIndex];
    NCCLCHECKGOTO(cocclGetBuffer(
        specs[i].ownerComm, sizeof(cocclPipelineBatchPlan),
        specs[i].stream, &buffers[i]), result, cleanup);
    if (stage.direction == cocclPipelineSend) {
      if (config.autoDepth) {
        const cocclPipelineTuningDecision tuning =
            cocclAutotunePipelineLayout(specs + i);
        (*plans)[i] = {(uint64_t)tuning.targetSliceBytes,
                       (uint32_t)tuning.maxDepth, 0};
      } else {
        (*plans)[i] = {0, (uint32_t)std::max(1, config.depth), 0};
      }
      CUDACHECKGOTO(cudaMemcpyAsync(
          buffers[i].ptr, plans->data() + i, sizeof((*plans)[i]),
          cudaMemcpyHostToDevice, specs[i].stream), result, cleanup);
    }
    exchanges[i] = {
        stage.peer,
        stage.direction == cocclPipelineSend ? buffers[i].ptr : nullptr,
        stage.direction == cocclPipelineRecv ? buffers[i].ptr : nullptr,
        stage.direction == cocclPipelineSend ? sizeof((*plans)[i]) : 0,
        stage.direction == cocclPipelineRecv ? sizeof((*plans)[i]) : 0,
        sizeof((*plans)[i]), stage.comm, specs[i].stream};
  }
  NCCLCHECKGOTO(cocclCommitFrameExchange(
      exchanges.data(), exchanges.size(), nullptr, nullptr), result,
      cleanup);
  for (size_t i = 0; i < count; ++i) {
    const cocclPipelineStage& stage =
        specs[i].stages[sendRecvStage(specs + i)];
    if (stage.direction == cocclPipelineRecv) {
      CUDACHECKGOTO(cudaMemcpyAsync(
          plans->data() + i, buffers[i].ptr, sizeof((*plans)[i]),
          cudaMemcpyDeviceToHost, specs[i].stream), result, cleanup);
    }
  }
  for (size_t i = 0; i < count; ++i) {
    const cocclPipelineStage& stage =
        specs[i].stages[sendRecvStage(specs + i)];
    if (stage.direction == cocclPipelineRecv) {
      CUDACHECKGOTO(cudaStreamSynchronize(specs[i].stream), result,
                    cleanup);
      if ((*plans)[i].maxDepth == 0 ||
          (*plans)[i].maxDepth > kCocclPipelineMaxDepth) {
        result = ncclInvalidUsage;
        goto cleanup;
      }
    }
  }

cleanup:
  for (size_t i = 0; i < count; ++i) {
    if (buffers[i].ptr == nullptr) continue;
    const ncclResult_t release =
        cocclReleaseBuffer(&buffers[i], specs[i].stream);
    if (result == ncclSuccess) result = release;
  }
  return result;
}

struct cocclPipelineBatchFrame {
  cocclPipelineBatchState* state;
  int slice;
  int phase;
  cocclCompressorFrameMetadata* deviceMetadata;
  size_t slotBytes;
};

void applyReceivedFrame(
    const cocclPipelineStage& stage,
    const cocclCompressorFrameMetadata& metadata,
    cocclPipelineEdge* edge, const cocclPipelineStageOutput& output) {
  edge->ptr = output.ptr;
  edge->compressor = stage.compressor;
  if (cocclCompressorSupports(
          stage.compressor, cocclCompressorCapabilityFramed)) {
    edge->bytes = output.capacityBytes;
    edge->totalElements = output.capacityBytes;
    edge->datatype = ncclInt8;
    edge->frameMetadata = output.frameMetadata;
    edge->frameStrideBytes = output.frameStrideBytes;
  } else {
    edge->bytes = (size_t)metadata.payloadBytes;
    edge->totalElements = edge->bytes;
    edge->datatype = metadata.encoding == cocclCompressorFrameRaw
        ? COCCL_COMPRESSOR_RAW_PASSTHROUGH : ncclInt8;
    edge->frameMetadata = nullptr;
    edge->frameStrideBytes = 0;
  }
}

ncclResult_t runPipelineBatchWave(
    std::vector<cocclPipelineBatchState>* states, int slice) {
  std::vector<cocclPipelineBatchFrame> frames;
  std::vector<cocclFrameExchange> metadataExchanges;
  for (cocclPipelineBatchState& state : *states) {
    cocclPipelineContext& context = state.execution.context;
    if (slice >= context.depth) continue;
    cocclPipelineResources* resources = state.execution.resources;
    const int stage = state.communicationStage;
    const int phase = cocclPipelinePhaseFirstStage + stage;
    if (stage != 0) {
      NCCLCHECK(waitForPhase(resources, phase, phase - 1, slice));
    }
    state.outputs[slice] = stageOutput(
        context, state.execution.workspace, stage, slice);
    const cocclPipelineStage& exchangeStage = context.spec->stages[stage];
    cocclCompressorFrameMetadata* metadata =
        exchangeStage.direction == cocclPipelineSend
        ? state.edges[slice].frameMetadata
        : state.outputs[slice].frameMetadata;
    frames.push_back({&state, slice, phase, metadata,
                      exchangeStage.direction == cocclPipelineSend
                      ? state.edges[slice].frameStrideBytes
                      : state.outputs[slice].frameStrideBytes});
    metadataExchanges.push_back({
        exchangeStage.peer,
        exchangeStage.direction == cocclPipelineSend
            ? metadata : nullptr,
        exchangeStage.direction == cocclPipelineRecv
            ? metadata : nullptr,
        exchangeStage.direction == cocclPipelineSend
            ? sizeof(*metadata) : 0,
        exchangeStage.direction == cocclPipelineRecv
            ? sizeof(*metadata) : 0,
        sizeof(*metadata), exchangeStage.comm,
        resources->streams[phase]});
  }

  NCCLCHECK(cocclCommitFrameExchange(
      metadataExchanges.data(), metadataExchanges.size(), nullptr, nullptr));
  std::vector<cocclCompressorFrameMetadata> hostMetadata(frames.size());
  for (size_t i = 0; i < frames.size(); ++i) {
    cocclPipelineBatchFrame& frame = frames[i];
    cocclPipelineResources* resources =
        frame.state->execution.resources;
    CUDACHECK(cudaMemcpyAsync(
        hostMetadata.data() + i, frame.deviceMetadata,
        sizeof(cocclCompressorFrameMetadata), cudaMemcpyDeviceToHost,
        resources->streams[frame.phase]));
  }
  std::vector<cudaStream_t> synchronizedStreams;
  for (const cocclPipelineBatchFrame& frame : frames) {
    const cudaStream_t stream =
        frame.state->execution.resources->streams[frame.phase];
    bool seen = false;
    for (cudaStream_t existing : synchronizedStreams) {
      if (existing == stream) seen = true;
    }
    if (!seen) {
      CUDACHECK(cudaStreamSynchronize(stream));
      synchronizedStreams.push_back(stream);
    }
  }

  std::vector<cocclFrameExchange> payloadExchanges(frames.size());
  for (size_t i = 0; i < frames.size(); ++i) {
    cocclPipelineBatchFrame& frame = frames[i];
    cocclPipelineBatchState& state = *frame.state;
    cocclPipelineContext& context = state.execution.context;
    const cocclPipelineStage& stage =
        context.spec->stages[state.communicationStage];
    if (!cocclFrameMetadataValid(hostMetadata[i], frame.slotBytes)) {
      return ncclInvalidUsage;
    }
    payloadExchanges[i] = {
        stage.peer,
        stage.direction == cocclPipelineSend
            ? state.edges[slice].ptr : nullptr,
        stage.direction == cocclPipelineRecv
            ? state.outputs[slice].ptr : nullptr,
        stage.direction == cocclPipelineSend
            ? (size_t)hostMetadata[i].payloadBytes : 0,
        stage.direction == cocclPipelineRecv
            ? (size_t)hostMetadata[i].payloadBytes : 0,
        frame.slotBytes, stage.comm,
        state.execution.resources->streams[frame.phase]};
  }
  NCCLCHECK(cocclCommitFrameExchange(
      payloadExchanges.data(), payloadExchanges.size(), nullptr, nullptr));
  for (size_t i = 0; i < frames.size(); ++i) {
    cocclPipelineBatchFrame& frame = frames[i];
    cocclPipelineBatchState& state = *frame.state;
    cocclPipelineContext& context = state.execution.context;
    const cocclPipelineStage& stage =
        context.spec->stages[state.communicationStage];
    if (stage.direction == cocclPipelineRecv) {
      applyReceivedFrame(stage, hostMetadata[i], state.edges + slice,
                         state.outputs[slice]);
    }
    NCCLCHECK(recordPhase(
        state.execution.resources, frame.phase, slice));
  }

  for (cocclPipelineBatchState& state : *states) {
    cocclPipelineContext& context = state.execution.context;
    if (slice >= context.depth) continue;
    cocclPipelineResources* resources = state.execution.resources;
    for (int stage = state.communicationStage + 1;
         stage < context.spec->stageCount; ++stage) {
      const int phase = cocclPipelinePhaseFirstStage + stage;
      NCCLCHECK(waitForPhase(resources, phase, phase - 1, slice));
      const cocclPipelineStageContext stageContext =
          stageContextForSlice(context, slice);
      const cocclPipelineStageOutput output = stageOutput(
          context, state.execution.workspace, stage, slice);
      NCCLCHECK(cocclExecutePipelineStage(
          &stageContext, context.spec->stages + stage,
          state.edges + slice, &output, resources->streams[phase]));
      NCCLCHECK(recordPhase(resources, phase, slice));
    }
  }
  return ncclSuccess;
}

ncclResult_t runPipelineBatch(
    std::vector<cocclPipelineBatchState>* states) {
  for (cocclPipelineBatchState& state : *states) {
    cocclPipelineContext& context = state.execution.context;
    cocclPipelineResources* resources = state.execution.resources;
    CUDACHECK(cudaEventRecord(resources->inputReady,
                               context.spec->stream));
    CUDACHECK(cudaStreamWaitEvent(
        resources->streams[cocclPipelinePhaseFirstStage],
        resources->inputReady, 0));
    for (int slice = 0; slice < context.depth; ++slice) {
      state.edges[slice] = inputEdge(context, slice);
    }
  }

  int maxDepth = 0;
  for (const cocclPipelineBatchState& state : *states) {
    maxDepth = std::max(maxDepth, state.execution.context.depth);
  }

  // Queue the same wave for every call before later waves. Calls sharing one
  // communicator then reach the grouped metadata exchange together.
  for (int slice = 0; slice < maxDepth; ++slice) {
    for (cocclPipelineBatchState& state : *states) {
      cocclPipelineContext& context = state.execution.context;
      if (slice >= context.depth) continue;
      cocclPipelineResources* resources = state.execution.resources;
      for (int stage = 0; stage < state.communicationStage; ++stage) {
        const int phase = cocclPipelinePhaseFirstStage + stage;
        if (stage != 0) {
          NCCLCHECK(waitForPhase(
              resources, phase, phase - 1, slice));
        }
        const cocclPipelineStageContext stageContext =
            stageContextForSlice(context, slice);
        const cocclPipelineStageOutput output = stageOutput(
            context, state.execution.workspace, stage, slice);
        NCCLCHECK(cocclExecutePipelineStage(
            &stageContext, context.spec->stages + stage,
            state.edges + slice, &output, resources->streams[phase]));
        NCCLCHECK(recordPhase(resources, phase, slice));
      }
    }
  }

  for (int slice = 0; slice < maxDepth; ++slice) {
    NCCLCHECK(runPipelineBatchWave(states, slice));
  }

  for (cocclPipelineBatchState& state : *states) {
    const cocclPipelineContext& context = state.execution.context;
    const int finalPhase = cocclPipelinePhaseFirstStage +
        context.spec->stageCount - 1;
    CUDACHECK(cudaStreamWaitEvent(
        context.spec->stream,
        state.execution.resources->events[finalPhase][context.depth - 1],
        0));
  }
  return ncclSuccess;
}

}  // namespace

static ncclResult_t cocclRunPipelineWithLayout(
    const cocclPipelineSpec* spec, int requestedDepth,
    size_t targetSliceBytes, int maxDepth) {
  CUDACHECK(cudaSetDevice(spec->ownerComm->cudaDev));
  cocclPipelineExecution execution = {};
  NCCLCHECK(prepareExecution(
      spec, requestedDepth, targetSliceBytes, maxDepth, &execution));
  cocclPipelineContext& context = execution.context;
  const bool framed = pipelineUsesFramedCompressor(spec);
  ncclResult_t result = ncclSuccess;
  if (context.depth == 1) {
    result = runSerial(context, execution.workspace);
  } else if (framed) {
    result = runFramedOverlap(
        context, execution.workspace, execution.resources);
  } else {
    result = runOverlap(context, execution.workspace, execution.resources);
  }

  if (result != ncclSuccess) (void)cudaDeviceSynchronize();
  const ncclResult_t release = releaseExecution(&execution, spec->stream);
  return result == ncclSuccess ? release : result;
}

ncclResult_t cocclRunPipeline(const cocclPipelineSpec* spec) {
  const bool singleNode = spec->ownerComm->localRanks ==
      spec->ownerComm->nRanks;
  // Raw boundary copies cost more than slice overlap recovers on one node.
  const cocclPipelineConfig& config = cocclGetConfig().pipeline;
  const bool serialFixedLayout =
      singleNode && !pipelineUsesFramedCompressor(spec);
  if (serialFixedLayout || !config.autoDepth) {
    return cocclRunPipelineWithLayout(
        spec, serialFixedLayout ? 1 : config.depth, 0,
        kCocclPipelineMaxDepth);
  }
  const cocclPipelineTuningDecision tuning =
      cocclAutotunePipelineLayout(spec);
  return cocclRunPipelineWithLayout(
      spec, 1, tuning.targetSliceBytes, tuning.maxDepth);
}

ncclResult_t cocclRunPipelineBatch(
    const cocclPipelineSpec* specs, size_t count) {
  if (specs == nullptr || count == 0) return ncclInvalidArgument;
  CUDACHECK(cudaSetDevice(specs[0].ownerComm->cudaDev));
  std::vector<cocclPipelineBatchState> states(count);
  std::vector<cocclPipelineBatchPlan> plans(count);
  ncclResult_t result = ncclSuccess;
  NCCLCHECK(exchangeBatchPlans(specs, count, &plans));
  size_t prepared = 0;
  for (; prepared < count; ++prepared) {
    cocclPipelineBatchState& state = states[prepared];
    state.communicationStage = sendRecvStage(specs + prepared);
    if (state.communicationStage < 0) {
      result = ncclInvalidArgument;
      break;
    }
    if (plans[prepared].targetSliceBytes != 0) {
      result = prepareExecution(
          specs + prepared, 1,
          (size_t)plans[prepared].targetSliceBytes,
          (int)plans[prepared].maxDepth, &state.execution);
    } else {
      result = prepareExecution(
          specs + prepared, (int)plans[prepared].maxDepth, 0,
          kCocclPipelineMaxDepth, &state.execution);
    }
    if (result != ncclSuccess) break;
  }
  if (result == ncclSuccess) result = runPipelineBatch(&states);
  if (result != ncclSuccess) (void)cudaDeviceSynchronize();
  for (size_t i = 0; i < prepared; ++i) {
    const ncclResult_t release = releaseExecution(
        &states[i].execution, specs[i].stream);
    if (result == ncclSuccess) result = release;
  }
  return result;
}

ncclResult_t cocclRunPipelineSerial(const cocclPipelineSpec* spec) {
  return cocclRunPipelineWithLayout(
      spec, 1, 0, kCocclPipelineMaxDepth);
}

ncclResult_t cocclPipelineCommDestroy(ncclComm_t comm) {
  cocclAutotunePipelineCommDestroy(comm);
  auto found = resourcesByComm.find(comm);
  if (found != resourcesByComm.end()) {
    cocclPipelineResources* resources = found->second;
    resourcesByComm.erase(found);
    destroyResources(resources);
  }
  return ncclSuccess;
}
