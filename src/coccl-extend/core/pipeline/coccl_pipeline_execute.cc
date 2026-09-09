#include "core/pipeline/coccl_pipeline_execution.h"

#include "core/config/coccl_config.h"
#include "core/tuning/coccl_autotune_pipeline.h"
#include "comm.h"
#include "core/compression/compress.h"
#include "debug.h"

#include <stdlib.h>

#include <map>
#include <new>

using namespace cocclPipelineInternal;

namespace {

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
  free(resources->frameResources.waitDescriptors);
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
  // The same communicator may alternate serial and pipelined recipes.
  ncclResult_t result = createResources(&resources);
  if (result == ncclSuccess) {
    resourcesByComm.emplace(comm, resources);
    *output = resources;
  }
  return result;
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
  bool usesRawWorkspace;
};

bool stageOutputIsFramed(
    const cocclPipelineStage& stage, bool inputFramed) {
  switch (stage.kind) {
    case cocclPipelineStageCompress:
    case cocclPipelineStageDecompReduceComp:
      return cocclCompressorSupports(
          stage.compressor, cocclCompressorCapabilityFramed);
    case cocclPipelineStageDecompress:
    case cocclPipelineStageDecompressReduce:
    case cocclPipelineStageReduceScatter:
      return false;
    case cocclPipelineStageSendRecv:
      return stage.direction == cocclPipelineRecv || inputFramed;
    default:
      return inputFramed;
  }
}

int collectCommunicationComms(
    const cocclPipelineSpec* spec,
    const cocclPipelinePlan& plan,
    cocclPipelineCommunicationComm* comms) {
  int count = 0;
  bool edgeFramed = false;
  int inputTemp = plan.inputStagingTemp;
  for (int stage = 0; stage < spec->stageCount; ++stage) {
    const cocclPipelineStage& pipelineStage = spec->stages[stage];
    const bool inputFramed = edgeFramed;
    edgeFramed = stageOutputIsFramed(pipelineStage, edgeFramed);
    const int outputTemp = plan.stageOutputTemp[stage];
    ncclComm_t comm = pipelineStage.comm;
    if (comm == nullptr) {
      inputTemp = outputTemp;
      continue;
    }

    const bool usesRawWorkspace =
        (inputTemp >= 0 &&
         plan.temps[inputTemp].storage == cocclPipelineRawRing) ||
        (outputTemp >= 0 &&
         plan.temps[outputTemp].storage == cocclPipelineRawRing);

    const cocclBufferRegistrationKind registration =
        cocclBackendStageRegistration(spec->ownerComm, pipelineStage,
                                       inputFramed);
    const bool symmetric =
        registration != cocclBufferRegistrationKind::Ordinary;
    int existing = 0;
    while (existing < count && comms[existing].comm != comm) ++existing;
    if (existing == count) {
      comms[count++] = {comm, registration, usesRawWorkspace};
    } else if (symmetric) {
      comms[existing].registration = registration;
    }
    if (existing < count) {
      comms[existing].usesRawWorkspace |= usesRawWorkspace;
    }
    inputTemp = outputTemp;
  }
  return count;
}

}  // namespace

ncclResult_t cocclPipelineInternal::releaseExecution(cocclPipelineExecution* execution,
                              cudaStream_t stream) {
  const ncclResult_t raw =
      cocclReleaseBuffer(&execution->rawWorkspace, stream);
  const ncclResult_t core =
      cocclReleaseBuffer(&execution->coreWorkspace, stream);
  return raw == ncclSuccess ? core : raw;
}

ncclResult_t cocclPipelineInternal::prepareExecution(
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
  const int communicationCommCount = collectCommunicationComms(
      spec, execution->context.plan, communicationComms);
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

  execution->context.stageContext.registeredBase =
      execution->coreWorkspace.ptr;
  const bool framed = pipelineUsesFramedCompressor(spec);
  for (int i = 0; framed && i < communicationCommCount; ++i) {
    cocclBufferRmaInfo info;
    if (!cocclGetBufferRmaInfo(
            execution->coreWorkspace, communicationComms[i].comm,
            &info)) {
      continue;
    }
    cocclPipelineRmaWindow& window =
        execution->context.stageContext.rmaWindows[
            execution->context.stageContext.rmaWindowCount++];
    window = {communicationComms[i].comm, info.window,
              info.bufferOffset, info.singleSegment};
  }

  if (execution->context.plan.rawBytes != 0) {
    int rawRegistration = 0;
    while (rawRegistration < communicationCommCount &&
           (!communicationComms[rawRegistration].usesRawWorkspace ||
            communicationComms[rawRegistration].registration ==
                cocclBufferRegistrationKind::Ordinary)) {
      ++rawRegistration;
    }
    if (rawRegistration == communicationCommCount) {
      result = cocclGetUnregisteredBuffer(
          spec->ownerComm, execution->context.plan.rawBytes,
          spec->stream, &execution->rawWorkspace);
    } else {
      result = cocclGetBufferForComm(
          spec->ownerComm, communicationComms[rawRegistration].comm,
          execution->context.plan.rawBytes,
          communicationComms[rawRegistration].registration, spec->stream,
          &execution->rawWorkspace);
      for (int i = rawRegistration + 1;
           result == ncclSuccess && i < communicationCommCount; ++i) {
        if (communicationComms[i].usesRawWorkspace &&
            communicationComms[i].registration !=
                cocclBufferRegistrationKind::Ordinary) {
          result = cocclRegisterBufferForComm(
              &execution->rawWorkspace, communicationComms[i].comm,
              communicationComms[i].registration);
        }
      }
    }
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

namespace {

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

inline ncclResult_t packSlice(
    const cocclPipelineContext& context,
    const cocclPipelineWorkspace& workspace,
    cocclPipelineResources* resources,
    const cocclPipelineStageContext& stageContext,
    int slice, bool reusesInputRaw, cocclPipelineEdge* edge) {
  const int rawSlot = slice % kCocclPipelineRawRingSlots;
  if (reusesInputRaw && slice >= kCocclPipelineRawRingSlots) {
    CUDACHECK(cudaStreamWaitEvent(
        resources->streams[cocclPipelinePhasePack],
        resources->inputRawConsumed[rawSlot], 0));
  }
  const cocclPipelineStage pack = cocclPipelinePack();
  const cocclPipelineStageOutput packOutput = {
      tempPtr(context, workspace, slice,
              context.plan.inputStagingTemp),
      edge->bytes, nullptr, 0};
  NCCLCHECK(cocclExecutePipelineStage(
      &stageContext, &pack, edge, &packOutput,
      resources->streams[cocclPipelinePhasePack]));
  NCCLCHECK(recordPhase(resources, cocclPipelinePhasePack, slice));
  return ncclSuccess;
}

inline ncclResult_t unpackSlice(
    const cocclPipelineContext& context, cocclPipelineResources* resources,
    const cocclPipelineStageContext& stageContext, int slice,
    int finalStagePhase, bool reusesOutputRaw, cocclPipelineEdge* edge) {
  const int rawSlot = slice % kCocclPipelineRawRingSlots;
  NCCLCHECK(waitForPhase(resources, cocclPipelinePhaseUnpack,
                          finalStagePhase, slice));
  const cocclPipelineStage unpack = cocclPipelineUnpack();
  const cocclPipelineStageOutput unpackOutput = {
      static_cast<char*>(context.spec->output) +
          context.slices[slice].byteOffset,
      edge->bytes, nullptr, 0};
  NCCLCHECK(cocclExecutePipelineStage(
      &stageContext, &unpack, edge, &unpackOutput,
      resources->streams[cocclPipelinePhaseUnpack]));
  if (reusesOutputRaw) {
    CUDACHECK(cudaEventRecord(
        resources->outputRawConsumed[rawSlot],
        resources->streams[cocclPipelinePhaseUnpack]));
  }
  if (slice == context.depth - 1) {
    NCCLCHECK(recordPhase(resources, cocclPipelinePhaseUnpack, slice));
  }
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
    cocclPipelineEdge& edge = resources->slices[slice].edge;
    edge = inputEdge(context, slice);
    if (stagesInput) {
      NCCLCHECK(packSlice(
          context, workspace, resources, stageContext, slice,
          reusesInputRaw, &edge));
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
        context.spec->stages[stage], resources->slices[0].edge);
    const bool finalStage = stage + 1 == context.spec->stageCount;

    if (variable) {
      for (int slice = 0; slice < context.depth; ++slice) {
        const cocclPipelineStageContext stageContext =
            stageContextForSlice(context, slice);
        NCCLCHECK(waitForPhase(
            resources, phase, previousPhase, slice));
        resources->slices[slice].output =
            stageOutput(context, workspace, stage, slice);
        NCCLCHECK(cocclPreparePipelineFrameExchange(
            &stageContext, context.spec->stages + stage,
            &resources->slices[slice].edge, &resources->slices[slice].output,
            resources->streams[phase]));
      }
      for (int slice = 0; slice < context.depth; ++slice) {
        const cocclPipelineStageContext stageContext =
            stageContextForSlice(context, slice);
        NCCLCHECK(cocclCommitPipelineFrameExchange(
            &stageContext, context.spec->stages + stage,
            &resources->slices[slice].edge, &resources->slices[slice].output,
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
      cocclPipelineEdge& edge = resources->slices[slice].edge;
      const cocclPipelineStageOutput output =
          stageOutput(context, workspace, stage, slice);
      NCCLCHECK(cocclExecutePipelineStage(
          &stageContext, context.spec->stages + stage,
          &edge, &output, resources->streams[phase]));
      NCCLCHECK(recordPhase(resources, phase, slice));

      if (finalStage && context.plan.outputStagingTemp >= 0) {
        NCCLCHECK(unpackSlice(
            context, resources, stageContext, slice, phase,
            reusesOutputRaw, &edge));
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
      NCCLCHECK(packSlice(
          context, workspace, resources, stageContext, slice,
          reusesInputRaw, &edge));
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
      NCCLCHECK(unpackSlice(
          context, resources, stageContext, slice, finalStagePhase,
          reusesOutputRaw, &edge));
    }
  }

  const int completionPhase = context.plan.outputStagingTemp >= 0
      ? cocclPipelinePhaseUnpack : finalStagePhase;
  CUDACHECK(cudaStreamWaitEvent(
      context.spec->stream,
      resources->events[completionPhase][context.depth - 1], 0));
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
