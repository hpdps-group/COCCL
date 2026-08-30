#include "core/pipeline/coccl_pipeline.h"

#include "checks.h"
#include "config/collconfig.h"
#include "core/memory/coccl_buffer_management.h"
#include "core/config/coccl_config.h"
#include "core/pipeline/coccl_pipeline_internal.h"
#include "core/pipeline/coccl_pipeline_depth.h"
#include "comm.h"
#include "core/compression/compress.h"
#include "rma/rma.h"

#include <stdlib.h>

#include <map>
#include <new>

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

    const int stagePolicy = cocclPipelineStageCtaPolicy(
        spec->ownerComm, pipelineStage);
    const int effectivePolicy = ncclCollConfigResolveCTAPolicy(
        stagePolicy, comm->config.CTAPolicy,
        ncclGetEnvCtaPolicy() != NCCL_CONFIG_UNDEF_INT);
    const bool zeroCta =
        (effectivePolicy & NCCL_CTA_POLICY_ZERO) != 0;
    const bool framedRma = inputFramed &&
        pipelineStage.kind == cocclPipelineStageAllToAll &&
        comm->config.rmaEagerInit && comm->hostRmaSupport &&
        comm->config.numRmaSig > 0 &&
        (comm->nNodes == 1 || ncclRmaProxyEnabled(comm));
    const bool symmetric = framedRma ||
        (!inputFramed &&
         (pipelineStage.kind == cocclPipelineStageAllGather ||
          (pipelineStage.kind == cocclPipelineStageAllToAll && zeroCta) ||
          pipelineStage.kind == cocclPipelineStageReduceScatter));
    const cocclBufferRegistrationKind registration = framedRma
        ? cocclBufferRegistrationKind::Rma
        : (symmetric ? cocclBufferRegistrationKind::Symmetric
                     : cocclBufferRegistrationKind::Ordinary);

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

}  // namespace

static ncclResult_t cocclRunPipelineWithDepth(
    const cocclPipelineSpec* spec, int requestedDepth) {
  cocclPipelineContext context = {};
  NCCLCHECK(cocclPreparePipeline(spec, requestedDepth, &context));
  CUDACHECK(cudaSetDevice(spec->ownerComm->cudaDev));

  const bool framed = pipelineUsesFramedCompressor(spec);
  cocclPipelineCommunicationComm
      communicationComms[kCocclPipelineExplicitStages] = {};
  const int communicationCommCount =
      collectCommunicationComms(spec, context.plan, communicationComms);
  cocclBufferHandle coreWorkspace = {};
  ncclResult_t result = cocclGetBufferForComm(
      spec->ownerComm, communicationComms[0].comm,
      context.plan.registeredBytes, communicationComms[0].registration,
      spec->stream, &coreWorkspace);
  if (result != ncclSuccess) return result;
  for (int i = 1; i < communicationCommCount; ++i) {
    result = cocclRegisterBufferForComm(
        &coreWorkspace, communicationComms[i].comm,
        communicationComms[i].registration);
    if (result != ncclSuccess) {
      (void)cocclReleaseBuffer(&coreWorkspace, spec->stream);
      return result;
    }
  }
  context.stageContext.registeredBase = coreWorkspace.ptr;
  for (int i = 0; framed && i < communicationCommCount; ++i) {
    cocclBufferRmaInfo info;
    if (!cocclGetBufferRmaInfo(
            coreWorkspace, communicationComms[i].comm, &info)) {
      continue;
    }
    cocclPipelineRmaWindow& window =
        context.stageContext.rmaWindows[
            context.stageContext.rmaWindowCount++];
    window = {communicationComms[i].comm, info.window,
              info.bufferOffset, info.singleSegment};
  }

  cocclBufferHandle rawWorkspace = {};
  if (context.plan.rawBytes != 0) {
    // Split boundary storage stays unregistered unless a Zero stage directly
    // consumes or produces it, as in inter-only native ReduceScatter.
    int rawRegistration = 0;
    while (rawRegistration < communicationCommCount &&
           (!communicationComms[rawRegistration].usesRawWorkspace ||
            communicationComms[rawRegistration].registration ==
                cocclBufferRegistrationKind::Ordinary)) {
      ++rawRegistration;
    }
    ncclResult_t rawResult = ncclSuccess;
    if (rawRegistration == communicationCommCount) {
      rawResult = cocclGetUnregisteredBuffer(
          spec->ownerComm, context.plan.rawBytes, spec->stream,
          &rawWorkspace);
    } else {
      rawResult = cocclGetBufferForComm(
          spec->ownerComm, communicationComms[rawRegistration].comm,
          context.plan.rawBytes,
          communicationComms[rawRegistration].registration,
          spec->stream, &rawWorkspace);
      for (int i = rawRegistration + 1;
           rawResult == ncclSuccess && i < communicationCommCount; ++i) {
        if (communicationComms[i].usesRawWorkspace &&
            communicationComms[i].registration !=
                cocclBufferRegistrationKind::Ordinary) {
          rawResult = cocclRegisterBufferForComm(
              &rawWorkspace, communicationComms[i].comm,
              communicationComms[i].registration);
        }
      }
    }
    if (rawResult != ncclSuccess) {
      (void)cocclReleaseBuffer(&coreWorkspace, spec->stream);
      return rawResult;
    }
  }
  const cocclPipelineWorkspace workspace = {
      coreWorkspace.ptr, rawWorkspace.ptr};

  result = ncclSuccess;
  if (context.depth == 1 && !framed) {
    result = runSerial(context, workspace);
  } else {
    cocclPipelineResources* resources = nullptr;
    result = findOrCreateResources(spec->ownerComm, &resources);
    if (result == ncclSuccess) {
      context.stageContext.frameResources = &resources->frameResources;
      if (context.depth == 1) {
        result = runSerial(context, workspace);
      } else if (framed) {
        result = runFramedOverlap(context, workspace, resources);
      } else {
        result = runOverlap(context, workspace, resources);
      }
    }
  }

  if (result != ncclSuccess) (void)cudaDeviceSynchronize();
  const ncclResult_t rawRelease =
      cocclReleaseBuffer(&rawWorkspace, spec->stream);
  const ncclResult_t coreRelease =
      cocclReleaseBuffer(&coreWorkspace, spec->stream);
  if (result == ncclSuccess) result = rawRelease;
  return result == ncclSuccess ? coreRelease : result;
}

ncclResult_t cocclRunPipeline(const cocclPipelineSpec* spec) {
  const bool singleNode = spec->ownerComm->localRanks ==
      spec->ownerComm->nRanks;
  // Raw boundary copies cost more than slice overlap recovers on one node.
  const cocclPipelineConfig& config = cocclGetConfig().pipeline;
  const bool serialFixedLayout =
      singleNode && !pipelineUsesFramedCompressor(spec);
  const int depth = serialFixedLayout
      ? 1
      : (config.autoDepth ? cocclAutotunePipelineDepth(spec) : config.depth);
  return cocclRunPipelineWithDepth(spec, depth);
}

ncclResult_t cocclRunPipelineSerial(const cocclPipelineSpec* spec) {
  return cocclRunPipelineWithDepth(spec, 1);
}

ncclResult_t cocclPipelineCommDestroy(ncclComm_t comm) {
  cocclPipelineDepthCommDestroy(comm);
  auto found = resourcesByComm.find(comm);
  if (found != resourcesByComm.end()) {
    cocclPipelineResources* resources = found->second;
    resourcesByComm.erase(found);
    destroyResources(resources);
  }
  return ncclSuccess;
}
