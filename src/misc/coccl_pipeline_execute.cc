#include "coccl_pipeline.h"

#include "checks.h"
#include "coccl_buffer_management.h"
#include "coccl_config.h"
#include "coccl_pipeline_internal.h"
#include "comm.h"

#include <map>
#include <new>

namespace {

enum cocclPipelinePhase {
  cocclPipelinePhasePack = 0,
  cocclPipelinePhaseCompress = 1,
  cocclPipelinePhaseAllToAll = 2,
  cocclPipelinePhaseDecompress = 3,
  cocclPipelinePhaseUnpack = 4,
};

struct cocclPipelineResources {
  int depth;
  cudaStream_t streams[kCocclPipelinePhysicalStages];
  cudaEvent_t events[kCocclPipelinePhysicalStages]
                    [kCocclPipelineMaxDepth];
  cudaEvent_t inputReady;
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
  delete resources;
}

ncclResult_t createResources(int depth, cocclPipelineResources** output) {
  cocclPipelineResources* resources =
      new (std::nothrow) cocclPipelineResources();
  if (resources == nullptr) return ncclSystemError;
  *resources = {};
  resources->depth = depth;

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
    for (int slice = 0; slice < depth; ++slice) {
      if (phase == cocclPipelinePhaseUnpack && slice != depth - 1) continue;
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
  *output = resources;
  return ncclSuccess;
}

ncclResult_t findOrCreateResources(ncclComm_t comm, int depth,
                                   cocclPipelineResources** output) {
  auto found = resourcesByComm.find(comm);
  if (found != resourcesByComm.end()) {
    *output = found->second;
    return ncclSuccess;
  }

  cocclPipelineResources* resources = nullptr;
  ncclResult_t result = createResources(depth, &resources);
  if (result == ncclSuccess) {
    resourcesByComm.emplace(comm, resources);
    *output = resources;
  }
  return result;
}

void* tempPtr(const cocclPipelineContext& context, void* workspace,
              int slice, int tempIndex) {
  return static_cast<char*>(workspace) +
      (size_t)slice * context.plan.sliceWorkspaceBytes +
      context.plan.temps[tempIndex].offset;
}

cocclPipelineEdge inputEdge(const cocclPipelineContext& context,
                            int slice) {
  return {
      static_cast<char*>(const_cast<void*>(context.spec->input)) +
          (size_t)slice * context.stageContext.rawSliceBytes,
      context.stageContext.rawSliceBytes * context.spec->inputChunks,
      context.stageContext.rawSliceCount * context.spec->inputChunks,
      context.spec->datatype,
      context.spec->inputChunks,
  };
}

cocclPipelineStageOutput stageOutput(const cocclPipelineContext& context,
                                     void* workspace, int stage, int slice) {
  const int tempIndex = context.plan.stageOutputTemp[stage];
  if (tempIndex >= 0) {
    return {tempPtr(context, workspace, slice, tempIndex),
            context.plan.stageOutputCapacityBytes[stage]};
  }
  return {static_cast<char*>(context.spec->output) +
              (size_t)slice * context.stageContext.rawSliceBytes,
          context.plan.stageOutputCapacityBytes[stage]};
}

ncclResult_t runSerial(const cocclPipelineContext& context,
                       void* workspace) {
  cocclPipelineEdge edge = inputEdge(context, 0);
  for (int stage = 0; stage < context.spec->stageCount; ++stage) {
    const cocclPipelineStageOutput output =
        stageOutput(context, workspace, stage, 0);
    NCCLCHECK(cocclExecutePipelineStage(
        &context.stageContext, context.spec->stages + stage, &edge,
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

ncclResult_t runOverlap(const cocclPipelineContext& context, void* workspace,
                        cocclPipelineResources* resources) {
  CUDACHECK(cudaEventRecord(resources->inputReady, context.spec->stream));
  CUDACHECK(cudaStreamWaitEvent(
      resources->streams[cocclPipelinePhasePack], resources->inputReady, 0));

  for (int slice = 0; slice < context.depth; ++slice) {
    cocclPipelineEdge edge = inputEdge(context, slice);

    const cocclPipelineStage pack = cocclPipelinePack();
    const cocclPipelineStageOutput packOutput = {
        tempPtr(context, workspace, slice,
                context.plan.inputStagingTemp),
        edge.bytes};
    NCCLCHECK(cocclExecutePipelineStage(
        &context.stageContext, &pack, &edge, &packOutput,
        resources->streams[cocclPipelinePhasePack]));
    NCCLCHECK(recordPhase(resources, cocclPipelinePhasePack, slice));

    for (int stage = 0; stage < context.spec->stageCount; ++stage) {
      const int phase = cocclPipelinePhaseCompress + stage;
      NCCLCHECK(waitForPhase(resources, phase, phase - 1, slice));
      const cocclPipelineStageOutput output =
          stageOutput(context, workspace, stage, slice);
      NCCLCHECK(cocclExecutePipelineStage(
          &context.stageContext, context.spec->stages + stage, &edge,
          &output, resources->streams[phase]));
      NCCLCHECK(recordPhase(resources, phase, slice));
    }

    NCCLCHECK(waitForPhase(resources, cocclPipelinePhaseUnpack,
                            cocclPipelinePhaseDecompress, slice));
    const cocclPipelineStage unpack = cocclPipelineUnpack();
    const cocclPipelineStageOutput unpackOutput = {
        static_cast<char*>(context.spec->output) +
            (size_t)slice * context.stageContext.rawSliceBytes,
        edge.bytes};
    NCCLCHECK(cocclExecutePipelineStage(
        &context.stageContext, &unpack, &edge, &unpackOutput,
        resources->streams[cocclPipelinePhaseUnpack]));
    if (slice == context.depth - 1) {
      NCCLCHECK(recordPhase(resources, cocclPipelinePhaseUnpack, slice));
    }
  }

  CUDACHECK(cudaStreamWaitEvent(
      context.spec->stream,
      resources->events[cocclPipelinePhaseUnpack][context.depth - 1], 0));
  return ncclSuccess;
}

}  // namespace

ncclResult_t cocclRunPipeline(const cocclPipelineSpec* spec) {
  cocclPipelineContext context = {};
  NCCLCHECK(cocclPreparePipeline(spec, cocclGetConfig().pipeline.depth,
                                 &context));
  CUDACHECK(cudaSetDevice(spec->ownerComm->cudaDev));

  cocclBufferHandle workspace = {};
  NCCLCHECK(cocclGetBuffer(spec->ownerComm, context.plan.workspaceBytes,
                           spec->stream, &workspace));

  ncclResult_t result = ncclSuccess;
  if (context.depth == 1) {
    result = runSerial(context, workspace.ptr);
  } else {
    cocclPipelineResources* resources = nullptr;
    result = findOrCreateResources(spec->ownerComm, context.depth,
                                   &resources);
    if (result == ncclSuccess) {
      result = runOverlap(context, workspace.ptr, resources);
    }
  }

  if (result != ncclSuccess) (void)cudaDeviceSynchronize();
  const ncclResult_t releaseResult =
      cocclReleaseBuffer(&workspace, spec->stream);
  return result == ncclSuccess ? releaseResult : result;
}

ncclResult_t cocclPipelineCommDestroy(ncclComm_t comm) {
  auto found = resourcesByComm.find(comm);
  if (found != resourcesByComm.end()) {
    cocclPipelineResources* resources = found->second;
    resourcesByComm.erase(found);
    destroyResources(resources);
  }
  return ncclSuccess;
}
