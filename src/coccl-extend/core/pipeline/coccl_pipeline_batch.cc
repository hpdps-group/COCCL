#include "core/pipeline/coccl_pipeline_execution.h"

#include "core/config/coccl_config.h"
#include "core/tuning/coccl_autotune_pipeline.h"
#include "core/compression/compress.h"
#include "comm.h"
#include "debug.h"

#include <algorithm>
#include <vector>

using namespace cocclPipelineInternal;

namespace {

struct cocclPipelineBatchState {
  cocclPipelineExecution execution;
  cocclPipelineSliceState slices[kCocclPipelineMaxDepth];
  int communicationStage;
};

struct alignas(16) cocclPipelineBatchPlan {
  uint64_t targetSliceBytes;
  uint32_t maxDepth;
  uint32_t reserved;
};

static_assert(sizeof(cocclPipelineBatchPlan) == 16,
              "Pipeline batch plan is a fixed control message");

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
    state.slices[slice].output = stageOutput(
        context, state.execution.workspace, stage, slice);
    const cocclPipelineStage& exchangeStage = context.spec->stages[stage];
    cocclCompressorFrameMetadata* metadata =
        exchangeStage.direction == cocclPipelineSend
        ? state.slices[slice].edge.frameMetadata
        : state.slices[slice].output.frameMetadata;
    frames.push_back({&state, slice, phase, metadata,
                      exchangeStage.direction == cocclPipelineSend
                      ? state.slices[slice].edge.frameStrideBytes
                      : state.slices[slice].output.frameStrideBytes});
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
            ? state.slices[slice].edge.ptr : nullptr,
        stage.direction == cocclPipelineRecv
            ? state.slices[slice].output.ptr : nullptr,
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
      applyReceivedFrame(stage, hostMetadata[i], &state.slices[slice].edge,
                         state.slices[slice].output);
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
          &state.slices[slice].edge, &output, resources->streams[phase]));
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
      state.slices[slice].edge = inputEdge(context, slice);
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
            &state.slices[slice].edge, &output, resources->streams[phase]));
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
