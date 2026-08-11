#include "coccl_pipeline_internal.h"

#include "comm.h"

namespace {

struct cocclPipelinePlannedEdge {
  size_t bytes;
  size_t totalElements;
  ncclDataType_t datatype;
  size_t logicalChunks;
};

bool cocclPipelineBoundaryTemp(const cocclPipelinePlan& plan, int temp) {
  return temp == plan.inputStagingTemp || temp == plan.outputStagingTemp;
}

bool cocclValidLogicalPipelinePlan(const cocclPipelinePlan& plan, int depth) {
  if (depth <= 0 || plan.tempCount <= 0 ||
      plan.tempCount > kMaxPipelineTemps) {
    return false;
  }
  if (plan.inputStagingTemp < -1 || plan.outputStagingTemp < -1 ||
      plan.inputStagingTemp >= plan.tempCount ||
      plan.outputStagingTemp >= plan.tempCount ||
      (plan.inputStagingTemp >= 0 &&
       plan.inputStagingTemp == plan.outputStagingTemp)) {
    return false;
  }
  for (int i = 0; i < plan.tempCount; ++i) {
    const auto& temp = plan.temps[i];
    if (temp.logicalBytes == 0 || temp.bytes < temp.logicalBytes ||
        temp.bytes % kPipelineBufferAlignment != 0) {
      return false;
    }
  }
  return true;
}

bool cocclBuildUnifiedWorkspace(const cocclPipelinePlan& logical, int depth,
                                cocclPipelinePlan* candidate) {
  *candidate = logical;
  candidate->workspaceKind = cocclPipelineWorkspaceUnified;
  candidate->registeredSliceBytes = candidate->temps[0].bytes;
  candidate->rawBytes = 0;

  for (int i = 1; i < candidate->tempCount; ++i) {
    size_t adjacentBytes = 0;
    if (!cocclPipelineCheckedAdd(candidate->temps[i - 1].bytes,
                                 candidate->temps[i].bytes,
                                 &adjacentBytes)) {
      return false;
    }
    if (adjacentBytes > candidate->registeredSliceBytes) {
      candidate->registeredSliceBytes = adjacentBytes;
    }
  }

  for (int i = 0; i < candidate->tempCount; ++i) {
    auto& temp = candidate->temps[i];
    temp.storage = cocclPipelineRegisteredArena;
    temp.offset = (i % 2) == 0
        ? 0 : candidate->registeredSliceBytes - temp.bytes;
  }
  if (!cocclPipelineCheckedMultiply(candidate->registeredSliceBytes,
                                    (size_t)depth,
                                    &candidate->registeredBytes) ||
      candidate->registeredBytes == 0) {
    return false;
  }
  candidate->totalBytes = candidate->registeredBytes;
  return true;
}

bool cocclBuildSplitWorkspace(const cocclPipelinePlan& logical, int depth,
                              cocclPipelinePlan* candidate) {
  *candidate = logical;
  candidate->workspaceKind = cocclPipelineWorkspaceSplit;
  candidate->registeredSliceBytes = 0;
  candidate->registeredBytes = 0;
  candidate->rawBytes = 0;

  const int boundaryTemps[] = {
      candidate->inputStagingTemp, candidate->outputStagingTemp};
  for (int tempIndex : boundaryTemps) {
    if (tempIndex < 0) continue;
    auto& temp = candidate->temps[tempIndex];
    temp.storage = cocclPipelineRawRing;
    temp.offset = candidate->rawBytes;
    size_t ringBytes = 0;
    if (!cocclPipelineCheckedMultiply(temp.bytes,
                                      (size_t)kPipelineRawRingSlots,
                                      &ringBytes) ||
        !cocclPipelineCheckedAdd(candidate->rawBytes, ringBytes,
                                 &candidate->rawBytes)) {
      return false;
    }
  }

  int previous = -1;
  for (int i = 0; i < candidate->tempCount; ++i) {
    if (cocclPipelineBoundaryTemp(*candidate, i)) continue;
    if (previous < 0) {
      candidate->registeredSliceBytes = candidate->temps[i].bytes;
    } else {
      size_t adjacentBytes = 0;
      if (!cocclPipelineCheckedAdd(candidate->temps[previous].bytes,
                                   candidate->temps[i].bytes,
                                   &adjacentBytes)) {
        return false;
      }
      if (adjacentBytes > candidate->registeredSliceBytes) {
        candidate->registeredSliceBytes = adjacentBytes;
      }
    }
    previous = i;
  }
  if (previous < 0 || candidate->registeredSliceBytes == 0) return false;

  int corePosition = 0;
  for (int i = 0; i < candidate->tempCount; ++i) {
    if (cocclPipelineBoundaryTemp(*candidate, i)) continue;
    auto& temp = candidate->temps[i];
    temp.storage = cocclPipelineRegisteredArena;
    temp.offset = (corePosition++ % 2) == 0
        ? 0 : candidate->registeredSliceBytes - temp.bytes;
  }
  if (!cocclPipelineCheckedMultiply(candidate->registeredSliceBytes,
                                    (size_t)depth,
                                    &candidate->registeredBytes) ||
      !cocclPipelineCheckedAdd(candidate->registeredBytes,
                               candidate->rawBytes,
                               &candidate->totalBytes) ||
      candidate->registeredBytes == 0 || candidate->totalBytes == 0) {
    return false;
  }
  return true;
}

bool cocclStageCreatesTemp(cocclPipelineStageKind kind) {
  return kind == cocclPipelineStageCompress ||
      kind == cocclPipelineStageAllToAll ||
      kind == cocclPipelineStageAllGather ||
      kind == cocclPipelineStageDecompReduceComp;
}

ncclResult_t cocclAssignPipelineTemp(cocclPipelinePlan* plan,
                                     size_t logicalBytes,
                                     int* assignedTemp) {
  if (plan == nullptr || assignedTemp == nullptr || logicalBytes == 0 ||
      plan->tempCount < 0 || plan->tempCount >= kMaxPipelineTemps) {
    return ncclInvalidArgument;
  }

  size_t capacity = 0;
  if (!cocclAlignPipelineBytes(logicalBytes, &capacity)) {
    return ncclInvalidArgument;
  }
  const int temp = plan->tempCount++;
  plan->temps[temp].logicalBytes = logicalBytes;
  plan->temps[temp].bytes = capacity;
  *assignedTemp = temp;
  return ncclSuccess;
}

ncclResult_t cocclPlanRawPipelineEdge(
    const cocclPipelineContext* context, size_t chunks,
    cocclPipelinePlannedEdge* output) {
  if (context == nullptr || chunks == 0 || output == nullptr) {
    return ncclInvalidArgument;
  }
  size_t elements = 0;
  size_t bytes = 0;
  if (!cocclPipelineCheckedMultiply(context->rawSliceCount, chunks,
                                    &elements) ||
      !cocclPipelineCheckedMultiply(context->rawSliceBytes, chunks, &bytes)) {
    return ncclInvalidArgument;
  }
  *output = {bytes, elements, context->spec->datatype, chunks};
  return ncclSuccess;
}

ncclResult_t cocclQueryPipelineEncodedBound(
    const cocclPipelineContext* context,
    cocclCompressorOperation operation,
    const cocclPipelinePlannedEdge& rawShape, size_t* encodedBytes) {
  return cocclGetCompressorEncodedSizeBound(
      context->spec->compressor, operation, rawShape.totalElements,
      rawShape.logicalChunks, rawShape.datatype, encodedBytes);
}

ncclResult_t cocclPlanPipelineStageOutput(
    const cocclPipelineContext* context, const cocclPipelineStage& stage,
    const cocclPipelinePlannedEdge& input,
    cocclPipelinePlannedEdge* output) {
  if (context == nullptr || output == nullptr || input.bytes == 0 ||
      input.totalElements == 0 || input.logicalChunks == 0) {
    return ncclInvalidArgument;
  }

  size_t outputChunks = 0;
  NCCLCHECK(cocclPipelineStageOutputChunks(
      stage, input.logicalChunks, &outputChunks));
  switch (stage.kind) {
    case cocclPipelineStageCompress: {
      size_t encodedBytes = 0;
      NCCLCHECK(cocclQueryPipelineEncodedBound(
          context, cocclCompressorOperationCompress, input, &encodedBytes));
      encodedBytes = encodedBytes < input.bytes ? encodedBytes : input.bytes;
      *output = {encodedBytes, encodedBytes, ncclInt8, outputChunks};
      break;
    }
    case cocclPipelineStageAllToAll:
    case cocclPipelineStagePack:
    case cocclPipelineStageUnpack:
      *output = input;
      output->logicalChunks = outputChunks;
      break;
    case cocclPipelineStageAllGather:
      if (stage.comm == nullptr || stage.comm->nRanks <= 0 ||
          !cocclPipelineCheckedMultiply(input.bytes,
                                        (size_t)stage.comm->nRanks,
                                        &output->bytes) ||
          !cocclPipelineCheckedMultiply(input.totalElements,
                                        (size_t)stage.comm->nRanks,
                                        &output->totalElements)) {
        return ncclInvalidArgument;
      }
      output->datatype = input.datatype;
      output->logicalChunks = outputChunks;
      break;
    case cocclPipelineStageDecompReduceComp: {
      cocclPipelinePlannedEdge rawOutput = {};
      NCCLCHECK(cocclPlanRawPipelineEdge(context, outputChunks, &rawOutput));

      size_t genericBytes = 0;
      NCCLCHECK(cocclQueryPipelineEncodedBound(
          context, cocclCompressorOperationCompress, rawOutput,
          &genericBytes));
      size_t plannedBytes = genericBytes < rawOutput.bytes
          ? genericBytes : rawOutput.bytes;

      if (cocclCompressorSupports(
              context->spec->compressor,
              cocclCompressorCapabilityDecompressReduceCompress)) {
        size_t fusedBytes = 0;
        NCCLCHECK(cocclQueryPipelineEncodedBound(
            context, cocclCompressorOperationDecompressReduceCompress,
            rawOutput, &fusedBytes));
        if (fusedBytes <= rawOutput.bytes && fusedBytes > plannedBytes) {
          plannedBytes = fusedBytes;
        }
      }
      *output = {plannedBytes, plannedBytes, ncclInt8, outputChunks};
      break;
    }
    case cocclPipelineStageDecompressReduce:
    case cocclPipelineStageDecompress:
      NCCLCHECK(cocclPlanRawPipelineEdge(context, outputChunks, output));
      break;
    default:
      return ncclInvalidArgument;
  }

  if (output->bytes == 0 || output->totalElements == 0 ||
      output->logicalChunks != outputChunks ||
      output->totalElements % output->logicalChunks != 0) {
    return ncclInvalidUsage;
  }
  return ncclSuccess;
}

ncclResult_t cocclBuildPipelinePlan(cocclPipelineContext* context) {
  const auto* spec = context->spec;
  auto* plan = &context->plan;
  *plan = {};
  for (int i = 0; i < kMaxPipelineStages; ++i) {
    plan->stageOutputTemp[i] = -1;
  }
  plan->inputStagingTemp = -1;
  plan->outputStagingTemp = -1;

  cocclPipelinePlannedEdge edge = {};
  NCCLCHECK(cocclPlanRawPipelineEdge(context, spec->inputChunks, &edge));
  if (context->depth > 1 && spec->inputChunks > 1) {
    NCCLCHECK(cocclAssignPipelineTemp(
        plan, edge.bytes, &plan->inputStagingTemp));
  }

  for (int stageIndex = 0; stageIndex < spec->stageCount; ++stageIndex) {
    cocclPipelinePlannedEdge output = {};
    NCCLCHECK(cocclPlanPipelineStageOutput(
        context, spec->stages[stageIndex], edge, &output));
    plan->stageOutputCapacityBytes[stageIndex] = output.bytes;
    const bool finalStage = stageIndex + 1 == spec->stageCount;
    if (finalStage) {
      if (context->depth > 1 && output.logicalChunks > 1) {
        NCCLCHECK(cocclAssignPipelineTemp(
            plan, output.bytes, &plan->outputStagingTemp));
      }
    } else if (!cocclStageCreatesTemp(spec->stages[stageIndex].kind)) {
      return ncclInvalidArgument;
    } else {
      NCCLCHECK(cocclAssignPipelineTemp(
          plan, output.bytes, &plan->stageOutputTemp[stageIndex]));
    }
    edge = output;
  }
  plan->finalChunks = edge.logicalChunks;
  return cocclPlanPipelineWorkspace(plan, context->depth);
}

}  // namespace

ncclResult_t cocclPlanPipelineWorkspace(cocclPipelinePlan* plan, int depth) {
  if (plan == nullptr || !cocclValidLogicalPipelinePlan(*plan, depth)) {
    return ncclInvalidArgument;
  }

  cocclPipelinePlan unified = {};
  cocclPipelinePlan split = {};
  const bool unifiedValid =
      cocclBuildUnifiedWorkspace(*plan, depth, &unified);
  const bool splitValid = cocclBuildSplitWorkspace(*plan, depth, &split);
  if (!unifiedValid && !splitValid) return ncclInvalidArgument;

  *plan = splitValid && (!unifiedValid || split.totalBytes < unified.totalBytes)
      ? split : unified;
  return ncclSuccess;
}

ncclResult_t cocclPipelineUserBuffersRequireSerial(
    const void* input, size_t inputChunks, void* output, size_t outputChunks,
    size_t rawChunkBytes, int rank,
    cocclPipelineInPlaceLayout inPlaceLayout, bool* requireSerial) {
  if (input == nullptr || output == nullptr || inputChunks == 0 ||
      outputChunks == 0 || rawChunkBytes == 0 || requireSerial == nullptr) {
    return ncclInvalidArgument;
  }

  size_t inputBytes = 0;
  size_t outputBytes = 0;
  if (!cocclPipelineCheckedMultiply(rawChunkBytes, inputChunks, &inputBytes) ||
      !cocclPipelineCheckedMultiply(rawChunkBytes, outputChunks,
                                    &outputBytes)) {
    return ncclInvalidArgument;
  }

  const uintptr_t inputBegin = (uintptr_t)input;
  const uintptr_t outputBegin = (uintptr_t)output;
  if (inputBytes > UINTPTR_MAX - inputBegin ||
      outputBytes > UINTPTR_MAX - outputBegin) {
    return ncclInvalidArgument;
  }
  const uintptr_t inputEnd = inputBegin + inputBytes;
  const uintptr_t outputEnd = outputBegin + outputBytes;
  if (inputBegin >= outputEnd || outputBegin >= inputEnd) {
    *requireSerial = false;
    return ncclSuccess;
  }

  bool matches = false;
  switch (inPlaceLayout) {
    case cocclPipelineInPlaceNone:
      break;
    case cocclPipelineInPlaceSameBuffer:
      matches = inputChunks == outputChunks && inputBegin == outputBegin;
      break;
    case cocclPipelineInPlaceInputRankChunk:
    case cocclPipelineInPlaceOutputRankChunk: {
      if (rank < 0) return ncclInvalidArgument;
      const size_t rankChunk = (size_t)rank;
      const size_t availableChunks =
          inPlaceLayout == cocclPipelineInPlaceInputRankChunk
          ? outputChunks : inputChunks;
      size_t rankOffset = 0;
      if (rankChunk >= availableChunks ||
          !cocclPipelineCheckedMultiply(rawChunkBytes, rankChunk,
                                        &rankOffset)) {
        return ncclInvalidArgument;
      }

      if (inPlaceLayout == cocclPipelineInPlaceInputRankChunk) {
        if (rankOffset > UINTPTR_MAX - outputBegin) {
          return ncclInvalidArgument;
        }
        matches = inputChunks == 1 &&
            inputBegin == outputBegin + rankOffset;
      } else {
        if (rankOffset > UINTPTR_MAX - inputBegin) {
          return ncclInvalidArgument;
        }
        matches = outputChunks == 1 &&
            outputBegin == inputBegin + rankOffset;
      }
      break;
    }
    default:
      return ncclInvalidArgument;
  }

  // Exact layouts overlap only within one slice. The stage chain consumes that
  // slice before the final stage writes it back to the user buffer.
  *requireSerial = !matches;
  return ncclSuccess;
}

ncclResult_t cocclValidatePipelineSpec(const cocclPipelineSpec* spec) {
  if (spec == nullptr || spec->name == nullptr || spec->input == nullptr ||
      spec->output == nullptr || spec->ownerComm == nullptr ||
      !spec->compressor || spec->stages == nullptr || spec->stageCount < 2 ||
      spec->stageCount > kMaxPipelineStages || spec->rawChunkCount == 0 ||
      spec->inputChunks == 0) {
    return ncclInvalidArgument;
  }
  switch (spec->inPlaceLayout) {
    case cocclPipelineInPlaceNone:
    case cocclPipelineInPlaceSameBuffer:
    case cocclPipelineInPlaceInputRankChunk:
    case cocclPipelineInPlaceOutputRankChunk:
      break;
    default:
      return ncclInvalidArgument;
  }
  if (spec->stages[0].kind != cocclPipelineStageCompress) {
    return ncclInvalidArgument;
  }
  const cocclPipelineStageKind last =
      spec->stages[spec->stageCount - 1].kind;
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
      case cocclPipelineStagePack:
      case cocclPipelineStageUnpack:
      default:
        return ncclInvalidArgument;
    }
  }
  return ncclSuccess;
}

ncclResult_t cocclPreparePipeline(cocclPipelineContext* context) {
  if (context == nullptr || context->spec == nullptr || context->depth <= 0) {
    return ncclInvalidArgument;
  }
  context->rawSliceCount =
      context->spec->rawChunkCount / (size_t)context->depth;
  const int typeBytes = ncclTypeSize(context->spec->datatype);
  size_t rawChunkBytes = 0;
  if (context->rawSliceCount == 0 || typeBytes <= 0 ||
      !cocclPipelineCheckedMultiply(context->rawSliceCount,
                                    (size_t)typeBytes,
                                    &context->rawSliceBytes) ||
      !cocclPipelineCheckedMultiply(context->spec->rawChunkCount,
                                    (size_t)typeBytes, &rawChunkBytes)) {
    return ncclInvalidArgument;
  }
  context->stageContext = {
      context->rawSliceCount,
      rawChunkBytes,
      context->spec->datatype,
      context->spec->ownerComm,
      context->spec->compressor,
  };
  return cocclBuildPipelinePlan(context);
}
