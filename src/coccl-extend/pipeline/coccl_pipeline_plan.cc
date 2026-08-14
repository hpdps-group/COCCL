#include "coccl_pipeline_internal.h"

#include "comm.h"

namespace {

struct cocclPipelinePlannedEdge {
  size_t bytes;
  size_t totalElements;
  ncclDataType_t datatype;
  size_t logicalChunks;
  bool framed;
  size_t frameStrideBytes;
};

bool cocclPipelineBoundaryTemp(const cocclPipelinePlan& plan, int temp) {
  return temp == plan.inputStagingTemp || temp == plan.outputStagingTemp;
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

ncclResult_t cocclAssignPipelineTemp(cocclPipelinePlan* plan,
                                     const cocclPipelinePlannedEdge& edge,
                                     int* assignedTemp) {
  size_t logicalBytes = edge.bytes;
  size_t metadataOffset = 0;
  size_t metadataBytes = 0;
  if (edge.framed) {
    if (edge.frameStrideBytes == 0 || edge.logicalChunks == 0 ||
        !cocclPipelineCheckedMultiply(
            edge.logicalChunks, sizeof(cocclCompressorFrameMetadata),
            &metadataBytes) ||
        !cocclAlignPipelineBytes(edge.bytes, &metadataOffset) ||
        !cocclPipelineCheckedAdd(metadataOffset, metadataBytes,
                                 &logicalBytes)) {
      return ncclInvalidArgument;
    }
  }
  size_t capacity = 0;
  if (!cocclAlignPipelineBytes(logicalBytes, &capacity)) {
    return ncclInvalidArgument;
  }
  const int temp = plan->tempCount++;
  plan->temps[temp].logicalBytes = logicalBytes;
  plan->temps[temp].bytes = capacity;
  plan->temps[temp].payloadBytes = edge.bytes;
  plan->temps[temp].frameMetadataOffset = metadataOffset;
  plan->temps[temp].frameMetadataBytes = metadataBytes;
  plan->temps[temp].frameStrideBytes = edge.frameStrideBytes;
  *assignedTemp = temp;
  return ncclSuccess;
}

ncclResult_t cocclPlanRawPipelineEdge(
    const cocclPipelineContext* context, size_t chunks,
    cocclPipelinePlannedEdge* output) {
  size_t elements = 0;
  size_t bytes = 0;
  if (!cocclPipelineCheckedMultiply(context->rawSliceCount, chunks,
                                    &elements) ||
      !cocclPipelineCheckedMultiply(context->rawSliceBytes, chunks, &bytes)) {
    return ncclInvalidArgument;
  }
  *output = {bytes, elements, context->spec->datatype, chunks, false, 0};
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
  size_t outputChunks = 0;
  NCCLCHECK(cocclPipelineStageOutputChunks(
      stage, input.logicalChunks, &outputChunks));
  switch (stage.kind) {
    case cocclPipelineStageCompress: {
      const bool framed = cocclCompressorSupports(
          context->spec->compressor, cocclCompressorCapabilityFramed);
      size_t encodedBytes = 0;
      NCCLCHECK(cocclQueryPipelineEncodedBound(
          context, cocclCompressorOperationCompress, input, &encodedBytes));
      // Variable-length frames retain one raw-sized slot per chunk. The size
      // query describes wire bytes, not the physical random-access layout.
      encodedBytes = framed
          ? input.bytes
          : (encodedBytes < input.bytes ? encodedBytes : input.bytes);
      const size_t frameStride = framed ? input.bytes / input.logicalChunks : 0;
      if (framed && (input.bytes % input.logicalChunks != 0 ||
                     frameStride == 0)) {
        return ncclInvalidArgument;
      }
      *output = {encodedBytes, encodedBytes, ncclInt8, outputChunks,
                 framed, frameStride};
      break;
    }
    case cocclPipelineStageAllToAll:
    case cocclPipelineStagePack:
    case cocclPipelineStageUnpack:
      *output = input;
      output->logicalChunks = outputChunks;
      break;
    case cocclPipelineStageAllGather:
      if (!cocclPipelineCheckedMultiply(input.bytes,
                                        (size_t)stage.comm->nRanks,
                                        &output->bytes) ||
          !cocclPipelineCheckedMultiply(input.totalElements,
                                        (size_t)stage.comm->nRanks,
                                        &output->totalElements)) {
        return ncclInvalidArgument;
      }
      output->datatype = input.datatype;
      output->logicalChunks = outputChunks;
      output->framed = input.framed;
      output->frameStrideBytes = input.frameStrideBytes;
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
      const bool framed = cocclCompressorSupports(
          context->spec->compressor, cocclCompressorCapabilityFramed);
      const size_t frameStride =
          framed ? rawOutput.bytes / rawOutput.logicalChunks : 0;
      if (framed && (rawOutput.bytes % rawOutput.logicalChunks != 0 ||
                     frameStride == 0)) {
        return ncclInvalidArgument;
      }
      if (framed) plannedBytes = rawOutput.bytes;
      *output = {plannedBytes, plannedBytes, ncclInt8, outputChunks,
                 framed, frameStride};
      break;
    }
    case cocclPipelineStageDecompressReduce:
    case cocclPipelineStageDecompress:
      NCCLCHECK(cocclPlanRawPipelineEdge(context, outputChunks, output));
      break;
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
        plan, edge, &plan->inputStagingTemp));
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
            plan, output, &plan->outputStagingTemp));
      }
    } else {
      NCCLCHECK(cocclAssignPipelineTemp(
          plan, output, &plan->stageOutputTemp[stageIndex]));
    }
    edge = output;
  }
  plan->finalChunks = edge.logicalChunks;
  return cocclPlanPipelineWorkspace(plan, context->depth);
}

}  // namespace

ncclResult_t cocclPlanPipelineWorkspace(cocclPipelinePlan* plan, int depth) {
  if (plan == nullptr || plan->tempCount <= 0 || depth <= 0) {
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
  }

  // Exact layouts overlap only within one slice. The stage chain consumes that
  // slice before the final stage writes it back to the user buffer.
  *requireSerial = !matches;
  return ncclSuccess;
}

ncclResult_t cocclValidatePipelineSpec(const cocclPipelineSpec* spec) {
  if (spec == nullptr || spec->name == nullptr || spec->input == nullptr ||
      spec->output == nullptr || spec->ownerComm == nullptr ||
      !spec->compressor || spec->stages == nullptr || spec->stageCount <= 0 ||
      spec->stageCount > kMaxPipelineStages || spec->rawChunkCount == 0 ||
      spec->inputChunks == 0) {
    return ncclInvalidArgument;
  }
  return ncclSuccess;
}

ncclResult_t cocclPreparePipeline(cocclPipelineContext* context) {
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
      nullptr,
  };
  return cocclBuildPipelinePlan(context);
}
