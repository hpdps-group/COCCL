#include "coccl_pipeline_internal.h"

#include "checks.h"
#include "collectives.h"
#include "comm.h"
#include "compress.h"

namespace {

struct cocclPipelinePlannedEdge {
  size_t bytes;
  size_t totalElements;
  ncclDataType_t datatype;
  size_t logicalChunks;
};

bool buffersOverlap(const void* first, size_t firstBytes,
                    const void* second, size_t secondBytes) {
  const uintptr_t firstBegin = reinterpret_cast<uintptr_t>(first);
  const uintptr_t secondBegin = reinterpret_cast<uintptr_t>(second);
  return firstBegin < secondBegin + secondBytes &&
      secondBegin < firstBegin + firstBytes;
}

bool boundaryTemp(const cocclPipelinePlan& plan, int temp) {
  return temp == plan.inputStagingTemp || temp == plan.outputStagingTemp;
}

bool buildUnifiedWorkspace(const cocclPipelinePlan& logical, int depth,
                           cocclPipelinePlan* candidate) {
  *candidate = logical;
  candidate->workspaceKind = cocclPipelineWorkspaceUnified;
  candidate->registeredSliceBytes = candidate->temps[0].alignedBytes;
  candidate->rawBytes = 0;

  for (int temp = 1; temp < candidate->tempCount; ++temp) {
    size_t adjacentBytes = 0;
    if (!cocclPipelineCheckedAdd(
            candidate->temps[temp - 1].alignedBytes,
            candidate->temps[temp].alignedBytes, &adjacentBytes)) {
      return false;
    }
    if (adjacentBytes > candidate->registeredSliceBytes) {
      candidate->registeredSliceBytes = adjacentBytes;
    }
  }

  for (int temp = 0; temp < candidate->tempCount; ++temp) {
    cocclPipelineTempPlan& item = candidate->temps[temp];
    item.storage = cocclPipelineRegisteredArena;
    item.offset = temp % 2 == 0
        ? 0 : candidate->registeredSliceBytes - item.alignedBytes;
  }
  if (!cocclPipelineCheckedMultiply(candidate->registeredSliceBytes,
                                    (size_t)depth,
                                    &candidate->registeredBytes)) {
    return false;
  }
  candidate->totalBytes = candidate->registeredBytes;
  return true;
}

bool buildSplitWorkspace(const cocclPipelinePlan& logical, int depth,
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
    cocclPipelineTempPlan& item = candidate->temps[tempIndex];
    item.storage = cocclPipelineRawRing;
    item.offset = candidate->rawBytes;
    size_t ringBytes = 0;
    if (!cocclPipelineCheckedMultiply(
            item.alignedBytes, (size_t)kCocclPipelineRawRingSlots,
            &ringBytes) ||
        !cocclPipelineCheckedAdd(candidate->rawBytes, ringBytes,
                                 &candidate->rawBytes)) {
      return false;
    }
  }

  int previous = -1;
  for (int temp = 0; temp < candidate->tempCount; ++temp) {
    if (boundaryTemp(*candidate, temp)) continue;
    if (previous < 0) {
      candidate->registeredSliceBytes =
          candidate->temps[temp].alignedBytes;
    } else {
      size_t adjacentBytes = 0;
      if (!cocclPipelineCheckedAdd(
              candidate->temps[previous].alignedBytes,
              candidate->temps[temp].alignedBytes, &adjacentBytes)) {
        return false;
      }
      if (adjacentBytes > candidate->registeredSliceBytes) {
        candidate->registeredSliceBytes = adjacentBytes;
      }
    }
    previous = temp;
  }
  if (previous < 0) return false;

  int corePosition = 0;
  for (int temp = 0; temp < candidate->tempCount; ++temp) {
    if (boundaryTemp(*candidate, temp)) continue;
    cocclPipelineTempPlan& item = candidate->temps[temp];
    item.storage = cocclPipelineRegisteredArena;
    item.offset = corePosition++ % 2 == 0
        ? 0 : candidate->registeredSliceBytes - item.alignedBytes;
  }
  if (!cocclPipelineCheckedMultiply(candidate->registeredSliceBytes,
                                    (size_t)depth,
                                    &candidate->registeredBytes) ||
      !cocclPipelineCheckedAdd(candidate->registeredBytes,
                               candidate->rawBytes,
                               &candidate->totalBytes)) {
    return false;
  }
  return true;
}

cocclPipelineTempRole outputRole(cocclPipelineStageKind kind) {
  switch (kind) {
    case cocclPipelineStageCompress:
      return cocclPipelineTempCompressOutput;
    case cocclPipelineStageAllToAll:
      return cocclPipelineTempAllToAllOutput;
    case cocclPipelineStageAllGather:
      return cocclPipelineTempAllGatherOutput;
    case cocclPipelineStageDecompReduceComp:
      return cocclPipelineTempDecompReduceCompOutput;
    case cocclPipelineStageDecompressReduce:
      return cocclPipelineTempDecompressReduceOutput;
    case cocclPipelineStageDecompress:
    case cocclPipelineStagePack:
    case cocclPipelineStageUnpack:
      return cocclPipelineTempOutputStaging;
  }
  __builtin_unreachable();
}

ncclResult_t addTemp(cocclPipelinePlan* plan, cocclPipelineTempRole role,
                     size_t logicalBytes, int* tempIndex) {
  cocclPipelineTempPlan& temp = plan->temps[plan->tempCount];
  temp.role = role;
  temp.logicalBytes = logicalBytes;
  if (!cocclAlignPipelineBytes(logicalBytes, &temp.alignedBytes)) {
    return ncclInvalidArgument;
  }
  *tempIndex = plan->tempCount++;
  return ncclSuccess;
}

ncclResult_t planRawEdge(const cocclPipelineContext* context, size_t chunks,
                         cocclPipelinePlannedEdge* output) {
  if (!cocclPipelineCheckedMultiply(
          context->stageContext.rawSliceCount, chunks,
          &output->totalElements) ||
      !cocclPipelineCheckedMultiply(
          context->stageContext.rawSliceBytes, chunks, &output->bytes)) {
    return ncclInvalidArgument;
  }
  output->datatype = context->spec->datatype;
  output->logicalChunks = chunks;
  return ncclSuccess;
}

ncclResult_t queryEncodedBound(
    const cocclPipelineContext* context,
    cocclCompressorOperation operation,
    const cocclPipelinePlannedEdge& rawShape, size_t* encodedBytes) {
  return cocclGetCompressorEncodedSizeBound(
      context->spec->compressorPolicy, operation, rawShape.totalElements,
      rawShape.logicalChunks, rawShape.datatype, encodedBytes);
}

ncclResult_t planStageOutput(
    const cocclPipelineContext* context, const cocclPipelineStage& stage,
    size_t outputChunks, const cocclPipelinePlannedEdge& input,
    cocclPipelinePlannedEdge* output) {
  switch (stage.kind) {
    case cocclPipelineStageCompress: {
      size_t encodedBytes = 0;
      NCCLCHECK(queryEncodedBound(
          context, cocclCompressorOperationCompress, input,
          &encodedBytes));
      if (encodedBytes > input.bytes) encodedBytes = input.bytes;
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
      if (!cocclPipelineCheckedMultiply(
              input.bytes, (size_t)stage.comm->nRanks, &output->bytes) ||
          !cocclPipelineCheckedMultiply(
              input.totalElements, (size_t)stage.comm->nRanks,
              &output->totalElements)) {
        return ncclInvalidArgument;
      }
      output->datatype = input.datatype;
      output->logicalChunks = outputChunks;
      break;
    case cocclPipelineStageDecompReduceComp: {
      cocclPipelinePlannedEdge rawOutput = {};
      NCCLCHECK(planRawEdge(context, outputChunks, &rawOutput));
      size_t genericBytes = 0;
      NCCLCHECK(queryEncodedBound(
          context, cocclCompressorOperationCompress, rawOutput,
          &genericBytes));
      size_t plannedBytes = genericBytes < rawOutput.bytes
          ? genericBytes : rawOutput.bytes;

      if (cocclCompressorPolicySupports(
              context->spec->compressorPolicy,
              cocclCompressorCapabilityDecompressReduceCompress)) {
        size_t fusedBytes = 0;
        NCCLCHECK(queryEncodedBound(
            context,
            cocclCompressorOperationDecompressReduceCompress,
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
      NCCLCHECK(planRawEdge(context, outputChunks, output));
      break;
  }
  return ncclSuccess;
}

}  // namespace

ncclResult_t cocclPlanPipelineWorkspace(cocclPipelinePlan* plan, int depth) {
  cocclPipelinePlan unified = {};
  cocclPipelinePlan split = {};
  const bool unifiedValid = buildUnifiedWorkspace(*plan, depth, &unified);
  const bool splitValid = buildSplitWorkspace(*plan, depth, &split);
  if (!unifiedValid && !splitValid) return ncclInvalidArgument;
  *plan = splitValid && (!unifiedValid || split.totalBytes < unified.totalBytes)
      ? split : unified;
  return ncclSuccess;
}

ncclResult_t cocclPipelineStageOutputChunks(
    const cocclPipelineStage& stage, size_t inputChunks,
    size_t* outputChunks) {
  switch (stage.kind) {
    case cocclPipelineStageCompress:
    case cocclPipelineStageAllToAll:
    case cocclPipelineStageDecompress:
    case cocclPipelineStagePack:
    case cocclPipelineStageUnpack:
      *outputChunks = inputChunks;
      return ncclSuccess;
    case cocclPipelineStageAllGather:
      if (!cocclPipelineCheckedMultiply(
              inputChunks, (size_t)stage.comm->nRanks, outputChunks)) {
        return ncclInvalidArgument;
      }
      return ncclSuccess;
    case cocclPipelineStageDecompReduceComp:
    case cocclPipelineStageDecompressReduce:
      if (stage.reduceChunks == 0 ||
          inputChunks % stage.reduceChunks != 0) {
        return ncclInvalidArgument;
      }
      *outputChunks = inputChunks / stage.reduceChunks;
      return ncclSuccess;
  }
  __builtin_unreachable();
}

ncclResult_t cocclPreparePipeline(const cocclPipelineSpec* spec,
                                  int requestedDepth,
                                  cocclPipelineContext* context) {
  if (spec == nullptr || context == nullptr || spec->input == nullptr ||
      spec->output == nullptr || spec->ownerComm == nullptr ||
      spec->rawChunkCount == 0 || spec->inputChunks == 0 ||
      spec->stages == nullptr || spec->stageCount <= 0 ||
      spec->stageCount > kCocclPipelineExplicitStages) {
    return ncclInvalidArgument;
  }

  const int typeBytes = ncclTypeSize(spec->datatype);
  if (typeBytes <= 0) return ncclInvalidArgument;

  *context = {};
  context->spec = spec;
  context->depth = requestedDepth > 1 ? requestedDepth : 1;
  if (context->depth > kCocclPipelineMaxDepth ||
      spec->rawChunkCount % (size_t)context->depth != 0) {
    context->depth = 1;
  }

  if (!cocclPipelineCheckedMultiply(
          spec->rawChunkCount, (size_t)typeBytes,
          &context->stageContext.rawChunkBytes)) {
    return ncclInvalidArgument;
  }

  size_t stageChunks[kCocclPipelineExplicitStages] = {};
  size_t chunks = spec->inputChunks;
  for (int stage = 0; stage < spec->stageCount; ++stage) {
    NCCLCHECK(cocclPipelineStageOutputChunks(
        spec->stages[stage], chunks, &stageChunks[stage]));
    chunks = stageChunks[stage];
  }
  size_t inputBytes = 0;
  size_t outputBytes = 0;
  if (!cocclPipelineCheckedMultiply(
          context->stageContext.rawChunkBytes, spec->inputChunks,
          &inputBytes) ||
      !cocclPipelineCheckedMultiply(
          context->stageContext.rawChunkBytes, chunks, &outputBytes)) {
    return ncclInvalidArgument;
  }
  if (buffersOverlap(spec->input, inputBytes, spec->output, outputBytes)) {
    context->depth = 1;
  }

  context->stageContext.rawSliceCount =
      spec->rawChunkCount / (size_t)context->depth;
  if (!cocclPipelineCheckedMultiply(
          context->stageContext.rawSliceCount, (size_t)typeBytes,
          &context->stageContext.rawSliceBytes)) {
    return ncclInvalidArgument;
  }
  context->stageContext.rawDatatype = spec->datatype;
  context->stageContext.compressorPolicy = spec->compressorPolicy;
  context->stageContext.ownerComm = spec->ownerComm;

  cocclPipelinePlan& plan = context->plan;
  plan.inputStagingTemp = -1;
  plan.outputStagingTemp = -1;
  for (int stage = 0; stage < kCocclPipelineExplicitStages; ++stage) {
    plan.stageOutputTemp[stage] = -1;
  }

  cocclPipelinePlannedEdge edge = {};
  NCCLCHECK(planRawEdge(context, spec->inputChunks, &edge));
  if (context->depth > 1 && spec->inputChunks > 1) {
    NCCLCHECK(addTemp(&plan, cocclPipelineTempInputStaging, edge.bytes,
                      &plan.inputStagingTemp));
  }

  for (int stage = 0; stage < spec->stageCount; ++stage) {
    cocclPipelinePlannedEdge output = {};
    NCCLCHECK(planStageOutput(context, spec->stages[stage],
                              stageChunks[stage], edge, &output));
    plan.stageOutputCapacityBytes[stage] = output.bytes;
    const bool finalStage = stage + 1 == spec->stageCount;
    if (finalStage) {
      if (context->depth > 1 && output.logicalChunks > 1) {
        NCCLCHECK(addTemp(&plan, cocclPipelineTempOutputStaging,
                          output.bytes, &plan.outputStagingTemp));
        plan.stageOutputTemp[stage] = plan.outputStagingTemp;
      }
    } else {
      NCCLCHECK(addTemp(&plan, outputRole(spec->stages[stage].kind),
                        output.bytes, &plan.stageOutputTemp[stage]));
    }
    edge = output;
  }
  plan.finalChunks = edge.logicalChunks;
  return cocclPlanPipelineWorkspace(&plan, context->depth);
}
