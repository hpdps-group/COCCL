#include "coccl_pipeline_internal.h"

#include "checks.h"
#include "collectives.h"
#include "comm.h"

namespace {

bool buffersOverlap(const void* first, size_t firstBytes,
                    const void* second, size_t secondBytes) {
  const uintptr_t firstBegin = reinterpret_cast<uintptr_t>(first);
  const uintptr_t secondBegin = reinterpret_cast<uintptr_t>(second);
  return firstBegin < secondBegin + secondBytes &&
      secondBegin < firstBegin + firstBytes;
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
  temp.offset = 0;
  temp.logicalBytes = logicalBytes;
  if (!cocclAlignPipelineBytes(logicalBytes, &temp.alignedBytes)) {
    return ncclInvalidArgument;
  }
  *tempIndex = plan->tempCount++;
  return ncclSuccess;
}

}  // namespace

ncclResult_t cocclPlanUnifiedWorkspace(cocclPipelinePlan* plan, int depth) {
  size_t sliceBytes = plan->temps[0].alignedBytes;
  for (int temp = 1; temp < plan->tempCount; ++temp) {
    size_t adjacentBytes = 0;
    if (!cocclPipelineCheckedAdd(plan->temps[temp - 1].alignedBytes,
                                 plan->temps[temp].alignedBytes,
                                 &adjacentBytes)) {
      return ncclInvalidArgument;
    }
    if (adjacentBytes > sliceBytes) sliceBytes = adjacentBytes;
  }

  for (int temp = 0; temp < plan->tempCount; ++temp) {
    cocclPipelineTempPlan& item = plan->temps[temp];
    item.offset = temp % 2 == 0 ? 0 : sliceBytes - item.alignedBytes;
  }
  plan->sliceWorkspaceBytes = sliceBytes;
  if (!cocclPipelineCheckedMultiply(sliceBytes, (size_t)depth,
                                    &plan->workspaceBytes)) {
    return ncclInvalidArgument;
  }
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

  size_t sliceBytes = 0;
  if (!cocclPipelineCheckedMultiply(context->stageContext.rawSliceBytes,
                                    spec->inputChunks, &sliceBytes)) {
    return ncclInvalidArgument;
  }

  cocclPipelinePlan& plan = context->plan;
  plan.inputStagingTemp = -1;
  plan.outputStagingTemp = -1;
  for (int i = 0; i < kCocclPipelineExplicitStages; ++i) {
    plan.stageOutputTemp[i] = -1;
  }

  if (context->depth > 1 && spec->inputChunks > 1) {
    NCCLCHECK(addTemp(&plan, cocclPipelineTempInputStaging, sliceBytes,
                      &plan.inputStagingTemp));
  }
  for (int stage = 0; stage < spec->stageCount; ++stage) {
    size_t outputBytes = 0;
    if (!cocclPipelineCheckedMultiply(
            context->stageContext.rawSliceBytes, stageChunks[stage],
            &outputBytes)) {
      return ncclInvalidArgument;
    }
    plan.stageOutputCapacityBytes[stage] = outputBytes;
    if (stage + 1 < spec->stageCount) {
      NCCLCHECK(addTemp(&plan, outputRole(spec->stages[stage].kind),
                        outputBytes, &plan.stageOutputTemp[stage]));
    }
  }
  plan.finalChunks = chunks;

  if (context->depth > 1 && plan.finalChunks > 1) {
    const int finalStage = spec->stageCount - 1;
    NCCLCHECK(addTemp(&plan, cocclPipelineTempOutputStaging,
                      plan.stageOutputCapacityBytes[finalStage],
                      &plan.outputStagingTemp));
    plan.stageOutputTemp[finalStage] = plan.outputStagingTemp;
  }

  return cocclPlanUnifiedWorkspace(&plan, context->depth);
}
