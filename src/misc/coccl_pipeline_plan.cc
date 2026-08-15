#include "coccl_pipeline_internal.h"

#include "checks.h"
#include "collectives.h"
#include "comm.h"

namespace {

ncclResult_t addTemp(cocclPipelinePlan* plan, cocclPipelineTempRole role,
                     size_t logicalBytes, int* tempIndex) {
  cocclPipelineTempPlan& temp = plan->temps[plan->tempCount];
  temp.role = role;
  temp.offset = plan->sliceWorkspaceBytes;
  temp.logicalBytes = logicalBytes;
  if (!cocclAlignPipelineBytes(logicalBytes, &temp.alignedBytes) ||
      !cocclPipelineCheckedAdd(plan->sliceWorkspaceBytes, temp.alignedBytes,
                               &plan->sliceWorkspaceBytes)) {
    return ncclInvalidArgument;
  }
  *tempIndex = plan->tempCount++;
  return ncclSuccess;
}

}  // namespace

ncclResult_t cocclPreparePipeline(const cocclPipelineSpec* spec,
                                  int requestedDepth,
                                  cocclPipelineContext* context) {
  if (spec == nullptr || context == nullptr || spec->input == nullptr ||
      spec->output == nullptr || spec->ownerComm == nullptr ||
      spec->rawChunkCount == 0 || spec->inputChunks == 0 ||
      spec->stages == nullptr ||
      spec->stageCount != kCocclPipelineExplicitStages) {
    return ncclInvalidArgument;
  }

  const int typeBytes = ncclTypeSize(spec->datatype);
  if (typeBytes <= 0) return ncclInvalidArgument;

  *context = {};
  context->spec = spec;
  context->depth = requestedDepth > 1 ? requestedDepth : 1;
  if (context->depth > kCocclPipelineMaxDepth ||
      spec->rawChunkCount % (size_t)context->depth != 0 ||
      spec->input == spec->output) {
    context->depth = 1;
  }

  context->stageContext.rawSliceCount =
      spec->rawChunkCount / (size_t)context->depth;
  if (!cocclPipelineCheckedMultiply(
          context->stageContext.rawSliceCount, (size_t)typeBytes,
          &context->stageContext.rawSliceBytes) ||
      !cocclPipelineCheckedMultiply(spec->rawChunkCount, (size_t)typeBytes,
                                    &context->stageContext.rawChunkBytes)) {
    return ncclInvalidArgument;
  }
  context->stageContext.rawDatatype = spec->datatype;
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
    plan.stageOutputCapacityBytes[i] = sliceBytes;
  }
  plan.finalChunks = spec->inputChunks;

  if (context->depth > 1 && spec->inputChunks > 1) {
    NCCLCHECK(addTemp(&plan, cocclPipelineTempInputStaging, sliceBytes,
                      &plan.inputStagingTemp));
  }
  NCCLCHECK(addTemp(&plan, cocclPipelineTempCompressOutput, sliceBytes,
                    &plan.stageOutputTemp[0]));
  NCCLCHECK(addTemp(&plan, cocclPipelineTempAllToAllOutput, sliceBytes,
                    &plan.stageOutputTemp[1]));
  if (context->depth > 1 && plan.finalChunks > 1) {
    NCCLCHECK(addTemp(&plan, cocclPipelineTempOutputStaging, sliceBytes,
                      &plan.outputStagingTemp));
    plan.stageOutputTemp[2] = plan.outputStagingTemp;
  }

  if (!cocclPipelineCheckedMultiply(plan.sliceWorkspaceBytes,
                                    (size_t)context->depth,
                                    &plan.workspaceBytes)) {
    return ncclInvalidArgument;
  }
  return ncclSuccess;
}
