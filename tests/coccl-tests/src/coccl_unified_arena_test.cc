#include "core/pipeline/coccl_pipeline_internal.h"

#include "comm.h"

#include <cstdio>
#include <cstdlib>

namespace {

void fail(const char* expression, int line) {
  std::fprintf(stderr, "line %d: %s\n", line, expression);
  std::exit(1);
}

#define EXPECT(expression) \
  do { if (!(expression)) fail(#expression, __LINE__); } while (0)

bool rangesOverlap(size_t firstOffset, size_t firstBytes,
                   size_t secondOffset, size_t secondBytes) {
  return firstOffset < secondOffset + secondBytes &&
      secondOffset < firstOffset + firstBytes;
}

void checkLayout(const cocclPipelinePlan& plan, int depth) {
  EXPECT(plan.registeredBytes ==
         (size_t)depth * plan.registeredSliceBytes);
  EXPECT(plan.totalBytes == plan.registeredBytes + plan.rawBytes);

  for (int temp = 0; temp < plan.tempCount; ++temp) {
    const cocclPipelineTempPlan& item = plan.temps[temp];
    EXPECT(item.alignedBytes % kCocclPipelineAlignment == 0);
    EXPECT(item.offset % kCocclPipelineAlignment == 0);
    EXPECT(item.logicalBytes <= item.alignedBytes);
    if (item.storage == cocclPipelineRawRing) {
      EXPECT(item.offset + 2 * item.alignedBytes <= plan.rawBytes);
    } else {
      EXPECT(item.offset + item.alignedBytes <=
             plan.registeredSliceBytes);
    }
    if (temp == 0) continue;
    const cocclPipelineTempPlan& previous = plan.temps[temp - 1];
    if (previous.storage == item.storage) {
      EXPECT(!rangesOverlap(previous.offset, previous.alignedBytes,
                            item.offset, item.alignedBytes));
    }
  }
}

void checkSyntheticLayouts() {
  cocclPipelinePlan single = {};
  single.tempCount = 1;
  single.inputStagingTemp = -1;
  single.outputStagingTemp = -1;
  single.temps[0].logicalBytes = 1024;
  single.temps[0].alignedBytes = 1024;
  EXPECT(cocclPlanPipelineWorkspace(&single, 4) == ncclSuccess);
  checkLayout(single, 4);
  EXPECT(single.workspaceKind == cocclPipelineWorkspaceUnified);
  EXPECT(single.registeredSliceBytes == 1024);
  EXPECT(single.totalBytes == 4096);

  const size_t sizes[] = {1024, 256, 2048, 512};
  cocclPipelinePlan varying = {};
  varying.tempCount = 4;
  varying.inputStagingTemp = -1;
  varying.outputStagingTemp = -1;
  for (int temp = 0; temp < varying.tempCount; ++temp) {
    varying.temps[temp].logicalBytes = sizes[temp];
    varying.temps[temp].alignedBytes = sizes[temp];
  }
  EXPECT(cocclPlanPipelineWorkspace(&varying, 3) == ncclSuccess);
  checkLayout(varying, 3);
  EXPECT(varying.workspaceKind == cocclPipelineWorkspaceUnified);
  EXPECT(varying.registeredSliceBytes == 2560);
  EXPECT(varying.temps[0].offset == varying.temps[2].offset);

  cocclPipelinePlan allGather = {};
  allGather.tempCount = 3;
  allGather.inputStagingTemp = -1;
  allGather.outputStagingTemp = -1;
  const size_t gatheredSizes[] = {256, 1024, 1024};
  for (int temp = 0; temp < allGather.tempCount; ++temp) {
    allGather.temps[temp].logicalBytes = gatheredSizes[temp];
    allGather.temps[temp].alignedBytes = gatheredSizes[temp];
  }
  EXPECT(cocclPlanPipelineWorkspace(&allGather, 4) == ncclSuccess);
  checkLayout(allGather, 4);
  EXPECT(allGather.workspaceKind == cocclPipelineWorkspaceUnified);
  EXPECT(allGather.registeredSliceBytes == 2048);
  EXPECT(allGather.temps[0].offset == allGather.temps[2].offset);

  cocclPipelinePlan overflow = {};
  overflow.tempCount = 2;
  overflow.inputStagingTemp = -1;
  overflow.outputStagingTemp = -1;
  overflow.temps[0].alignedBytes =
      SIZE_MAX - (kCocclPipelineAlignment - 1);
  overflow.temps[1].alignedBytes = kCocclPipelineAlignment;
  EXPECT(cocclPlanPipelineWorkspace(&overflow, 1) == ncclInvalidArgument);
}

cocclPipelineSpec makeSpec(const char* name, ncclComm_t comm,
                           const cocclPipelineStage* stages, int stageCount,
                           size_t totalInputBytes, size_t inputChunks,
                           cocclOperation operation) {
  return {
      name,
      reinterpret_cast<const void*>(0x100000000ULL),
      reinterpret_cast<void*>(0x400000000ULL),
      totalInputBytes / inputChunks / sizeof(float),
      inputChunks,
      ncclFloat32,
      comm,
      nullptr,
      stages,
      stageCount,
  };
}

void checkRecipe(cocclPipelineSpec* spec, int depth, int expectedTemps,
                 cocclPipelineWorkspaceKind expectedKind,
                 size_t expectedWorkspaceBytes) {
  cocclPipelineContext context = {};
  EXPECT(cocclPreparePipeline(spec, depth, &context) == ncclSuccess);
  EXPECT(context.depth == depth);
  EXPECT(context.plan.tempCount == expectedTemps);
  checkLayout(context.plan, depth);
  EXPECT(context.plan.workspaceKind == expectedKind);
  EXPECT(context.plan.totalBytes == expectedWorkspaceBytes);
}

void checkRecipes() {
  constexpr size_t rawBytes = 64ULL << 20;
  constexpr size_t oneShotBytes = 32ULL << 20;
  ncclComm owner = {};
  owner.nRanks = 4;
  ncclComm intra = {};
  intra.nRanks = 2;
  ncclComm inter = {};
  inter.nRanks = 2;

  const cocclPipelineStage allToAllStages[] = {
      cocclPipelineCompress(reinterpret_cast<void*>(0x1)), cocclPipelineAllToAll(&owner),
      cocclPipelineDecompress()};
  cocclPipelineSpec allToAll = makeSpec(
      "alltoall", &owner, allToAllStages, 3, rawBytes, 4,
      cocclOperation::AllToAll);
  checkRecipe(&allToAll, 4, 4, cocclPipelineWorkspaceUnified,
              2 * rawBytes);

  const cocclPipelineStage allGatherStages[] = {
      cocclPipelineCompress(reinterpret_cast<void*>(0x1)), cocclPipelineAllGather(&owner),
      cocclPipelineDecompress()};
  cocclPipelineSpec allGather = makeSpec(
      "allgather", &owner, allGatherStages, 3, rawBytes / 4, 1,
      cocclOperation::AllGather);
  checkRecipe(&allGather, 4, 3, cocclPipelineWorkspaceSplit,
              7 * rawBytes / 4);

  const cocclPipelineStage reduceScatterOneShotStages[] = {
      cocclPipelineCompress(reinterpret_cast<void*>(0x1)), cocclPipelineAllToAll(&owner),
      cocclPipelineDecompressReduce(4)};
  cocclPipelineSpec reduceScatterOneShot = makeSpec(
      "reducescatter-oneshot", &owner, reduceScatterOneShotStages, 3,
      rawBytes, 4, cocclOperation::ReduceScatter);
  checkRecipe(&reduceScatterOneShot, 4, 3,
              cocclPipelineWorkspaceUnified, 2 * rawBytes);

  const cocclPipelineStage reduceScatterTwoShotStages[] = {
      cocclPipelineCompress(reinterpret_cast<void*>(0x1)), cocclPipelineAllToAll(&intra),
      cocclPipelineDecompReduceComp(2, reinterpret_cast<void*>(0x1)),
      cocclPipelineAllToAll(&inter),
      cocclPipelineDecompressReduce(2)};
  cocclPipelineSpec reduceScatterTwoShot = makeSpec(
      "reducescatter-twoshot", &owner, reduceScatterTwoShotStages, 5,
      rawBytes, 4, cocclOperation::ReduceScatter);
  checkRecipe(&reduceScatterTwoShot, 4, 5,
              cocclPipelineWorkspaceUnified, 2 * rawBytes);

  const cocclPipelineStage allReduceOneShotStages[] = {
      cocclPipelineCompress(reinterpret_cast<void*>(0x1)), cocclPipelineAllGather(&owner),
      cocclPipelineDecompressReduce(4)};
  cocclPipelineSpec allReduceOneShot = makeSpec(
      "allreduce-oneshot", &owner, allReduceOneShotStages, 3,
      oneShotBytes, 4, cocclOperation::AllReduce);
  checkRecipe(&allReduceOneShot, 1, 2,
              cocclPipelineWorkspaceUnified, 5 * oneShotBytes);

  const cocclPipelineStage allReduceTwoShotStages[] = {
      cocclPipelineCompress(reinterpret_cast<void*>(0x1)), cocclPipelineAllToAll(&owner),
      cocclPipelineDecompReduceComp(4, reinterpret_cast<void*>(0x1)),
      cocclPipelineAllGather(&owner),
      cocclPipelineDecompress()};
  cocclPipelineSpec allReduceTwoShot = makeSpec(
      "allreduce-twoshot", &owner, allReduceTwoShotStages, 5,
      rawBytes, 4, cocclOperation::AllReduce);
  checkRecipe(&allReduceTwoShot, 4, 6,
              cocclPipelineWorkspaceUnified, 2 * rawBytes);

  const cocclPipelineStage allReduceTripleShotStages[] = {
      cocclPipelineCompress(reinterpret_cast<void*>(0x1)), cocclPipelineAllToAll(&intra),
      cocclPipelineDecompReduceComp(2, reinterpret_cast<void*>(0x1)),
      cocclPipelineAllToAll(&inter),
      cocclPipelineDecompReduceComp(2, reinterpret_cast<void*>(0x1)),
      cocclPipelineAllGather(&owner),
      cocclPipelineDecompress()};
  cocclPipelineSpec allReduceTripleShot = makeSpec(
      "allreduce-tripleshot", &owner, allReduceTripleShotStages, 7,
      rawBytes, 4, cocclOperation::AllReduce);
  checkRecipe(&allReduceTripleShot, 4, 8,
              cocclPipelineWorkspaceUnified, 2 * rawBytes);
}

}  // namespace

thread_local int ncclDebugNoWarn = 0;
void ncclDebugLog(ncclDebugLogLevel, unsigned long, const char*, int,
                  const char*, ...) {}

int main() {
  checkSyntheticLayouts();
  checkRecipes();
  std::printf("coccl unified arena: PASS\n");
  return 0;
}
