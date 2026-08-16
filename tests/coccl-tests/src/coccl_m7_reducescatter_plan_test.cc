#include "coccl_pipeline_internal.h"

#include "comm.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

void fail(const char* expression, int line) {
  std::fprintf(stderr, "line %d: %s\n", line, expression);
  std::exit(1);
}

#define EXPECT(expression) \
  do { if (!(expression)) fail(#expression, __LINE__); } while (0)

const char* roleName(cocclPipelineTempRole role) {
  switch (role) {
    case cocclPipelineTempInputStaging: return "input-staging";
    case cocclPipelineTempCompressOutput: return "compress-output";
    case cocclPipelineTempAllToAllOutput: return "alltoall-output";
    case cocclPipelineTempAllGatherOutput: return "allgather-output";
    case cocclPipelineTempDecompReduceCompOutput: return "drc-output";
    case cocclPipelineTempDecompressReduceOutput: return "dr-output";
    case cocclPipelineTempOutputStaging: return "output-staging";
  }
  return "unknown";
}

cocclPipelineSpec oneShotSpec(ncclComm_t comm, size_t outputBytes,
                              const cocclPipelineStage* stages) {
  return {
      "reducescatter-oneshot",
      reinterpret_cast<const void*>(0x100000000ULL),
      reinterpret_cast<void*>(0x400000000ULL),
      outputBytes / sizeof(float),
      (size_t)comm->nRanks,
      ncclFloat32,
      cocclDefaultPolicy(cocclOperation::ReduceScatter),
      comm,
      nullptr,
      stages,
      3,
  };
}

void checkOneShot(ncclComm_t comm, int depth) {
  const cocclPipelineStage stages[] = {
      cocclPipelineCompress(), cocclPipelineAllToAll(comm),
      cocclPipelineDecompressReduce((size_t)comm->nRanks)};
  constexpr size_t outputBytes = 16 << 20;
  cocclPipelineSpec spec = oneShotSpec(comm, outputBytes, stages);
  cocclPipelineContext context = {};
  EXPECT(cocclPreparePipeline(&spec, depth, &context) == ncclSuccess);
  EXPECT(context.depth == depth && context.plan.finalChunks == 1);

  const size_t inputSliceBytes =
      outputBytes * (size_t)comm->nRanks / (size_t)depth;
  size_t aligned = 0;
  EXPECT(cocclAlignPipelineBytes(inputSliceBytes, &aligned));
  EXPECT(context.plan.tempCount == (depth == 1 ? 2 : 3));
  EXPECT(context.plan.workspaceBytes ==
         (size_t)(depth == 1 ? 2 : 3) * aligned * (size_t)depth);
  EXPECT(context.plan.outputStagingTemp == -1);
  EXPECT(context.plan.stageOutputCapacityBytes[0] == inputSliceBytes);
  EXPECT(context.plan.stageOutputCapacityBytes[1] == inputSliceBytes);
  EXPECT(context.plan.stageOutputCapacityBytes[2] ==
         outputBytes / (size_t)depth);
}

void checkTwoShot(ncclComm_t owner, ncclComm_t intra, ncclComm_t inter,
                  size_t expectedNodes) {
  const size_t localRanks = (size_t)intra->nRanks;
  const cocclPipelineStage stages[] = {
      cocclPipelineCompress(), cocclPipelineAllToAll(intra),
      cocclPipelineDecompReduceComp(localRanks),
      cocclPipelineAllToAll(inter),
      cocclPipelineDecompressReduce(expectedNodes)};
  constexpr size_t outputBytes = 16 << 20;
  cocclPipelineSpec spec = {
      "reducescatter-twoshot",
      reinterpret_cast<const void*>(0x100000000ULL),
      reinterpret_cast<void*>(0x400000000ULL),
      outputBytes / sizeof(float),
      (size_t)owner->nRanks,
      ncclFloat32,
      cocclHierarchicalPolicy(cocclOperation::ReduceScatter),
      owner,
      nullptr,
      stages,
      5,
  };
  cocclPipelineContext context = {};
  EXPECT(cocclPreparePipeline(&spec, 4, &context) == ncclSuccess);
  EXPECT(context.plan.finalChunks == 1);
  EXPECT(context.plan.stageOutputCapacityBytes[0] == outputBytes);
  EXPECT(context.plan.stageOutputCapacityBytes[1] == outputBytes);
  EXPECT(context.plan.stageOutputCapacityBytes[2] ==
         outputBytes * expectedNodes / 4);
  EXPECT(context.plan.stageOutputCapacityBytes[3] ==
         outputBytes * expectedNodes / 4);
  EXPECT(context.plan.stageOutputCapacityBytes[4] == outputBytes / 4);
  EXPECT(context.plan.tempCount == 5);
}

void dumpPlans(ncclComm_t comm) {
  const cocclPipelineStage stages[] = {
      cocclPipelineCompress(), cocclPipelineAllToAll(comm),
      cocclPipelineDecompressReduce((size_t)comm->nRanks)};
  const size_t sizes[] = {64ULL << 20, 1ULL << 30, 8ULL << 30};
  const int depths[] = {1, 2, 4, 8};
  std::printf("bytes,requested_depth,effective_depth,temp_index,temp_role,"
              "logical_bytes,aligned_bytes,offset,slice_workspace_bytes,"
              "workspace_bytes\n");
  for (size_t bytes : sizes) {
    for (int depth : depths) {
      cocclPipelineSpec spec = oneShotSpec(comm, bytes, stages);
      cocclPipelineContext context = {};
      EXPECT(cocclPreparePipeline(&spec, depth, &context) == ncclSuccess);
      for (int temp = 0; temp < context.plan.tempCount; ++temp) {
        const cocclPipelineTempPlan& item = context.plan.temps[temp];
        std::printf("%zu,%d,%d,%d,%s,%zu,%zu,%zu,%zu,%zu\n", bytes,
                    depth, context.depth, temp, roleName(item.role),
                    item.logicalBytes, item.alignedBytes, item.offset,
                    context.plan.sliceWorkspaceBytes,
                    context.plan.workspaceBytes);
      }
    }
  }
}

}  // namespace

thread_local int ncclDebugNoWarn = 0;
void ncclDebugLog(ncclDebugLogLevel, unsigned long, const char*, int,
                  const char*, ...) {}

int main(int argc, char** argv) {
  ncclComm owner = {};
  owner.nRanks = 4;
  owner.rank = 1;
  if (argc == 2 && std::strcmp(argv[1], "--csv") == 0) {
    dumpPlans(&owner);
    return 0;
  }

  checkOneShot(&owner, 1);
  checkOneShot(&owner, 2);
  checkOneShot(&owner, 4);
  checkOneShot(&owner, 8);

  ncclComm intra = {};
  ncclComm inter = {};
  intra.nRanks = 4;
  inter.nRanks = 1;
  checkTwoShot(&owner, &intra, &inter, 1);
  intra.nRanks = 2;
  inter.nRanks = 2;
  checkTwoShot(&owner, &intra, &inter, 2);

  size_t chunks = 0;
  const cocclPipelineStage invalid = cocclPipelineDecompReduceComp(3);
  EXPECT(cocclPipelineStageOutputChunks(invalid, 4, &chunks) ==
         ncclInvalidArgument);

  const cocclPipelineStage stages[] = {
      cocclPipelineCompress(), cocclPipelineAllToAll(&owner),
      cocclPipelineDecompressReduce(4)};
  cocclPipelineSpec overflow = oneShotSpec(&owner, 16 << 20, stages);
  overflow.rawChunkCount = SIZE_MAX;
  cocclPipelineContext context = {};
  EXPECT(cocclPreparePipeline(&overflow, 1, &context) ==
         ncclInvalidArgument);
  std::printf("coccl M7 reducescatter plan: PASS\n");
  return 0;
}
