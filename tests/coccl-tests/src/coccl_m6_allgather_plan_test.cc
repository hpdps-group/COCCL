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
    case cocclPipelineTempCompressOutput: return "compress-output";
    case cocclPipelineTempAllGatherOutput: return "allgather-output";
    case cocclPipelineTempOutputStaging: return "output-staging";
    case cocclPipelineTempInputStaging: return "input-staging";
    case cocclPipelineTempAllToAllOutput: return "alltoall-output";
  }
  return "unknown";
}

cocclPipelineSpec makeSpec(ncclComm_t comm, size_t gatheredBytes,
                           const cocclPipelineStage* stages) {
  const size_t rawChunkCount =
      gatheredBytes / (size_t)comm->nRanks / sizeof(float);
  return {
      "allgather",
      reinterpret_cast<const void*>(0x100000000ULL),
      reinterpret_cast<void*>(0x400000000ULL),
      rawChunkCount,
      1,
      ncclFloat32,
      cocclDefaultPolicy(cocclOperation::AllGather),
      comm,
      nullptr,
      stages,
      3,
  };
}

void checkDepth(ncclComm_t comm, int depth) {
  const cocclPipelineStage stages[] = {
      cocclPipelineCompress(), cocclPipelineAllGather(comm),
      cocclPipelineDecompress()};
  constexpr size_t gatheredBytes = 16 * 1024 * 1024;
  cocclPipelineSpec spec = makeSpec(comm, gatheredBytes, stages);
  cocclPipelineContext context = {};
  EXPECT(cocclPreparePipeline(&spec, depth, &context) == ncclSuccess);
  EXPECT(context.depth == depth);
  EXPECT(context.plan.finalChunks == 4);

  const size_t sendSliceBytes =
      gatheredBytes / (size_t)comm->nRanks / (size_t)depth;
  const size_t gatheredSliceBytes = gatheredBytes / (size_t)depth;
  size_t alignedSend = 0;
  size_t alignedGathered = 0;
  EXPECT(cocclAlignPipelineBytes(sendSliceBytes, &alignedSend));
  EXPECT(cocclAlignPipelineBytes(gatheredSliceBytes, &alignedGathered));
  EXPECT(context.plan.tempCount == (depth == 1 ? 2 : 3));
  if (depth <= 2) {
    const size_t expectedSliceBytes = depth == 1
        ? alignedSend + alignedGathered : 2 * alignedGathered;
    EXPECT(context.plan.workspaceKind == cocclPipelineWorkspaceUnified);
    EXPECT(context.plan.registeredSliceBytes == expectedSliceBytes);
    EXPECT(context.plan.registeredBytes ==
           (size_t)depth * expectedSliceBytes);
    EXPECT(context.plan.rawBytes == 0);
  } else {
    EXPECT(context.plan.workspaceKind == cocclPipelineWorkspaceSplit);
    EXPECT(context.plan.registeredSliceBytes ==
           alignedSend + alignedGathered);
    EXPECT(context.plan.registeredBytes ==
           (size_t)depth * (alignedSend + alignedGathered));
    EXPECT(context.plan.rawBytes == 2 * alignedGathered);
  }
  EXPECT(context.plan.totalBytes ==
         context.plan.registeredBytes + context.plan.rawBytes);
  EXPECT(context.plan.inputStagingTemp == -1);
  EXPECT(context.plan.stageOutputCapacityBytes[0] == sendSliceBytes);
  EXPECT(context.plan.stageOutputCapacityBytes[1] == gatheredSliceBytes);
  EXPECT(context.plan.stageOutputCapacityBytes[2] == gatheredSliceBytes);
  EXPECT(context.plan.temps[0].role == cocclPipelineTempCompressOutput);
  EXPECT(context.plan.temps[1].role == cocclPipelineTempAllGatherOutput);
  if (depth > 1) {
    EXPECT(context.plan.temps[2].role ==
           cocclPipelineTempOutputStaging);
  }
}

void dumpPlans(ncclComm_t comm) {
  const cocclPipelineStage stages[] = {
      cocclPipelineCompress(), cocclPipelineAllGather(comm),
      cocclPipelineDecompress()};
  const size_t sizes[] = {64ULL << 20, 1ULL << 30, 8ULL << 30};
  const int depths[] = {1, 2, 4, 8};
  std::printf("bytes,requested_depth,effective_depth,temp_index,temp_role,"
              "logical_bytes,aligned_bytes,offset,workspace_kind,storage,"
              "registered_slice_bytes,registered_bytes,raw_bytes,"
              "total_bytes\n");
  for (size_t bytes : sizes) {
    for (int depth : depths) {
      cocclPipelineSpec spec = makeSpec(comm, bytes, stages);
      cocclPipelineContext context = {};
      EXPECT(cocclPreparePipeline(&spec, depth, &context) == ncclSuccess);
      for (int temp = 0; temp < context.plan.tempCount; ++temp) {
        const cocclPipelineTempPlan& item = context.plan.temps[temp];
        std::printf("%zu,%d,%d,%d,%s,%zu,%zu,%zu,%s,%s,%zu,%zu,%zu,%zu\n",
                    bytes,
                    depth, context.depth, temp, roleName(item.role),
                    item.logicalBytes, item.alignedBytes, item.offset,
                    context.plan.workspaceKind == cocclPipelineWorkspaceUnified
                        ? "unified" : "split",
                    item.storage == cocclPipelineRegisteredArena
                        ? "registered" : "raw-ring",
                    context.plan.registeredSliceBytes,
                    context.plan.registeredBytes, context.plan.rawBytes,
                    context.plan.totalBytes);
      }
    }
  }
}

}  // namespace

thread_local int ncclDebugNoWarn = 0;
void ncclDebugLog(ncclDebugLogLevel, unsigned long, const char*, int,
                  const char*, ...) {}

int main(int argc, char** argv) {
  ncclComm comm = {};
  comm.nRanks = 4;
  comm.rank = 2;
  if (argc == 2 && std::strcmp(argv[1], "--csv") == 0) {
    dumpPlans(&comm);
    return 0;
  }
  checkDepth(&comm, 1);
  checkDepth(&comm, 2);
  checkDepth(&comm, 4);
  checkDepth(&comm, 8);

  const cocclPipelineStage stage = cocclPipelineAllGather(&comm);
  size_t chunks = 0;
  EXPECT(cocclPipelineStageOutputChunks(stage, 1, &chunks) == ncclSuccess);
  EXPECT(chunks == 4);
  comm.nRanks = 2;
  EXPECT(cocclPipelineStageOutputChunks(stage, SIZE_MAX, &chunks) ==
         ncclInvalidArgument);

  comm.nRanks = 4;
  const cocclPipelineStage stages[] = {
      cocclPipelineCompress(), cocclPipelineAllGather(&comm),
      cocclPipelineDecompress()};
  cocclPipelineSpec inPlace = makeSpec(&comm, 16 << 20, stages);
  const size_t rankBytes =
      inPlace.rawChunkCount * sizeof(float);
  inPlace.input = static_cast<char*>(inPlace.output) +
      (size_t)comm.rank * rankBytes;
  cocclPipelineContext context = {};
  EXPECT(cocclPreparePipeline(&inPlace, 8, &context) == ncclSuccess);
  EXPECT(context.depth == 1);
  std::printf("coccl M6 allgather plan: PASS\n");
  return 0;
}
