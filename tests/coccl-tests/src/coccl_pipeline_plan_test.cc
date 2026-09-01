#include "core/pipeline/coccl_pipeline_internal.h"

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

bool rangesOverlap(size_t firstOffset, size_t firstBytes,
                   size_t secondOffset, size_t secondBytes) {
  return firstOffset < secondOffset + secondBytes &&
      secondOffset < firstOffset + firstBytes;
}

const char* roleName(cocclPipelineTempRole role) {
  switch (role) {
    case cocclPipelineTempInputStaging: return "input-staging";
    case cocclPipelineTempCompressOutput: return "compress-output";
    case cocclPipelineTempAllToAllOutput: return "alltoall-output";
    case cocclPipelineTempAllGatherOutput: return "allgather-output";
    case cocclPipelineTempOutputStaging: return "output-staging";
  }
  return "unknown";
}

cocclPipelineSpec makeSpec(ncclComm_t comm, size_t totalBytes,
                           bool sameBuffer,
                           const cocclPipelineStage* stages) {
  const size_t chunks = (size_t)comm->nRanks;
  const size_t rawChunkCount = totalBytes / chunks / sizeof(float);
  return {
      "alltoall",
      reinterpret_cast<const void*>(0x100000000ULL),
      reinterpret_cast<void*>(sameBuffer ? 0x100000000ULL : 0x400000000ULL),
      rawChunkCount,
      chunks,
      ncclFloat32,
      comm,
      nullptr,
      stages,
      3,
      cocclPipelineInPlaceNone,
      cocclPipelineInputContiguous,
  };
}

void checkDepth(ncclComm_t comm, int depth) {
  const cocclPipelineStage stages[] = {
      cocclPipelineCompress(reinterpret_cast<void*>(0x1)), cocclPipelineAllToAll(comm),
      cocclPipelineDecompress()};
  constexpr size_t totalBytes = 16 * 1024 * 1024;
  cocclPipelineSpec spec = makeSpec(comm, totalBytes, false, stages);
  cocclPipelineContext context = {};
  EXPECT(cocclPreparePipeline(&spec, depth, &context) == ncclSuccess);
  EXPECT(context.depth == depth);
  EXPECT(context.plan.finalChunks == 4);

  const size_t sliceBytes = totalBytes / (size_t)depth;
  size_t aligned = 0;
  EXPECT(cocclAlignPipelineBytes(sliceBytes, &aligned));
  const int expectedTemps = depth == 1 ? 2 : 4;
  EXPECT(context.plan.tempCount == expectedTemps);
  EXPECT(context.plan.workspaceKind == cocclPipelineWorkspaceUnified);
  EXPECT(context.plan.registeredSliceBytes == 2 * aligned);
  EXPECT(context.plan.registeredBytes ==
         (size_t)depth * 2 * aligned);
  EXPECT(context.plan.rawBytes == 0);
  EXPECT(context.plan.totalBytes == context.plan.registeredBytes);
  EXPECT(context.plan.stageOutputTemp[0] >= 0);
  EXPECT(context.plan.stageOutputTemp[1] >= 0);
  EXPECT((context.plan.stageOutputTemp[2] >= 0) == (depth > 1));
  EXPECT((context.plan.inputStagingTemp >= 0) == (depth > 1));
  EXPECT((context.plan.outputStagingTemp >= 0) == (depth > 1));

  for (int i = 0; i < context.plan.tempCount; ++i) {
    const cocclPipelineTempPlan& temp = context.plan.temps[i];
    EXPECT(temp.logicalBytes == sliceBytes);
    EXPECT(temp.alignedBytes == aligned);
    EXPECT(temp.offset % kCocclPipelineAlignment == 0);
    EXPECT(temp.offset + temp.alignedBytes <=
           context.plan.registeredSliceBytes);
    EXPECT(temp.storage == cocclPipelineRegisteredArena);
    if (i > 0) {
      const cocclPipelineTempPlan& previous = context.plan.temps[i - 1];
      EXPECT(!rangesOverlap(previous.offset, previous.alignedBytes,
                            temp.offset, temp.alignedBytes));
    }
    if (i > 1) EXPECT(temp.offset == context.plan.temps[i - 2].offset);
  }
}

void checkTargetSlice(ncclComm_t comm) {
  const cocclPipelineStage stages[] = {
      cocclPipelineCompress(reinterpret_cast<void*>(0x1)),
      cocclPipelineAllToAll(comm), cocclPipelineDecompress()};
  cocclPipelineContext context = {};

  cocclPipelineSpec tail = makeSpec(comm, 100ULL << 20, false, stages);
  EXPECT(cocclPreparePipelineForSlice(
             &tail, 32ULL << 20, 16, &context) == ncclSuccess);
  EXPECT(context.depth == 4);
  EXPECT(context.slices[0].bytes == 8ULL << 20);
  EXPECT(context.slices[1].bytes == 8ULL << 20);
  EXPECT(context.slices[2].bytes == 8ULL << 20);
  EXPECT(context.slices[3].bytes == 1ULL << 20);

  cocclPipelineSpec capped = makeSpec(comm, 1ULL << 30, false, stages);
  EXPECT(cocclPreparePipelineForSlice(
             &capped, 32ULL << 20, 16, &context) == ncclSuccess);
  EXPECT(context.depth == 16);
  EXPECT(context.slices[0].bytes == 16ULL << 20);
  EXPECT(context.slices[15].bytes == 16ULL << 20);
}

void dumpPlans(ncclComm_t comm) {
  const cocclPipelineStage stages[] = {
      cocclPipelineCompress(reinterpret_cast<void*>(0x1)), cocclPipelineAllToAll(comm),
      cocclPipelineDecompress()};
  const size_t sizes[] = {64ULL << 20, 1ULL << 30, 8ULL << 30};
  const int depths[] = {1, 2, 4, 8};
  std::printf("bytes,requested_depth,effective_depth,temp_index,temp_role,"
              "logical_bytes,aligned_bytes,offset,workspace_kind,storage,"
              "registered_slice_bytes,registered_bytes,raw_bytes,"
              "total_bytes\n");
  for (size_t bytes : sizes) {
    for (int depth : depths) {
      cocclPipelineSpec spec = makeSpec(comm, bytes, false, stages);
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

  if (argc == 2 && std::strcmp(argv[1], "--csv") == 0) {
    dumpPlans(&comm);
    return 0;
  }

  checkDepth(&comm, 1);
  checkDepth(&comm, 2);
  checkDepth(&comm, 4);
  checkDepth(&comm, 8);
  checkTargetSlice(&comm);

  const cocclPipelineStage stages[] = {
      cocclPipelineCompress(reinterpret_cast<void*>(0x1)), cocclPipelineAllToAll(&comm),
      cocclPipelineDecompress()};
  cocclPipelineSpec inPlace = makeSpec(&comm, 16 << 20, true, stages);
  cocclPipelineContext context = {};
  EXPECT(cocclPreparePipeline(&inPlace, 8, &context) == ncclSuccess);
  EXPECT(context.depth == 1 && context.plan.tempCount == 2);

  cocclPipelineSpec indivisible = makeSpec(&comm, 16 << 20, false, stages);
  indivisible.rawChunkCount += 1;
  EXPECT(cocclPreparePipeline(&indivisible, 8, &context) == ncclSuccess);
  EXPECT(context.depth == 8);
  EXPECT(context.slices[7].elementCount ==
         context.slices[0].elementCount + 1);
  EXPECT(context.slices[0].bytes % kCocclPipelineSliceAlignment == 0);

  std::printf("coccl pipeline plan: PASS\n");
  return 0;
}
