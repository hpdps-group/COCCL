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

cocclPipelineSpec makeSpec(const char* name, ncclComm_t comm,
                           const cocclPipelineStage* stages,
                           int stageCount) {
  return {
      name,
      reinterpret_cast<const void*>(0x100000000ULL),
      reinterpret_cast<void*>(0x400000000ULL),
      4ULL << 20,
      (size_t)comm->nRanks,
      ncclFloat32,
      cocclDefaultPolicy(cocclOperation::AllReduce),
      comm,
      nullptr,
      stages,
      stageCount,
  };
}

void expectChunks(const cocclPipelineStage* stages, int stageCount,
                  size_t inputChunks, const size_t* expected) {
  size_t chunks = inputChunks;
  for (int stage = 0; stage < stageCount; ++stage) {
    EXPECT(cocclPipelineStageOutputChunks(
               stages[stage], chunks, &chunks) == ncclSuccess);
    EXPECT(chunks == expected[stage]);
  }
}

void checkOneShot(ncclComm_t comm) {
  const cocclPipelineStage stages[] = {
      cocclPipelineCompress(), cocclPipelineAllGather(comm),
      cocclPipelineDecompressReduce((size_t)comm->nRanks)};
  const size_t chunks[] = {4, 16, 4};
  expectChunks(stages, 3, 4, chunks);

  cocclPipelineSpec spec = makeSpec("allreduce-oneshot", comm, stages, 3);
  cocclPipelineContext serial = {};
  EXPECT(cocclPreparePipeline(&spec, 1, &serial) == ncclSuccess);
  EXPECT(serial.plan.tempCount == 2 && serial.plan.finalChunks == 4);

  cocclPipelineContext overlap = {};
  EXPECT(cocclPreparePipeline(&spec, 4, &overlap) == ncclSuccess);
  EXPECT(overlap.plan.tempCount == 4 && overlap.plan.finalChunks == 4);
  EXPECT(overlap.plan.outputStagingTemp >= 0);
}

void checkTwoShot(ncclComm_t comm) {
  const cocclPipelineStage stages[] = {
      cocclPipelineCompress(), cocclPipelineAllToAll(comm),
      cocclPipelineDecompReduceComp((size_t)comm->nRanks),
      cocclPipelineAllGather(comm), cocclPipelineDecompress()};
  const size_t chunks[] = {4, 4, 1, 4, 4};
  expectChunks(stages, 5, 4, chunks);

  cocclPipelineSpec spec = makeSpec("allreduce-twoshot", comm, stages, 5);
  cocclPipelineContext context = {};
  EXPECT(cocclPreparePipeline(&spec, 4, &context) == ncclSuccess);
  EXPECT(context.plan.tempCount == 6 && context.plan.finalChunks == 4);
}

void checkTripleShot(ncclComm_t comm, ncclComm_t intra,
                     ncclComm_t inter) {
  const cocclPipelineStage stages[] = {
      cocclPipelineCompress(), cocclPipelineAllToAll(intra),
      cocclPipelineDecompReduceComp(2), cocclPipelineAllToAll(inter),
      cocclPipelineDecompReduceComp(2), cocclPipelineAllGather(comm),
      cocclPipelineDecompress()};
  const size_t chunks[] = {4, 4, 2, 2, 1, 4, 4};
  expectChunks(stages, 7, 4, chunks);

  cocclPipelineSpec spec = makeSpec("allreduce-tripleshot", comm, stages, 7);
  spec.compressorPolicy =
      cocclHierarchicalPolicy(cocclOperation::AllReduce);
  cocclPipelineContext context = {};
  EXPECT(cocclPreparePipeline(&spec, 4, &context) == ncclSuccess);
  EXPECT(context.plan.tempCount == 8 && context.plan.finalChunks == 4);
}

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

void dumpPlan(const char* algorithm, ncclComm_t comm,
              const cocclPipelineStage* stages, int stageCount,
              size_t bytes, int depth) {
  cocclPipelineSpec spec = makeSpec(algorithm, comm, stages, stageCount);
  spec.rawChunkCount = bytes / ((size_t)comm->nRanks * sizeof(float));
  cocclPipelineContext context = {};
  EXPECT(cocclPreparePipeline(&spec, depth, &context) == ncclSuccess);
  for (int temp = 0; temp < context.plan.tempCount; ++temp) {
    const cocclPipelineTempPlan& item = context.plan.temps[temp];
    std::printf("%s,%zu,%d,%d,%d,%s,%zu,%zu,%zu,%zu,%zu\n",
                algorithm, bytes, depth, context.depth, temp,
                roleName(item.role), item.logicalBytes, item.alignedBytes,
                item.offset, context.plan.sliceWorkspaceBytes,
                context.plan.workspaceBytes);
  }
}

void dumpPlans(ncclComm_t comm) {
  const cocclPipelineStage oneShot[] = {
      cocclPipelineCompress(), cocclPipelineAllGather(comm),
      cocclPipelineDecompressReduce((size_t)comm->nRanks)};
  const cocclPipelineStage twoShot[] = {
      cocclPipelineCompress(), cocclPipelineAllToAll(comm),
      cocclPipelineDecompReduceComp((size_t)comm->nRanks),
      cocclPipelineAllGather(comm), cocclPipelineDecompress()};
  const size_t sizes[] = {64ULL << 20, 1ULL << 30, 8ULL << 30};
  const int depths[] = {1, 2, 4, 8};
  std::printf("algorithm,bytes,requested_depth,effective_depth,temp_index,"
              "temp_role,logical_bytes,aligned_bytes,offset,"
              "slice_workspace_bytes,workspace_bytes\n");
  dumpPlan("oneshot", comm, oneShot, 3, 32ULL << 20, 1);
  for (size_t bytes : sizes) {
    for (int depth : depths) {
      dumpPlan("twoshot", comm, twoShot, 5, bytes, depth);
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
  ncclComm intra = {};
  intra.nRanks = 2;
  ncclComm inter = {};
  inter.nRanks = 2;

  checkOneShot(&owner);
  checkTwoShot(&owner);
  checkTripleShot(&owner, &intra, &inter);
  std::printf("coccl M8 allreduce plan: PASS\n");
  return 0;
}
