#include "coccl_m11_size_query_stub.h"
#include "coccl_pipeline_internal.h"

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

cocclPipelineSpec hierarchicalSpec(
    ncclComm_t comm, const cocclPipelineStage* stages) {
  return {
      "m15-hierarchical",
      reinterpret_cast<const void*>(0x100000000ULL),
      reinterpret_cast<void*>(0x400000000ULL),
      4096,
      (size_t)comm->nRanks,
      ncclFloat32,
      comm,
      nullptr,
      stages,
      2,
      cocclPipelineInPlaceNone,
      cocclPipelineInputHierarchicalSwizzle,
  };
}

void checkNonFused(ncclComm_t comm, const cocclPipelineStage* stages) {
  cocclM15ConfigureFusedSwizzleStub(false);
  const cocclPipelineSpec spec = hierarchicalSpec(comm, stages);
  for (int depth : {1, 2, 4, 8}) {
    cocclPipelineContext context = {};
    EXPECT(cocclPreparePipeline(&spec, depth, &context) == ncclSuccess);
    EXPECT(context.plan.inputStagingTemp >= 0);
    EXPECT(context.stageContext.inputLayout ==
           cocclPipelineInputHierarchicalSwizzle);
    EXPECT(context.stageContext.nNodes == 2);
    EXPECT(context.stageContext.ranksPerNode == 4);
  }
}

void checkFused(ncclComm_t comm, const cocclPipelineStage* stages) {
  cocclM15ConfigureFusedSwizzleStub(true);
  const cocclPipelineSpec spec = hierarchicalSpec(comm, stages);
  for (int depth : {1, 2, 4, 8}) {
    cocclPipelineContext context = {};
    EXPECT(cocclPreparePipeline(&spec, depth, &context) == ncclSuccess);
    EXPECT(context.plan.inputStagingTemp == (depth == 1 ? -1 : 0));
    EXPECT(context.stageContext.inputLayout == cocclPipelineInputContiguous);
  }
}

}  // namespace

thread_local int ncclDebugNoWarn = 0;
void ncclDebugLog(ncclDebugLogLevel, unsigned long, const char*, int,
                  const char*, ...) {}

int main() {
  cocclM11ConfigureSizeQueryStub(1, 4, 1, 4, false);
  ncclComm owner = {};
  owner.nRanks = 8;
  owner.localRanks = 4;
  const cocclPipelineStage stages[] = {
      cocclPipelineCompress(reinterpret_cast<void*>(0x1)), cocclPipelineDecompress()};
  checkNonFused(&owner, stages);
  checkFused(&owner, stages);
  std::printf("COCCL M15 swizzle path selection: PASS\n");
  return 0;
}
