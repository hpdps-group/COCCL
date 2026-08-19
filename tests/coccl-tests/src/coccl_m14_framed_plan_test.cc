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

cocclPipelineSpec makeSpec(
    const char* name, ncclComm_t comm, size_t inputChunks,
    const cocclPipelineStage* stages, int stageCount) {
  return {
      name,
      reinterpret_cast<const void*>(0x100000000ULL),
      reinterpret_cast<void*>(0x400000000ULL),
      256,
      inputChunks,
      ncclFloat32,
      cocclDefaultPolicy(cocclOperation::AllToAll),
      comm,
      nullptr,
      stages,
      stageCount,
      cocclPipelineInPlaceNone,
  };
}

void expectFramedTemp(const cocclPipelineTempPlan& temp,
                      size_t payloadBytes, size_t frames,
                      size_t strideBytes) {
  EXPECT(temp.payloadBytes == payloadBytes);
  EXPECT(temp.frameMetadataOffset == payloadBytes);
  EXPECT(temp.frameMetadataBytes ==
         frames * sizeof(cocclCompressorFrameMetadata));
  EXPECT(temp.frameStrideBytes == strideBytes);
  EXPECT(temp.logicalBytes ==
         payloadBytes + frames * sizeof(cocclCompressorFrameMetadata));
  EXPECT(temp.alignedBytes % kCocclPipelineAlignment == 0);
  EXPECT(temp.alignedBytes >= temp.logicalBytes);
}

void testAllToAll(ncclComm_t comm) {
  const cocclPipelineStage stages[] = {
      cocclPipelineCompress(), cocclPipelineAllToAll(comm),
      cocclPipelineDecompress()};
  const cocclPipelineSpec spec = makeSpec(
      "framed-alltoall", comm, 4, stages, 3);
  cocclPipelineContext context = {};
  EXPECT(cocclPreparePipeline(&spec, 4, &context) == ncclSuccess);
  EXPECT(context.plan.tempCount == 4);
  EXPECT(context.plan.workspaceKind == cocclPipelineWorkspaceUnified);
  expectFramedTemp(context.plan.temps[1], 1024, 4, 256);
  expectFramedTemp(context.plan.temps[2], 1024, 4, 256);
  EXPECT(context.plan.stageOutputCapacityBytes[0] == 1024);
  EXPECT(context.plan.stageOutputCapacityBytes[1] == 1024);
}

void testAllGather(ncclComm_t comm) {
  const cocclPipelineStage stages[] = {
      cocclPipelineCompress(), cocclPipelineAllGather(comm),
      cocclPipelineDecompress()};
  cocclPipelineSpec spec = makeSpec(
      "framed-allgather", comm, 1, stages, 3);
  spec.compressorPolicy = cocclDefaultPolicy(cocclOperation::AllGather);
  cocclPipelineContext context = {};
  EXPECT(cocclPreparePipeline(&spec, 4, &context) == ncclSuccess);
  EXPECT(context.plan.tempCount == 3);
  expectFramedTemp(context.plan.temps[0], 256, 1, 256);
  expectFramedTemp(context.plan.temps[1], 1024, 4, 256);
  EXPECT(context.plan.finalChunks == 4);
}

void testHierarchicalReduction(ncclComm_t owner, ncclComm_t intra,
                               ncclComm_t inter) {
  const cocclPipelineStage stages[] = {
      cocclPipelineCompress(),
      cocclPipelineAllToAll(intra),
      cocclPipelineDecompReduceComp(2),
      cocclPipelineAllToAll(inter),
      cocclPipelineDecompressReduce(2),
  };
  cocclPipelineSpec spec = makeSpec(
      "framed-reducescatter-twoshot", owner, 4, stages, 5);
  spec.compressorPolicy =
      cocclHierarchicalPolicy(cocclOperation::ReduceScatter);
  spec.inputLayout = cocclPipelineInputHierarchicalSwizzle;
  cocclPipelineContext context = {};
  EXPECT(cocclPreparePipeline(&spec, 4, &context) == ncclSuccess);
  const int drcTemp = context.plan.stageOutputTemp[2];
  EXPECT(drcTemp >= 0);
  expectFramedTemp(context.plan.temps[drcTemp], 512, 2, 256);
  EXPECT(context.plan.stageOutputCapacityBytes[2] == 512);
  EXPECT(context.plan.finalChunks == 1);
}

}  // namespace

thread_local int ncclDebugNoWarn = 0;
void ncclDebugLog(ncclDebugLogLevel, unsigned long, const char*, int,
                  const char*, ...) {}

int main() {
  cocclM11ConfigureSizeQueryStub(1, 1, 1, 1, false);
  cocclM14ConfigureFramedSizeQueryStub(true);

  ncclComm owner = {};
  owner.nRanks = 4;
  owner.localRanks = 2;
  ncclComm intra = {};
  intra.nRanks = 2;
  ncclComm inter = {};
  inter.nRanks = 2;

  testAllToAll(&owner);
  testAllGather(&owner);
  testHierarchicalReduction(&owner, &intra, &inter);
  std::printf("COCCL M14 framed pipeline plan: PASS\n");
  return 0;
}
