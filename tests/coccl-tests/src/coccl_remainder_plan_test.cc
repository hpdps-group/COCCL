#include "coccl_size_query_stub.h"
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

cocclPipelineSpec makeAllToAllSpec(
    ncclComm_t comm, size_t rawChunkCount,
    const cocclPipelineStage* stages) {
  return {
      "alltoall",
      reinterpret_cast<const void*>(0x100000000ULL),
      reinterpret_cast<void*>(0x400000000ULL),
      rawChunkCount,
      4,
      ncclFloat32,
      comm,
      nullptr,
      stages,
      3,
      cocclPipelineInPlaceNone,
      cocclPipelineInputContiguous,
  };
}

void expectSlices(size_t count, int requestedDepth) {
  ncclComm comm = {};
  comm.nRanks = 4;
  const cocclPipelineStage stages[] = {
      cocclPipelineCompress(reinterpret_cast<void*>(0x1)),
      cocclPipelineAllToAll(&comm),
      cocclPipelineDecompress(),
  };
  cocclPipelineSpec spec = makeAllToAllSpec(&comm, count, stages);
  cocclConfigureSizeQueryStub(1, 4, 1, 1, false);

  cocclPipelineContext context = {};
  EXPECT(cocclPreparePipeline(&spec, requestedDepth, &context) ==
         ncclSuccess);
  int expectedDepth = count < (size_t)requestedDepth
      ? (int)count : requestedDepth;
  const size_t alignmentElements =
      kCocclPipelineSliceAlignment / sizeof(float);
  size_t regularCount = count / (size_t)expectedDepth;
  if (expectedDepth > 1) {
    regularCount = regularCount / alignmentElements * alignmentElements;
    if (regularCount == 0) {
      expectedDepth = 1;
      regularCount = count;
    }
  }
  EXPECT(context.depth == expectedDepth);

  size_t elements = 0;
  for (int slice = 0; slice < context.depth; ++slice) {
    const cocclPipelineSliceShape& shape = context.slices[slice];
    const size_t expectedCount = slice + 1 == context.depth
        ? count - elements : regularCount;
    EXPECT(shape.elementOffset == elements);
    EXPECT(shape.elementCount == expectedCount);
    EXPECT(shape.byteOffset == elements * sizeof(float));
    EXPECT(shape.bytes == expectedCount * sizeof(float));
    if (slice + 1 != context.depth) {
      EXPECT(shape.bytes % kCocclPipelineSliceAlignment == 0);
    }
    elements += shape.elementCount;
  }
  EXPECT(elements == count);
  EXPECT(context.maxSliceCount ==
         context.slices[context.depth - 1].elementCount);
  EXPECT(context.maxSliceBytes ==
         context.maxSliceCount * sizeof(float));
}

void checkRemainders() {
  expectSlices(1, 8);
  expectSlices(8, 8);
  expectSlices(512, 8);
  expectSlices(129, 2);
  for (size_t remainder = 1; remainder < 4; ++remainder) {
    expectSlices(4 * 64 + remainder, 4);
  }
  for (size_t remainder = 1; remainder < 8; ++remainder) {
    expectSlices(8 * 64 + remainder, 8);
  }
  expectSlices(4099, 2);
  expectSlices(4099, 4);
  expectSlices(4099, 8);
}

void checkTailCapacity() {
  ncclComm comm = {};
  comm.nRanks = 4;
  const cocclPipelineStage stages[] = {
      cocclPipelineCompress(reinterpret_cast<void*>(0x1)),
      cocclPipelineAllToAll(&comm),
      cocclPipelineDecompress(),
  };
  cocclPipelineSpec spec = makeAllToAllSpec(&comm, 257, stages);
  cocclConfigureSizeQueryStub(1, 4, 1, 1, false);

  cocclPipelineContext context = {};
  EXPECT(cocclPreparePipeline(&spec, 4, &context) == ncclSuccess);
  EXPECT(cocclCompressQueryObservation().calls == 2);
  EXPECT(cocclCompressQueryObservation().elements == 260);
  EXPECT(context.plan.stageOutputCapacityBytes[0] == 260);
  EXPECT(context.plan.stageOutputCapacityBytes[1] == 260);
  EXPECT(context.plan.stageOutputCapacityBytes[2] == 1040);
  for (int slice = 0; slice < 3; ++slice) {
    EXPECT(context.sliceStageOutputBytes[slice][0] == 256);
    EXPECT(context.sliceStageOutputBytes[slice][1] == 256);
    EXPECT(context.sliceStageOutputBytes[slice][2] == 1024);
  }
  EXPECT(context.sliceStageOutputBytes[3][0] == 260);
  EXPECT(context.sliceStageOutputBytes[3][1] == 260);
  EXPECT(context.sliceStageOutputBytes[3][2] == 1040);
  EXPECT(context.plan.temps[0].payloadBytes == 1040);
  EXPECT(context.plan.temps[1].payloadBytes == 260);
  EXPECT(context.plan.temps[2].payloadBytes == 260);
  EXPECT(context.plan.temps[3].payloadBytes == 1040);
  EXPECT(context.plan.totalBytes == 7168);
}

void checkDivisiblePlan() {
  ncclComm comm = {};
  comm.nRanks = 4;
  const cocclPipelineStage stages[] = {
      cocclPipelineCompress(reinterpret_cast<void*>(0x1)),
      cocclPipelineAllToAll(&comm),
      cocclPipelineDecompress(),
  };
  cocclPipelineSpec spec = makeAllToAllSpec(&comm, 256, stages);
  cocclConfigureSizeQueryStub(1, 4, 1, 1, false);

  cocclPipelineContext context = {};
  EXPECT(cocclPreparePipeline(&spec, 4, &context) == ncclSuccess);
  EXPECT(cocclCompressQueryObservation().calls == 1);
  EXPECT(context.plan.stageOutputCapacityBytes[0] == 256);
  EXPECT(context.plan.stageOutputCapacityBytes[1] == 256);
  EXPECT(context.plan.stageOutputCapacityBytes[2] == 1024);
  EXPECT(context.plan.totalBytes == 5120);
}

void checkFramedTail() {
  ncclComm comm = {};
  comm.nRanks = 4;
  const cocclPipelineStage stages[] = {
      cocclPipelineCompress(reinterpret_cast<void*>(0x1)),
      cocclPipelineAllToAll(&comm),
      cocclPipelineDecompress(),
  };
  cocclPipelineSpec spec = makeAllToAllSpec(&comm, 257, stages);
  cocclConfigureSizeQueryStub(1, 1, 1, 1, false);
  cocclConfigureFramedSizeQueryStub(true);

  cocclPipelineContext context = {};
  EXPECT(cocclPreparePipeline(&spec, 4, &context) == ncclSuccess);
  const int compressTemp = context.plan.stageOutputTemp[0];
  const int allToAllTemp = context.plan.stageOutputTemp[1];
  EXPECT(compressTemp >= 0 && allToAllTemp >= 0);
  for (int tempIndex : {compressTemp, allToAllTemp}) {
    const cocclPipelineTempPlan& temp = context.plan.temps[tempIndex];
    EXPECT(temp.payloadBytes == 1040);
    EXPECT(temp.frameStrideBytes == 260);
    EXPECT(temp.frameMetadataBytes ==
           4 * sizeof(cocclCompressorFrameMetadata));
  }
  EXPECT(context.sliceStageOutputBytes[0][0] == 1024);
  EXPECT(context.sliceStageOutputBytes[3][0] == 1040);
  EXPECT(context.sliceStageFrameStrideBytes[0][0] == 256);
  EXPECT(context.sliceStageFrameStrideBytes[0][1] == 256);
  EXPECT(context.sliceStageFrameStrideBytes[3][0] == 260);
  EXPECT(context.sliceStageFrameStrideBytes[3][1] == 260);
}

void checkMixedHierarchicalTail() {
  ncclComm owner = {};
  owner.nRanks = 8;
  owner.localRanks = 4;
  ncclComm intra = {};
  intra.nRanks = 4;
  ncclComm inter = {};
  inter.nRanks = 2;
  const cocclPipelineStage stages[] = {
      cocclPipelineReduceScatter(&intra),
      cocclPipelineCompress(reinterpret_cast<void*>(0x1)),
      cocclPipelineAllToAll(&inter),
      cocclPipelineDecompressReduce(2),
  };
  const cocclPipelineSpec spec = {
      "inter-only-reducescatter",
      reinterpret_cast<const void*>(0x100000000ULL),
      reinterpret_cast<void*>(0x400000000ULL),
      257,
      8,
      ncclFloat32,
      &owner,
      nullptr,
      stages,
      4,
      cocclPipelineInPlaceNone,
      cocclPipelineInputHierarchicalSwizzle,
  };
  cocclConfigureSizeQueryStub(1, 4, 1, 1, false);

  cocclPipelineContext context = {};
  EXPECT(cocclPreparePipeline(&spec, 4, &context) == ncclSuccess);
  EXPECT(context.plan.finalChunks == 1);
  EXPECT(context.sliceStageOutputBytes[0][0] == 512);
  EXPECT(context.sliceStageOutputBytes[0][1] == 128);
  EXPECT(context.sliceStageOutputBytes[0][2] == 128);
  EXPECT(context.sliceStageOutputBytes[0][3] == 256);
  EXPECT(context.sliceStageOutputBytes[3][0] == 520);
  EXPECT(context.sliceStageOutputBytes[3][1] == 130);
  EXPECT(context.sliceStageOutputBytes[3][2] == 130);
  EXPECT(context.sliceStageOutputBytes[3][3] == 260);
  EXPECT(context.plan.stageOutputCapacityBytes[0] == 520);
  EXPECT(context.plan.stageOutputCapacityBytes[1] == 130);
  EXPECT(context.plan.stageOutputCapacityBytes[2] == 130);
  EXPECT(context.plan.stageOutputCapacityBytes[3] == 260);
}

}  // namespace

thread_local int ncclDebugNoWarn = 0;
void ncclDebugLog(ncclDebugLogLevel, unsigned long, const char*, int,
                  const char*, ...) {}

int main() {
  checkRemainders();
  checkTailCapacity();
  checkDivisiblePlan();
  checkFramedTail();
  checkMixedHierarchicalTail();
  std::printf("coccl remainder plan: PASS\n");
  return 0;
}
