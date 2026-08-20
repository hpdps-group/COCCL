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
  cocclM11ConfigureSizeQueryStub(1, 4, 1, 1, false);

  cocclPipelineContext context = {};
  EXPECT(cocclPreparePipeline(&spec, requestedDepth, &context) ==
         ncclSuccess);
  const int expectedDepth = count < (size_t)requestedDepth
      ? (int)count : requestedDepth;
  EXPECT(context.depth == expectedDepth);

  const size_t quotient = count / (size_t)expectedDepth;
  const size_t remainder = count % (size_t)expectedDepth;
  size_t elements = 0;
  for (int slice = 0; slice < context.depth; ++slice) {
    const cocclPipelineSliceShape& shape = context.slices[slice];
    const size_t expectedCount = quotient +
        (slice + 1 == context.depth ? remainder : 0);
    EXPECT(shape.elementOffset == elements);
    EXPECT(shape.elementCount == expectedCount);
    EXPECT(shape.byteOffset == elements * sizeof(float));
    EXPECT(shape.bytes == expectedCount * sizeof(float));
    elements += shape.elementCount;
  }
  EXPECT(elements == count);
  EXPECT(context.maxSliceCount == quotient + remainder);
  EXPECT(context.maxSliceBytes ==
         context.maxSliceCount * sizeof(float));
}

void checkRemainders() {
  expectSlices(1, 8);
  expectSlices(8, 8);
  expectSlices(32, 8);
  expectSlices(17, 2);
  for (size_t remainder = 1; remainder < 4; ++remainder) {
    expectSlices(4 * 16 + remainder, 4);
  }
  for (size_t remainder = 1; remainder < 8; ++remainder) {
    expectSlices(8 * 16 + remainder, 8);
  }
}

void checkTailCapacity() {
  ncclComm comm = {};
  comm.nRanks = 4;
  const cocclPipelineStage stages[] = {
      cocclPipelineCompress(reinterpret_cast<void*>(0x1)),
      cocclPipelineAllToAll(&comm),
      cocclPipelineDecompress(),
  };
  cocclPipelineSpec spec = makeAllToAllSpec(&comm, 17, stages);
  cocclM11ConfigureSizeQueryStub(1, 4, 1, 1, false);

  cocclPipelineContext context = {};
  EXPECT(cocclPreparePipeline(&spec, 4, &context) == ncclSuccess);
  EXPECT(cocclM11CompressQueryObservation().calls == 2);
  EXPECT(cocclM11CompressQueryObservation().elements == 20);
  EXPECT(context.plan.stageOutputCapacityBytes[0] == 20);
  EXPECT(context.plan.stageOutputCapacityBytes[1] == 20);
  EXPECT(context.plan.stageOutputCapacityBytes[2] == 80);
  for (int slice = 0; slice < 3; ++slice) {
    EXPECT(context.sliceStageOutputBytes[slice][0] == 16);
    EXPECT(context.sliceStageOutputBytes[slice][1] == 16);
    EXPECT(context.sliceStageOutputBytes[slice][2] == 64);
  }
  EXPECT(context.sliceStageOutputBytes[3][0] == 20);
  EXPECT(context.sliceStageOutputBytes[3][1] == 20);
  EXPECT(context.sliceStageOutputBytes[3][2] == 80);
  EXPECT(context.plan.temps[0].payloadBytes == 80);
  EXPECT(context.plan.temps[1].payloadBytes == 20);
  EXPECT(context.plan.temps[2].payloadBytes == 20);
  EXPECT(context.plan.temps[3].payloadBytes == 80);
  EXPECT(context.plan.totalBytes == 2048);
}

void checkDivisiblePlan() {
  ncclComm comm = {};
  comm.nRanks = 4;
  const cocclPipelineStage stages[] = {
      cocclPipelineCompress(reinterpret_cast<void*>(0x1)),
      cocclPipelineAllToAll(&comm),
      cocclPipelineDecompress(),
  };
  cocclPipelineSpec spec = makeAllToAllSpec(&comm, 16, stages);
  cocclM11ConfigureSizeQueryStub(1, 4, 1, 1, false);

  cocclPipelineContext context = {};
  EXPECT(cocclPreparePipeline(&spec, 4, &context) == ncclSuccess);
  EXPECT(cocclM11CompressQueryObservation().calls == 1);
  EXPECT(context.plan.stageOutputCapacityBytes[0] == 16);
  EXPECT(context.plan.stageOutputCapacityBytes[1] == 16);
  EXPECT(context.plan.stageOutputCapacityBytes[2] == 64);
  EXPECT(context.plan.totalBytes == 2048);
}

void checkFramedTail() {
  ncclComm comm = {};
  comm.nRanks = 4;
  const cocclPipelineStage stages[] = {
      cocclPipelineCompress(reinterpret_cast<void*>(0x1)),
      cocclPipelineAllToAll(&comm),
      cocclPipelineDecompress(),
  };
  cocclPipelineSpec spec = makeAllToAllSpec(&comm, 17, stages);
  cocclM11ConfigureSizeQueryStub(1, 1, 1, 1, false);
  cocclM14ConfigureFramedSizeQueryStub(true);

  cocclPipelineContext context = {};
  EXPECT(cocclPreparePipeline(&spec, 4, &context) == ncclSuccess);
  const int compressTemp = context.plan.stageOutputTemp[0];
  const int allToAllTemp = context.plan.stageOutputTemp[1];
  EXPECT(compressTemp >= 0 && allToAllTemp >= 0);
  for (int tempIndex : {compressTemp, allToAllTemp}) {
    const cocclPipelineTempPlan& temp = context.plan.temps[tempIndex];
    EXPECT(temp.payloadBytes == 80);
    EXPECT(temp.frameStrideBytes == 20);
    EXPECT(temp.frameMetadataBytes ==
           4 * sizeof(cocclCompressorFrameMetadata));
  }
  EXPECT(context.sliceStageOutputBytes[0][0] == 64);
  EXPECT(context.sliceStageOutputBytes[3][0] == 80);
  EXPECT(context.sliceStageFrameStrideBytes[0][0] == 16);
  EXPECT(context.sliceStageFrameStrideBytes[0][1] == 16);
  EXPECT(context.sliceStageFrameStrideBytes[3][0] == 20);
  EXPECT(context.sliceStageFrameStrideBytes[3][1] == 20);
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
      17,
      8,
      ncclFloat32,
      &owner,
      nullptr,
      stages,
      4,
      cocclPipelineInPlaceNone,
      cocclPipelineInputHierarchicalSwizzle,
  };
  cocclM11ConfigureSizeQueryStub(1, 4, 1, 1, false);

  cocclPipelineContext context = {};
  EXPECT(cocclPreparePipeline(&spec, 4, &context) == ncclSuccess);
  EXPECT(context.plan.finalChunks == 1);
  EXPECT(context.sliceStageOutputBytes[0][0] == 32);
  EXPECT(context.sliceStageOutputBytes[0][1] == 8);
  EXPECT(context.sliceStageOutputBytes[0][2] == 8);
  EXPECT(context.sliceStageOutputBytes[0][3] == 16);
  EXPECT(context.sliceStageOutputBytes[3][0] == 40);
  EXPECT(context.sliceStageOutputBytes[3][1] == 10);
  EXPECT(context.sliceStageOutputBytes[3][2] == 10);
  EXPECT(context.sliceStageOutputBytes[3][3] == 20);
  EXPECT(context.plan.stageOutputCapacityBytes[0] == 40);
  EXPECT(context.plan.stageOutputCapacityBytes[1] == 10);
  EXPECT(context.plan.stageOutputCapacityBytes[2] == 10);
  EXPECT(context.plan.stageOutputCapacityBytes[3] == 20);
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
  std::printf("coccl M20 remainder plan: PASS\n");
  return 0;
}
