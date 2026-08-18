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

void expectUserBuffers(const void* input, size_t inputChunks, void* output,
                       size_t outputChunks, size_t rawChunkBytes, int rank,
                       cocclPipelineInPlaceLayout layout,
                       ncclResult_t expectedResult, bool expectedSerial) {
  bool requireSerial = false;
  const ncclResult_t result = cocclPipelineUserBuffersRequireSerial(
      input, inputChunks, output, outputChunks, rawChunkBytes, rank, layout,
      &requireSerial);
  EXPECT(result == expectedResult);
  if (result == ncclSuccess) EXPECT(requireSerial == expectedSerial);
}

void checkPointerLayouts() {
  constexpr uintptr_t base = 0x100000;
  constexpr size_t chunk = 1024;

  expectUserBuffers(reinterpret_cast<void*>(base), 4,
                    reinterpret_cast<void*>(base + 8 * chunk), 4,
                    chunk, 0, cocclPipelineInPlaceNone, ncclSuccess, false);
  expectUserBuffers(reinterpret_cast<void*>(base), 4,
                    reinterpret_cast<void*>(base), 4, chunk, 0,
                    cocclPipelineInPlaceSameBuffer, ncclSuccess, false);
  for (int rank : {0, 3}) {
    expectUserBuffers(reinterpret_cast<void*>(
                          base + (size_t)rank * chunk), 1,
                      reinterpret_cast<void*>(base), 4, chunk, rank,
                      cocclPipelineInPlaceInputRankChunk, ncclSuccess, false);
    expectUserBuffers(reinterpret_cast<void*>(base), 4,
                      reinterpret_cast<void*>(
                          base + (size_t)rank * chunk), 1,
                      chunk, rank, cocclPipelineInPlaceOutputRankChunk,
                      ncclSuccess, false);
  }
  expectUserBuffers(reinterpret_cast<void*>(base + 3 * chunk + 1), 1,
                    reinterpret_cast<void*>(base), 4, chunk, 3,
                    cocclPipelineInPlaceInputRankChunk, ncclSuccess, true);
  expectUserBuffers(reinterpret_cast<void*>(base + 3 * chunk), 1,
                    reinterpret_cast<void*>(base), 4, chunk, 2,
                    cocclPipelineInPlaceInputRankChunk, ncclSuccess, true);
  expectUserBuffers(reinterpret_cast<void*>(base + chunk / 2), 4,
                    reinterpret_cast<void*>(base), 4, chunk, 0,
                    cocclPipelineInPlaceSameBuffer, ncclSuccess, true);
  expectUserBuffers(reinterpret_cast<void*>(base), 4,
                    reinterpret_cast<void*>(base), 4, chunk, 0,
                    cocclPipelineInPlaceNone, ncclSuccess, true);
  expectUserBuffers(reinterpret_cast<void*>(base), 4,
                    reinterpret_cast<void*>(base), 1, chunk, 0,
                    cocclPipelineInPlaceSameBuffer, ncclSuccess, true);
  expectUserBuffers(reinterpret_cast<void*>(base), 2,
                    reinterpret_cast<void*>(base), 1, SIZE_MAX, 0,
                    cocclPipelineInPlaceSameBuffer, ncclInvalidArgument,
                    false);
  expectUserBuffers(reinterpret_cast<void*>(UINTPTR_MAX - chunk / 2), 1,
                    reinterpret_cast<void*>(base), 1, chunk, 0,
                    cocclPipelineInPlaceNone, ncclInvalidArgument, false);
  expectUserBuffers(reinterpret_cast<void*>(base), 1,
                    reinterpret_cast<void*>(base), 4, chunk, -1,
                    cocclPipelineInPlaceInputRankChunk, ncclInvalidArgument,
                    false);
}

bool samePlan(const cocclPipelinePlan& left,
              const cocclPipelinePlan& right) {
  if (left.workspaceKind != right.workspaceKind ||
      left.tempCount != right.tempCount ||
      left.inputStagingTemp != right.inputStagingTemp ||
      left.outputStagingTemp != right.outputStagingTemp ||
      left.finalChunks != right.finalChunks ||
      left.registeredSliceBytes != right.registeredSliceBytes ||
      left.registeredBytes != right.registeredBytes ||
      left.rawBytes != right.rawBytes ||
      left.totalBytes != right.totalBytes) {
    return false;
  }
  for (int stage = 0; stage < kCocclPipelineExplicitStages; ++stage) {
    if (left.stageOutputTemp[stage] != right.stageOutputTemp[stage] ||
        left.stageOutputCapacityBytes[stage] !=
            right.stageOutputCapacityBytes[stage]) {
      return false;
    }
  }
  for (int temp = 0; temp < left.tempCount; ++temp) {
    const cocclPipelineTempPlan& a = left.temps[temp];
    const cocclPipelineTempPlan& b = right.temps[temp];
    if (a.role != b.role || a.offset != b.offset ||
        a.logicalBytes != b.logicalBytes ||
        a.alignedBytes != b.alignedBytes ||
        a.storage != b.storage) {
      return false;
    }
  }
  return true;
}

void expectSamePlan(cocclPipelineSpec outOfPlace,
                    cocclPipelineSpec inPlace, int requestedDepth) {
  cocclPipelineContext outContext = {};
  cocclPipelineContext inContext = {};
  EXPECT(cocclPreparePipeline(
             &outOfPlace, requestedDepth, &outContext) == ncclSuccess);
  EXPECT(cocclPreparePipeline(
             &inPlace, requestedDepth, &inContext) == ncclSuccess);
  EXPECT(outContext.depth == requestedDepth);
  EXPECT(inContext.depth == requestedDepth);
  EXPECT(samePlan(outContext.plan, inContext.plan));
}

void checkPlannerLayouts() {
  ncclComm comm = {};
  comm.nRanks = 4;
  comm.rank = 2;
  constexpr size_t rawChunkCount = 4096;
  constexpr size_t rawChunkBytes = rawChunkCount * sizeof(float);
  constexpr uintptr_t inputBase = 0x100000000ULL;
  constexpr uintptr_t outputBase = 0x400000000ULL;
  cocclM11ConfigureSizeQueryStub(1, 4, 1, 1, false);

  const cocclPipelineStage allGatherStages[] = {
      cocclPipelineCompress(), cocclPipelineAllGather(&comm),
      cocclPipelineDecompress()};
  cocclPipelineSpec allGather = {
      "allgather", reinterpret_cast<void*>(inputBase),
      reinterpret_cast<void*>(outputBase), rawChunkCount, 1, ncclFloat32,
      cocclDefaultPolicy(cocclOperation::AllGather), &comm, nullptr,
      allGatherStages, 3, cocclPipelineInPlaceInputRankChunk};
  cocclPipelineSpec allGatherInPlace = allGather;
  allGatherInPlace.output = reinterpret_cast<void*>(inputBase);
  allGatherInPlace.input = reinterpret_cast<void*>(
      inputBase + (size_t)comm.rank * rawChunkBytes);
  expectSamePlan(allGather, allGatherInPlace, 8);

  const cocclPipelineStage reduceScatterStages[] = {
      cocclPipelineCompress(), cocclPipelineAllToAll(&comm),
      cocclPipelineDecompressReduce(4)};
  cocclPipelineSpec reduceScatter = {
      "reducescatter-oneshot", reinterpret_cast<void*>(inputBase),
      reinterpret_cast<void*>(outputBase), rawChunkCount, 4, ncclFloat32,
      cocclDefaultPolicy(cocclOperation::ReduceScatter), &comm, nullptr,
      reduceScatterStages, 3, cocclPipelineInPlaceOutputRankChunk};
  cocclPipelineSpec reduceScatterInPlace = reduceScatter;
  reduceScatterInPlace.output = reinterpret_cast<void*>(
      inputBase + (size_t)comm.rank * rawChunkBytes);
  expectSamePlan(reduceScatter, reduceScatterInPlace, 8);

  const cocclPipelineStage allReduceStages[] = {
      cocclPipelineCompress(), cocclPipelineAllToAll(&comm),
      cocclPipelineDecompReduceComp(4), cocclPipelineAllGather(&comm),
      cocclPipelineDecompress()};
  cocclPipelineSpec allReduce = {
      "allreduce-twoshot", reinterpret_cast<void*>(inputBase),
      reinterpret_cast<void*>(outputBase), rawChunkCount, 4, ncclFloat32,
      cocclDefaultPolicy(cocclOperation::AllReduce), &comm, nullptr,
      allReduceStages, 5, cocclPipelineInPlaceSameBuffer};
  cocclPipelineSpec allReduceInPlace = allReduce;
  allReduceInPlace.output = reinterpret_cast<void*>(inputBase);
  expectSamePlan(allReduce, allReduceInPlace, 8);

  ncclComm intra = {};
  intra.nRanks = 2;
  ncclComm inter = {};
  inter.nRanks = 2;
  const cocclPipelineStage reduceScatterTwoShotStages[] = {
      cocclPipelineCompress(), cocclPipelineAllToAll(&intra),
      cocclPipelineDecompReduceComp(2), cocclPipelineAllToAll(&inter),
      cocclPipelineDecompressReduce(2)};
  cocclPipelineSpec reduceScatterTwoShot = {
      "reducescatter-twoshot", reinterpret_cast<void*>(inputBase),
      reinterpret_cast<void*>(outputBase), rawChunkCount, 4, ncclFloat32,
      cocclHierarchicalPolicy(cocclOperation::ReduceScatter), &comm, nullptr,
      reduceScatterTwoShotStages, 5,
      cocclPipelineInPlaceOutputRankChunk};
  cocclPipelineSpec reduceScatterTwoShotInPlace = reduceScatterTwoShot;
  reduceScatterTwoShotInPlace.output = reinterpret_cast<void*>(
      inputBase + (size_t)comm.rank * rawChunkBytes);
  expectSamePlan(reduceScatterTwoShot, reduceScatterTwoShotInPlace, 8);

  const cocclPipelineStage allReduceTripleShotStages[] = {
      cocclPipelineCompress(), cocclPipelineAllToAll(&intra),
      cocclPipelineDecompReduceComp(2), cocclPipelineAllToAll(&inter),
      cocclPipelineDecompReduceComp(2), cocclPipelineAllGather(&comm),
      cocclPipelineDecompress()};
  cocclPipelineSpec allReduceTripleShot = {
      "allreduce-tripleshot", reinterpret_cast<void*>(inputBase),
      reinterpret_cast<void*>(outputBase), rawChunkCount, 4, ncclFloat32,
      cocclHierarchicalPolicy(cocclOperation::AllReduce), &comm, nullptr,
      allReduceTripleShotStages, 7, cocclPipelineInPlaceSameBuffer};
  cocclPipelineSpec allReduceTripleShotInPlace = allReduceTripleShot;
  allReduceTripleShotInPlace.output = reinterpret_cast<void*>(inputBase);
  expectSamePlan(allReduceTripleShot, allReduceTripleShotInPlace, 8);

  cocclPipelineSpec wrongAllGather = allGatherInPlace;
  wrongAllGather.input = static_cast<char*>(wrongAllGather.output) +
      (size_t)comm.rank * rawChunkBytes + 1;
  cocclPipelineContext context = {};
  EXPECT(cocclPreparePipeline(&wrongAllGather, 8, &context) == ncclSuccess);
  EXPECT(context.depth == 1);

  const cocclPipelineStage allToAllStages[] = {
      cocclPipelineCompress(), cocclPipelineAllToAll(&comm),
      cocclPipelineDecompress()};
  cocclPipelineSpec allToAll = {
      "alltoall", reinterpret_cast<void*>(inputBase),
      reinterpret_cast<void*>(inputBase), rawChunkCount, 4, ncclFloat32,
      cocclDefaultPolicy(cocclOperation::AllToAll), &comm, nullptr,
      allToAllStages, 3, cocclPipelineInPlaceNone};
  EXPECT(cocclPreparePipeline(&allToAll, 8, &context) == ncclSuccess);
  EXPECT(context.depth == 1);
}

}  // namespace

thread_local int ncclDebugNoWarn = 0;
void ncclDebugLog(ncclDebugLogLevel, unsigned long, const char*, int,
                  const char*, ...) {}

int main() {
  checkPointerLayouts();
  checkPlannerLayouts();
  std::printf("coccl M12 in-place plan: PASS\n");
  return 0;
}
