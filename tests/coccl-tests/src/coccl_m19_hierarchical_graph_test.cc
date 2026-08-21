#include "extensions/primitives/coccl_hierarchical_reduction.h"
#include "core/pipeline/coccl_pipeline_internal.h"
#include "core/runtime/coccl_prepared_call.h"
#include "coccl_m11_size_query_stub.h"

#include "comm.h"

#include <array>
#include <cstdio>
#include <cstdlib>

namespace {

void fail(const char* expression, int line) {
  std::fprintf(stderr, "line %d: %s\n", line, expression);
  std::exit(1);
}

#define EXPECT(expression) \
  do { if (!(expression)) fail(#expression, __LINE__); } while (0)

void* const kIntra = reinterpret_cast<void*>(0x11);
void* const kInter = reinterpret_cast<void*>(0x22);
void* const kDefault = reinterpret_cast<void*>(0x33);

struct ExpectedStage {
  cocclPipelineStageKind kind;
  ncclComm_t comm;
  size_t reduceChunks;
  void* compressor;
};

void setCompressor(cocclPreparedCall* prepared,
                   cocclCompressionScope scope, void* compressor) {
  prepared->compressors.handles[static_cast<size_t>(scope)] = compressor;
  prepared->compressors.datatypeSupported[static_cast<size_t>(scope)] = true;
}

void checkGraph(bool intraEnabled, bool interEnabled, bool finalEnabled,
                ncclComm_t owner, ncclComm_t intra, ncclComm_t inter,
                const ExpectedStage* expected, int expectedCount) {
  cocclPreparedCall prepared = {};
  if (intraEnabled) {
    setCompressor(&prepared, cocclCompressionScope::Intra, kIntra);
  }
  if (interEnabled) {
    setCompressor(&prepared, cocclCompressionScope::Inter, kInter);
  }

  cocclPipelineStage stages[7] = {};
  int count = cocclBuildHierarchicalReduction(
      &prepared, intra, inter, finalEnabled ? kDefault : nullptr, stages);
  EXPECT(count == expectedCount);
  for (int index = 0; index < count; ++index) {
    EXPECT(stages[index].kind == expected[index].kind);
    EXPECT(stages[index].comm == expected[index].comm);
    EXPECT(stages[index].reduceChunks == expected[index].reduceChunks);
    EXPECT(stages[index].compressor == expected[index].compressor);
  }

  size_t chunks = (size_t)owner->nRanks;
  for (int index = 0; index < count; ++index) {
    size_t outputChunks = 0;
    EXPECT(cocclPipelineStageOutputChunks(
               stages[index], chunks, &outputChunks) == ncclSuccess);
    chunks = outputChunks;
  }
  EXPECT(chunks == 1);

  const cocclPipelineStage gather = cocclPipelineAllGather(owner);
  EXPECT(cocclPipelineStageOutputChunks(gather, chunks, &chunks) ==
         ncclSuccess);
  EXPECT(chunks == (size_t)owner->nRanks);
}

void checkDrcPlanning(ncclComm_t owner, ncclComm_t intra,
                      bool sameCompressor) {
  cocclM11ConfigureSizeQueryStub(1, 4, 1, 8, true);
  const void* input = reinterpret_cast<const void*>(0x100000000ULL);
  void* output = reinterpret_cast<void*>(0x200000000ULL);
  void* const encoder = sameCompressor ? kIntra : kInter;
  const cocclPipelineStage stages[] = {
      cocclPipelineCompress(kIntra),
      cocclPipelineAllToAll(intra),
      cocclPipelineDecompReduceComp((size_t)intra->nRanks, encoder),
      cocclPipelineDecompress(),
  };
  const cocclPipelineSpec spec = {
      "m19-drc", input, output, 1024, (size_t)owner->nRanks,
      ncclFloat32, owner, nullptr, stages, 4,
      cocclPipelineInPlaceNone, cocclPipelineInputContiguous};
  cocclPipelineContext context = {};
  EXPECT(cocclPreparePipeline(&spec, 1, &context) == ncclSuccess);
  EXPECT(cocclM11CompressQueryObservation().calls == 2);
  EXPECT(cocclM11DrcQueryObservation().calls ==
         (sameCompressor ? 1 : 0));
}

}  // namespace

thread_local int ncclDebugNoWarn = 0;
void ncclDebugLog(ncclDebugLogLevel, unsigned long, const char*, int,
                  const char*, ...) {}

int main() {
  ncclComm owner = {};
  owner.nRanks = 8;
  owner.localRanks = 4;
  ncclComm intra = {};
  intra.nRanks = 4;
  ncclComm inter = {};
  inter.nRanks = 2;

  const ExpectedStage bothRaw[] = {
      {cocclPipelineStageCompress, nullptr, 0, kIntra},
      {cocclPipelineStageAllToAll, &intra, 0, nullptr},
      {cocclPipelineStageDecompReduceComp, nullptr, 4, kInter},
      {cocclPipelineStageAllToAll, &inter, 0, nullptr},
      {cocclPipelineStageDecompressReduce, nullptr, 2, nullptr},
  };
  checkGraph(true, true, false, &owner, &intra, &inter, bothRaw, 5);

  const ExpectedStage intraOnly[] = {
      {cocclPipelineStageCompress, nullptr, 0, kIntra},
      {cocclPipelineStageAllToAll, &intra, 0, nullptr},
      {cocclPipelineStageDecompressReduce, nullptr, 4, nullptr},
      {cocclPipelineStageReduceScatter, &inter, 0, nullptr},
  };
  checkGraph(true, false, false, &owner, &intra, &inter, intraOnly, 4);

  const ExpectedStage interOnly[] = {
      {cocclPipelineStageReduceScatter, &intra, 0, nullptr},
      {cocclPipelineStageCompress, nullptr, 0, kInter},
      {cocclPipelineStageAllToAll, &inter, 0, nullptr},
      {cocclPipelineStageDecompressReduce, nullptr, 2, nullptr},
  };
  checkGraph(false, true, false, &owner, &intra, &inter, interOnly, 4);

  const ExpectedStage allRaw[] = {
      {cocclPipelineStageReduceScatter, &intra, 0, nullptr},
      {cocclPipelineStageReduceScatter, &inter, 0, nullptr},
  };
  checkGraph(false, false, false, &owner, &intra, &inter, allRaw, 2);

  const ExpectedStage bothFinal[] = {
      {cocclPipelineStageCompress, nullptr, 0, kIntra},
      {cocclPipelineStageAllToAll, &intra, 0, nullptr},
      {cocclPipelineStageDecompReduceComp, nullptr, 4, kInter},
      {cocclPipelineStageAllToAll, &inter, 0, nullptr},
      {cocclPipelineStageDecompReduceComp, nullptr, 2, kDefault},
  };
  checkGraph(true, true, true, &owner, &intra, &inter, bothFinal, 5);

  const ExpectedStage intraFinal[] = {
      {cocclPipelineStageCompress, nullptr, 0, kIntra},
      {cocclPipelineStageAllToAll, &intra, 0, nullptr},
      {cocclPipelineStageDecompressReduce, nullptr, 4, nullptr},
      {cocclPipelineStageReduceScatter, &inter, 0, nullptr},
      {cocclPipelineStageCompress, nullptr, 0, kDefault},
  };
  checkGraph(true, false, true, &owner, &intra, &inter, intraFinal, 5);

  const ExpectedStage interFinal[] = {
      {cocclPipelineStageReduceScatter, &intra, 0, nullptr},
      {cocclPipelineStageCompress, nullptr, 0, kInter},
      {cocclPipelineStageAllToAll, &inter, 0, nullptr},
      {cocclPipelineStageDecompReduceComp, nullptr, 2, kDefault},
  };
  checkGraph(false, true, true, &owner, &intra, &inter, interFinal, 4);

  const ExpectedStage rawFinal[] = {
      {cocclPipelineStageReduceScatter, &intra, 0, nullptr},
      {cocclPipelineStageReduceScatter, &inter, 0, nullptr},
      {cocclPipelineStageCompress, nullptr, 0, kDefault},
  };
  checkGraph(false, false, true, &owner, &intra, &inter, rawFinal, 3);

  checkDrcPlanning(&owner, &intra, true);
  checkDrcPlanning(&owner, &intra, false);

  std::printf("coccl M19 hierarchical graphs: PASS\n");
  return 0;
}
