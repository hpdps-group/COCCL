#include "pipeline/coccl_pipeline_stage.h"

#include "comm.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

namespace {

constexpr size_t kAlignment = 256;

size_t alignBytes(size_t bytes) {
  return (bytes + kAlignment - 1) / kAlignment * kAlignment;
}

int expectChunks(const char* scenario, const cocclPipelineStage& stage,
                 size_t inputChunks, ncclResult_t expectedResult,
                 size_t expectedChunks) {
  size_t outputChunks = 0;
  const ncclResult_t result =
      cocclPipelineStageOutputChunks(stage, inputChunks, &outputChunks);
  if (result != expectedResult ||
      (result == ncclSuccess && outputChunks != expectedChunks)) {
    fprintf(stderr,
            "%s: expected result=%d chunks=%zu, got result=%d chunks=%zu\n",
            scenario, (int)expectedResult, expectedChunks, (int)result,
            outputChunks);
    return 1;
  }
  return 0;
}

int testChunkPropagation() {
  ncclComm_t comm = (ncclComm_t)calloc(1, sizeof(ncclComm));
  if (comm == nullptr) return 1;
  comm->nRanks = 8;

  size_t chunks = 8;
  const cocclPipelineStage reduceScatter[] = {
      cocclPipelineCompress(),
      cocclPipelineAllToAll(comm),
      cocclPipelineDecompReduceComp(4),
      cocclPipelineAllToAll(comm),
      cocclPipelineDecompressReduce(2),
  };
  const size_t reduceScatterExpected[] = {8, 8, 2, 2, 1};
  for (size_t i = 0; i < sizeof(reduceScatter) / sizeof(reduceScatter[0]);
       ++i) {
    size_t outputChunks = 0;
    if (cocclPipelineStageOutputChunks(reduceScatter[i], chunks,
                                       &outputChunks) != ncclSuccess ||
        outputChunks != reduceScatterExpected[i]) {
      fprintf(stderr, "ReduceScatter TwoShot failed at stage %zu\n", i);
      return 1;
    }
    chunks = outputChunks;
  }

  chunks = 8;
  const cocclPipelineStage allReduce[] = {
      cocclPipelineCompress(),
      cocclPipelineAllToAll(comm),
      cocclPipelineDecompReduceComp(4),
      cocclPipelineAllToAll(comm),
      cocclPipelineDecompReduceComp(2),
      cocclPipelineAllGather(comm),
      cocclPipelineDecompress(),
  };
  const size_t allReduceExpected[] = {8, 8, 2, 2, 1, 8, 8};
  for (size_t i = 0; i < sizeof(allReduce) / sizeof(allReduce[0]); ++i) {
    size_t outputChunks = 0;
    if (cocclPipelineStageOutputChunks(allReduce[i], chunks,
                                       &outputChunks) != ncclSuccess ||
        outputChunks != allReduceExpected[i]) {
      fprintf(stderr, "AllReduce TripleShot failed at stage %zu\n", i);
      return 1;
    }
    chunks = outputChunks;
  }

  const int result =
      expectChunks("AllGather", cocclPipelineAllGather(comm), 2,
                   ncclSuccess, 16) ||
      expectChunks("non-divisible DRC", cocclPipelineDecompReduceComp(4),
                   10, ncclInvalidArgument, 0) ||
      expectChunks("zero reduction", cocclPipelineDecompReduceComp(0), 8,
                   ncclInvalidArgument, 0) ||
      expectChunks("zero input", cocclPipelineCompress(), 0,
                   ncclInvalidArgument, 0) ||
      expectChunks("missing AllGather comm", cocclPipelineAllGather(nullptr),
                   1, ncclInvalidArgument, 0);
  free(comm);
  return result;
}

int testWorkspaceFormulas() {
  const size_t ranks = 8;
  const size_t nodes = 2;
  const size_t chunkBytes = 1000;
  const size_t rankBytes = alignBytes(ranks * chunkBytes);
  const size_t nodeBytes = alignBytes(nodes * chunkBytes);
  const size_t oneChunkBytes = alignBytes(chunkBytes);

  const size_t reduceScatterSerial = 2 * rankBytes + 2 * nodeBytes;
  const size_t reduceScatterOverlap = reduceScatterSerial + rankBytes;
  const size_t allReduceSerial =
      3 * rankBytes + 2 * nodeBytes + oneChunkBytes;
  const size_t allReduceOverlap =
      allReduceSerial + 2 * rankBytes;

  if (reduceScatterSerial != 20480 || reduceScatterOverlap != 28672 ||
      allReduceSerial != 29696 || allReduceOverlap != 46080) {
    fprintf(stderr, "pipeline workspace formulas changed unexpectedly\n");
    return 1;
  }
  return 0;
}

}  // namespace

int main() {
  if (testChunkPropagation() || testWorkspaceFormulas()) return 1;
  printf("coccl pipeline stage tests passed\n");
  return 0;
}
