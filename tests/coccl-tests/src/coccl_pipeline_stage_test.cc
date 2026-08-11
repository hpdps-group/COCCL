#include "pipeline/coccl_pipeline_stage.h"

#include "comm.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

namespace {

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

int expectLayoutSpans(const char* scenario,
                      const cocclPipelineStageContext& context,
                      const cocclPipelineEdge& edge,
                      ncclResult_t expectedResult,
                      size_t expectedContiguousBytes,
                      size_t expectedPitchedBytes) {
  size_t contiguousBytes = 0;
  size_t pitchedBytes = 0;
  const ncclResult_t result = cocclPipelineStageLayoutSpans(
      &context, &edge, &contiguousBytes, &pitchedBytes);
  if (result != expectedResult ||
      (result == ncclSuccess &&
       (contiguousBytes != expectedContiguousBytes ||
        pitchedBytes != expectedPitchedBytes))) {
    fprintf(stderr,
            "%s: expected result=%d contiguous=%zu pitched=%zu, "
            "got result=%d contiguous=%zu pitched=%zu\n",
            scenario, (int)expectedResult, expectedContiguousBytes,
            expectedPitchedBytes, (int)result, contiguousBytes,
            pitchedBytes);
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
      expectChunks("Pack", cocclPipelinePack(), 4, ncclSuccess, 4) ||
      expectChunks("Unpack", cocclPipelineUnpack(), 4, ncclSuccess, 4) ||
      expectChunks("non-divisible DRC", cocclPipelineDecompReduceComp(4),
                   10, ncclInvalidArgument, 0) ||
      expectChunks("zero reduction", cocclPipelineDecompReduceComp(0), 8,
                   ncclInvalidArgument, 0) ||
      expectChunks("zero input", cocclPipelineCompress(), 0,
                   ncclInvalidArgument, 0) ||
      expectChunks("zero Pack input", cocclPipelinePack(), 0,
                   ncclInvalidArgument, 0) ||
      expectChunks("missing AllGather comm", cocclPipelineAllGather(nullptr),
                   1, ncclInvalidArgument, 0) ||
      expectChunks("AllGather overflow", cocclPipelineAllGather(comm),
                   SIZE_MAX, ncclInvalidArgument, 0);
  free(comm);
  return result;
}

int testLayoutSpans() {
  cocclPipelineStageContext context = {};
  context.rawSliceCount = 2;
  context.rawChunkBytes = 32;
  context.rawDatatype = ncclFloat32;

  cocclPipelineEdge edge = {};
  edge.bytes = 32;
  edge.totalElements = 8;
  edge.datatype = ncclFloat32;
  edge.logicalChunks = 4;
  if (expectLayoutSpans("four chunks", context, edge, ncclSuccess, 32, 104)) {
    return 1;
  }

  cocclPipelineEdge zeroChunks = edge;
  zeroChunks.logicalChunks = 0;
  if (expectLayoutSpans("zero chunks", context, zeroChunks,
                        ncclInvalidArgument, 0, 0)) {
    return 1;
  }

  cocclPipelineStageContext shortPitch = context;
  shortPitch.rawChunkBytes = 4;
  if (expectLayoutSpans("short pitch", shortPitch, edge,
                        ncclInvalidArgument, 0, 0)) {
    return 1;
  }

  cocclPipelineStageContext overflowing = context;
  overflowing.rawSliceCount = 1;
  overflowing.rawChunkBytes = SIZE_MAX;
  cocclPipelineEdge overflowEdge = edge;
  overflowEdge.bytes = 8;
  overflowEdge.totalElements = 2;
  overflowEdge.logicalChunks = 2;
  return expectLayoutSpans("pitched span overflow", overflowing,
                           overflowEdge, ncclInvalidArgument, 0, 0);
}

}  // namespace

int main() {
  if (testChunkPropagation() || testLayoutSpans()) {
    return 1;
  }
  printf("coccl pipeline stage tests passed\n");
  return 0;
}
