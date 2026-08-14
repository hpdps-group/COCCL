#include "core/pipeline/coccl_pipeline_stage.h"

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
                      size_t expectedContiguousBytes,
                      size_t expectedPitchedBytes) {
  size_t contiguousBytes = 0;
  size_t pitchedBytes = 0;
  cocclPipelineStageLayoutSpans(
      &context, &edge, &contiguousBytes, &pitchedBytes);
  if (contiguousBytes != expectedContiguousBytes ||
      pitchedBytes != expectedPitchedBytes) {
    fprintf(stderr,
            "%s: expected contiguous=%zu pitched=%zu, "
            "got contiguous=%zu pitched=%zu\n",
            scenario, expectedContiguousBytes, expectedPitchedBytes,
            contiguousBytes, pitchedBytes);
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
  if (expectLayoutSpans("four chunks", context, edge, 32, 104)) {
    return 1;
  }
  return 0;
}

int testFrameExchangeRouting() {
  ncclComm_t comm = (ncclComm_t)calloc(1, sizeof(ncclComm));
  if (comm == nullptr) return 1;
  comm->nRanks = 4;

  cocclCompressorFrameMetadata metadata = {};
  cocclPipelineEdge framed = {};
  framed.frameMetadata = &metadata;
  framed.frameStrideBytes = 64;
  const cocclPipelineEdge fixed = {};
  const bool valid =
      cocclPipelineStageUsesFrameExchange(
          cocclPipelineAllToAll(comm), framed) &&
      cocclPipelineStageUsesFrameExchange(
          cocclPipelineAllGather(comm), framed) &&
      !cocclPipelineStageUsesFrameExchange(
          cocclPipelineCompress(), framed) &&
      !cocclPipelineStageUsesFrameExchange(
          cocclPipelineAllToAll(comm), fixed);
  free(comm);
  if (!valid) {
    fprintf(stderr, "framed communication stage routing is incorrect\n");
    return 1;
  }
  return 0;
}

}  // namespace

int main() {
  if (testChunkPropagation() || testLayoutSpans() ||
      testFrameExchangeRouting()) {
    return 1;
  }
  printf("coccl pipeline stage tests passed\n");
  return 0;
}
