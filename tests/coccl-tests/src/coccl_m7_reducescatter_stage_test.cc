#include "core/pipeline/coccl_pipeline_internal.h"

#include "comm.h"
#include "core/compression/compress.h"

#include <cstdio>
#include <cstdlib>

namespace {

int compressCalls = 0;
int allToAllCalls = 0;
int drcCalls = 0;
int drCalls = 0;

void fail(const char* expression, int line) {
  std::fprintf(stderr, "line %d: %s\n", line, expression);
  std::exit(1);
}

#define EXPECT(expression) \
  do { if (!(expression)) fail(#expression, __LINE__); } while (0)

}  // namespace

thread_local int ncclDebugNoWarn = 0;
void ncclDebugLog(ncclDebugLogLevel, unsigned long, const char*, int,
                  const char*, ...) {}

ncclResult_t ncclCompress(
    void* compressor, const cocclCompressorView& input,
    cocclCompressorView* output, int, cudaStream_t) {
  ++compressCalls;
  EXPECT(compressor == reinterpret_cast<void*>(0x1));
  EXPECT(input.elements == 256 && input.chunks == 4 &&
         input.datatype == ncclFloat32);
  output->bytes = 128;
  output->elements = 128;
  output->chunks = 4;
  output->datatype = ncclInt8;
  return ncclSuccess;
}

ncclResult_t ncclAllToAll(const void*, void*, size_t count,
                          ncclDataType_t datatype, ncclComm_t comm,
                          cudaStream_t) {
  ++allToAllCalls;
  EXPECT(datatype == ncclUint8);
  EXPECT((comm->nRanks == 2 && count == 64) ||
         (comm->nRanks == 2 && count == 16));
  return ncclSuccess;
}

ncclResult_t ncclDecompReduceComp(
    void* decoder, void* encoder, ncclComm_t,
    const cocclCompressorView& input,
    cocclCompressorView* output, size_t reduceChunks,
    ncclDataType_t originalDatatype, size_t originalElements,
    cudaStream_t) {
  ++drcCalls;
  EXPECT(decoder == reinterpret_cast<void*>(0x1));
  EXPECT(encoder == reinterpret_cast<void*>(0x1));
  EXPECT(originalElements == 128 && originalDatatype == ncclFloat32);
  EXPECT(input.elements == 128 && input.datatype == ncclInt8);
  EXPECT(input.chunks == 4 && reduceChunks == 2);
  output->bytes = 32;
  output->elements = 32;
  output->chunks = 2;
  output->datatype = ncclInt8;
  return ncclSuccess;
}

ncclResult_t ncclDecompressReduce(
    void* compressor, ncclComm_t, const cocclCompressorView& input,
    cocclCompressorView* output, size_t reduceChunks, cudaStream_t) {
  ++drCalls;
  EXPECT(compressor == reinterpret_cast<void*>(0x1));
  EXPECT(input.elements == 32 && input.datatype == ncclInt8);
  EXPECT(output->elements == 64 && output->datatype == ncclFloat32);
  EXPECT(input.chunks == 2 && reduceChunks == 2);
  output->bytes = 256;
  return ncclSuccess;
}

ncclResult_t ncclAllGather(const void*, void*, size_t, ncclDataType_t,
                           ncclComm_t, cudaStream_t) {
  return ncclInternalError;
}

ncclResult_t ncclDecompress(
    void*, const cocclCompressorView&, cocclCompressorView*, cudaStream_t) {
  return ncclInternalError;
}

ncclResult_t cocclLaunchPackSlice(const void*, size_t, void*, size_t,
                                  size_t, cocclPipelineInputLayout, int, int,
                                  cudaStream_t) {
  return ncclInternalError;
}

ncclResult_t cocclLaunchUnpackSlice(const void*, void*, size_t, size_t,
                                    size_t, cudaStream_t) {
  return ncclInternalError;
}

int main() {
  ncclComm owner = {};
  owner.nRanks = 4;
  owner.rank = 1;
  ncclComm intra = {};
  intra.nRanks = 2;
  ncclComm inter = {};
  inter.nRanks = 2;
  const cocclPipelineStageContext context = {
      64, 256, 1024, ncclFloat32, &owner, nullptr,
      cocclPipelineInputHierarchicalSwizzle, 2, 2};
  cocclPipelineEdge edge = {
      reinterpret_cast<void*>(0x100000), 1024, 256, ncclFloat32, 4,
      nullptr, nullptr, 0};
  cocclPipelineStageOutput output = {
      reinterpret_cast<void*>(0x200000), 1024};

  const cocclPipelineStage compress = cocclPipelineCompress(reinterpret_cast<void*>(0x1));
  EXPECT(cocclExecutePipelineStage(
             &context, &compress, &edge, &output, nullptr) == ncclSuccess);
  EXPECT(compressCalls == 1 && edge.bytes == 128 &&
         edge.totalElements == 128 && edge.logicalChunks == 4);

  const cocclPipelineStage intraAllToAll = cocclPipelineAllToAll(&intra);
  output = {reinterpret_cast<void*>(0x300000), 1024};
  EXPECT(cocclExecutePipelineStage(
             &context, &intraAllToAll, &edge, &output, nullptr) ==
         ncclSuccess);

  const cocclPipelineStage drc = cocclPipelineDecompReduceComp(
      2, reinterpret_cast<void*>(0x1));
  output = {reinterpret_cast<void*>(0x400000), 512};
  EXPECT(cocclExecutePipelineStage(
             &context, &drc, &edge, &output, nullptr) == ncclSuccess);
  EXPECT(drcCalls == 1 && edge.bytes == 32 && edge.totalElements == 32 &&
         edge.logicalChunks == 2 && edge.datatype == ncclInt8);

  const cocclPipelineStage interAllToAll = cocclPipelineAllToAll(&inter);
  output = {reinterpret_cast<void*>(0x500000), 512};
  EXPECT(cocclExecutePipelineStage(
             &context, &interAllToAll, &edge, &output, nullptr) ==
         ncclSuccess);

  const cocclPipelineStage dr = cocclPipelineDecompressReduce(2);
  output = {reinterpret_cast<void*>(0x600000), 256};
  EXPECT(cocclExecutePipelineStage(
             &context, &dr, &edge, &output, nullptr) == ncclSuccess);
  EXPECT(drCalls == 1 && allToAllCalls == 2 && edge.bytes == 256 &&
         edge.totalElements == 64 && edge.logicalChunks == 1 &&
         edge.datatype == ncclFloat32);
  std::printf("coccl M7 reducescatter stage: PASS\n");
  return 0;
}
