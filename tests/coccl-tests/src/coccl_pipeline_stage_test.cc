#include "core/pipeline/coccl_pipeline_internal.h"

#include "comm.h"
#include "core/compression/compress.h"

#include <cstdio>
#include <cstdlib>

namespace {

int compressCalls = 0;
int allToAllCalls = 0;
int decompressCalls = 0;
int packCalls = 0;
int unpackCalls = 0;

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
  output->datatype = ncclUint8;
  return ncclSuccess;
}

ncclResult_t ncclDecompress(
    void* compressor, const cocclCompressorView& input,
    cocclCompressorView* output, cudaStream_t) {
  ++decompressCalls;
  EXPECT(compressor == reinterpret_cast<void*>(0x1));
  EXPECT(input.elements == 128 && input.chunks == 4 &&
         input.datatype == ncclUint8);
  EXPECT(output->elements == 256 && output->chunks == 4 &&
         output->datatype == ncclFloat32);
  output->bytes = 1024;
  return ncclSuccess;
}

ncclResult_t ncclAlltoAllConfig(
    const void*, void*, size_t count, ncclDataType_t datatype,
    ncclComm_t comm, cudaStream_t, const ncclCollConfig_t* config) {
  ++allToAllCalls;
  EXPECT(count == 32 && datatype == ncclUint8 && comm->nRanks == 4);
  EXPECT(config != nullptr && config->userProfilerTag == 0x1234);
  return ncclSuccess;
}

ncclResult_t ncclAllGatherConfig(
    const void*, void*, size_t, ncclDataType_t, ncclComm_t, cudaStream_t,
    const ncclCollConfig_t*) {
  return ncclInternalError;
}

ncclResult_t ncclDecompReduceComp(
    void*, void*, ncclComm_t, const cocclCompressorView&,
    cocclCompressorView*,
    size_t, ncclDataType_t, size_t, cudaStream_t) {
  return ncclInternalError;
}

ncclResult_t ncclDecompressReduce(
    void*, ncclComm_t, const cocclCompressorView&, cocclCompressorView*,
    size_t, cudaStream_t) {
  return ncclInternalError;
}

ncclResult_t cocclLaunchPackSlice(const void*, size_t pitch, void*,
                                  size_t sliceBytes, size_t chunks,
                                  cocclPipelineInputLayout, int, int,
                                  cudaStream_t) {
  ++packCalls;
  EXPECT(pitch == 1024 && sliceBytes == 256 && chunks == 4);
  return ncclSuccess;
}

ncclResult_t cocclLaunchUnpackSlice(const void*, void*, size_t pitch,
                                    size_t sliceBytes, size_t chunks,
                                    cudaStream_t) {
  ++unpackCalls;
  EXPECT(pitch == 1024 && sliceBytes == 256 && chunks == 4);
  return ncclSuccess;
}

int main() {
  ncclComm comm = {};
  comm.nRanks = 4;
  comm.rank = 2;
  const cocclPipelineStageContext context = {
      64, 256, 1024, ncclFloat32, &comm, nullptr,
      cocclPipelineInputContiguous, 1, 4, 0x1234};
  cocclPipelineEdge edge = {
      reinterpret_cast<void*>(0x100000), 1024, 256, ncclFloat32, 4,
      nullptr, nullptr, 0};

  const cocclPipelineStage pack = cocclPipelinePack();
  cocclPipelineStageOutput output = {
      reinterpret_cast<void*>(0x200000), 1024};
  EXPECT(cocclExecutePipelineStage(&context, &pack, &edge, &output,
                                   nullptr) == ncclSuccess);
  EXPECT(packCalls == 1 && edge.ptr == output.ptr && edge.logicalChunks == 4);

  const cocclPipelineStage compress = cocclPipelineCompress(reinterpret_cast<void*>(0x1));
  output = {reinterpret_cast<void*>(0x300000), 1024};
  EXPECT(cocclExecutePipelineStage(&context, &compress, &edge, &output,
                                   nullptr) == ncclSuccess);
  EXPECT(compressCalls == 1 && edge.bytes == 128 &&
         edge.totalElements == 128 && edge.logicalChunks == 4);

  const cocclPipelineStage allToAll = cocclPipelineAllToAll(&comm);
  output = {reinterpret_cast<void*>(0x400000), 1024};
  EXPECT(cocclExecutePipelineStage(&context, &allToAll, &edge, &output,
                                   nullptr) == ncclSuccess);
  EXPECT(allToAllCalls == 1 && edge.ptr == output.ptr);

  const cocclPipelineStage decompress = cocclPipelineDecompress();
  output = {reinterpret_cast<void*>(0x500000), 1024};
  EXPECT(cocclExecutePipelineStage(&context, &decompress, &edge, &output,
                                   nullptr) == ncclSuccess);
  EXPECT(decompressCalls == 1 && edge.bytes == 1024 &&
         edge.datatype == ncclFloat32 && edge.logicalChunks == 4);

  const cocclPipelineStage unpack = cocclPipelineUnpack();
  output = {reinterpret_cast<void*>(0x600000), 1024};
  EXPECT(cocclExecutePipelineStage(&context, &unpack, &edge, &output,
                                   nullptr) == ncclSuccess);
  EXPECT(unpackCalls == 1 && edge.ptr == output.ptr);

  output = {reinterpret_cast<char*>(edge.ptr) + 1, 1024};
  EXPECT(cocclExecutePipelineStage(&context, &unpack, &edge, &output,
                                   nullptr) == ncclInvalidUsage);
  EXPECT(unpackCalls == 1);
  std::printf("coccl pipeline stage: PASS\n");
  return 0;
}
