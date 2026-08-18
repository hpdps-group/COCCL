#include "coccl_pipeline_internal.h"

#include "comm.h"
#include "compress.h"

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
    const void*, void**, size_t rawChunkCount, ncclDataType_t,
    size_t* encodedChunkCount, ncclDataType_t* encodedDatatype,
    size_t chunks, int, ncclCommOp_t operation, cudaStream_t, size_t) {
  ++compressCalls;
  EXPECT(rawChunkCount == 64 && chunks == 4 && operation == AlltoAll);
  *encodedChunkCount = 32;
  *encodedDatatype = ncclUint8;
  return ncclSuccess;
}

ncclResult_t ncclDecompress(
    void*, const void*, size_t rawChunkCount, ncclDataType_t rawDatatype,
    size_t encodedChunkCount, ncclDataType_t encodedDatatype, size_t chunks,
    ncclCommOp_t operation, cudaStream_t) {
  ++decompressCalls;
  EXPECT(rawChunkCount == 64 && rawDatatype == ncclFloat32);
  EXPECT(encodedChunkCount == 32 && encodedDatatype == ncclUint8);
  EXPECT(chunks == 4 && operation == AlltoAll);
  return ncclSuccess;
}

ncclResult_t ncclAllToAll(const void*, void*, size_t count,
                          ncclDataType_t datatype, ncclComm_t comm,
                          cudaStream_t) {
  ++allToAllCalls;
  EXPECT(count == 32 && datatype == ncclUint8 && comm->nRanks == 4);
  return ncclSuccess;
}

ncclResult_t ncclAllGather(const void*, void*, size_t, ncclDataType_t,
                           ncclComm_t, cudaStream_t) {
  return ncclInternalError;
}

ncclResult_t ncclDecompReduceComp(
    const void*, void**, size_t, ncclDataType_t, size_t, ncclDataType_t,
    size_t*, ncclDataType_t*, size_t, size_t, ncclCommOp_t, cudaStream_t,
    ncclComm_t, size_t) {
  return ncclInternalError;
}

ncclResult_t ncclDecompressReduce(
    void*, const void*, size_t, ncclDataType_t, size_t, ncclDataType_t,
    size_t, size_t, ncclCommOp_t, cudaStream_t, ncclComm_t) {
  return ncclInternalError;
}

ncclResult_t cocclLaunchPackSlice(const void*, size_t pitch, void*,
                                  size_t sliceBytes, size_t chunks,
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
      64, 256, 1024, ncclFloat32,
      cocclDefaultPolicy(cocclOperation::AllToAll), &comm};
  cocclPipelineEdge edge = {
      reinterpret_cast<void*>(0x100000), 1024, 256, ncclFloat32, 4};

  const cocclPipelineStage pack = cocclPipelinePack();
  cocclPipelineStageOutput output = {
      reinterpret_cast<void*>(0x200000), 1024};
  EXPECT(cocclExecutePipelineStage(&context, &pack, &edge, &output,
                                   nullptr) == ncclSuccess);
  EXPECT(packCalls == 1 && edge.ptr == output.ptr && edge.logicalChunks == 4);

  const cocclPipelineStage compress = cocclPipelineCompress();
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
  std::printf("coccl M5 pipeline stage: PASS\n");
  return 0;
}
