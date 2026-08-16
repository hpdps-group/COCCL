#include "coccl_pipeline_internal.h"

#include "comm.h"
#include "compress.h"

#include <cstdio>
#include <cstdlib>

namespace {

int compressCalls = 0;
int allGatherCalls = 0;
int decompressCalls = 0;

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
    size_t chunks, int, ncclCommOp_t operation, cudaStream_t) {
  ++compressCalls;
  EXPECT(rawChunkCount == 64 && chunks == 1 && operation == AllGather);
  *encodedChunkCount = 32;
  *encodedDatatype = ncclInt8;
  return ncclSuccess;
}

ncclResult_t ncclAllGather(const void*, void*, size_t count,
                           ncclDataType_t datatype, ncclComm_t comm,
                           cudaStream_t) {
  ++allGatherCalls;
  EXPECT(count == 32 && datatype == ncclUint8 && comm->nRanks == 4);
  return ncclSuccess;
}

ncclResult_t ncclDecompress(
    void*, const void*, size_t rawChunkCount, ncclDataType_t rawDatatype,
    size_t encodedChunkCount, ncclDataType_t encodedDatatype, size_t chunks,
    ncclCommOp_t operation, cudaStream_t) {
  ++decompressCalls;
  EXPECT(rawChunkCount == 64 && rawDatatype == ncclFloat32);
  EXPECT(encodedChunkCount == 32 && encodedDatatype == ncclInt8);
  EXPECT(chunks == 4 && operation == AllGather);
  return ncclSuccess;
}

ncclResult_t ncclAllToAll(const void*, void*, size_t, ncclDataType_t,
                          ncclComm_t, cudaStream_t) {
  return ncclInternalError;
}

ncclResult_t cocclLaunchPackSlice(const void*, size_t, void*, size_t,
                                  size_t, cudaStream_t) {
  return ncclInternalError;
}

ncclResult_t cocclLaunchUnpackSlice(const void*, void*, size_t, size_t,
                                    size_t, cudaStream_t) {
  return ncclInternalError;
}

int main() {
  ncclComm comm = {};
  comm.nRanks = 4;
  comm.rank = 2;
  const cocclPipelineStageContext context = {
      64, 256, 1024, ncclFloat32,
      cocclDefaultPolicy(cocclOperation::AllGather), &comm};
  cocclPipelineEdge edge = {
      reinterpret_cast<void*>(0x100000), 256, 64, ncclFloat32, 1};

  const cocclPipelineStage compress = cocclPipelineCompress();
  cocclPipelineStageOutput output = {
      reinterpret_cast<void*>(0x200000), 256};
  EXPECT(cocclExecutePipelineStage(
             &context, &compress, &edge, &output, nullptr) == ncclSuccess);
  EXPECT(compressCalls == 1 && edge.bytes == 32 &&
         edge.totalElements == 32 && edge.logicalChunks == 1);

  const cocclPipelineStage allGather = cocclPipelineAllGather(&comm);
  output = {reinterpret_cast<void*>(0x300000), 1024};
  EXPECT(cocclExecutePipelineStage(
             &context, &allGather, &edge, &output, nullptr) == ncclSuccess);
  EXPECT(allGatherCalls == 1 && edge.ptr == output.ptr);
  EXPECT(edge.bytes == 128 && edge.totalElements == 128 &&
         edge.logicalChunks == 4);

  const cocclPipelineStage decompress = cocclPipelineDecompress();
  output = {reinterpret_cast<void*>(0x400000), 1024};
  EXPECT(cocclExecutePipelineStage(
             &context, &decompress, &edge, &output, nullptr) == ncclSuccess);
  EXPECT(decompressCalls == 1 && edge.bytes == 1024 &&
         edge.totalElements == 256 && edge.logicalChunks == 4 &&
         edge.datatype == ncclFloat32);

  output = {reinterpret_cast<char*>(edge.ptr) + 1, 1024};
  EXPECT(cocclExecutePipelineStage(
             &context, &decompress, &edge, &output, nullptr) ==
         ncclInvalidUsage);
  EXPECT(decompressCalls == 1);
  std::printf("coccl M6 allgather stage: PASS\n");
  return 0;
}
