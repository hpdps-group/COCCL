#include "coccl_pipeline_internal.h"

#include "comm.h"
#include "compress.h"

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
    const void*, void**, size_t rawChunkCount, ncclDataType_t,
    size_t* encodedChunkCount, ncclDataType_t* encodedDatatype,
    size_t chunks, int, ncclCommOp_t operation, cudaStream_t) {
  ++compressCalls;
  EXPECT(rawChunkCount == 64 && chunks == 4 &&
         operation == ReduceScatter_Inter);
  *encodedChunkCount = 32;
  *encodedDatatype = ncclInt8;
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
    const void*, void**, size_t originalElements,
    ncclDataType_t originalDatatype, size_t encodedChunkCount,
    ncclDataType_t encodedDatatype, size_t* recompressedChunkCount,
    ncclDataType_t* recompressedDatatype, size_t inputChunks,
    size_t reduceChunks, ncclCommOp_t operation, cudaStream_t,
    ncclComm_t) {
  ++drcCalls;
  EXPECT(originalElements == 128 && originalDatatype == ncclFloat32);
  EXPECT(encodedChunkCount == 32 && encodedDatatype == ncclInt8);
  EXPECT(inputChunks == 4 && reduceChunks == 2);
  EXPECT(operation == ReduceScatter_Inter);
  *recompressedChunkCount = 16;
  *recompressedDatatype = ncclInt8;
  return ncclSuccess;
}

ncclResult_t ncclDecompressReduce(
    void*, const void*, size_t encodedChunkCount,
    ncclDataType_t encodedDatatype, size_t outputElements,
    ncclDataType_t outputDatatype, size_t inputChunks,
    size_t reduceChunks, ncclCommOp_t operation, cudaStream_t,
    ncclComm_t) {
  ++drCalls;
  EXPECT(encodedChunkCount == 16 && encodedDatatype == ncclInt8);
  EXPECT(outputElements == 64 && outputDatatype == ncclFloat32);
  EXPECT(inputChunks == 2 && reduceChunks == 2);
  EXPECT(operation == ReduceScatter_Inter);
  return ncclSuccess;
}

ncclResult_t ncclAllGather(const void*, void*, size_t, ncclDataType_t,
                           ncclComm_t, cudaStream_t) {
  return ncclInternalError;
}

ncclResult_t ncclDecompress(void*, const void*, size_t, ncclDataType_t,
                            size_t, ncclDataType_t, size_t, ncclCommOp_t,
                            cudaStream_t) {
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
  ncclComm owner = {};
  owner.nRanks = 4;
  owner.rank = 1;
  ncclComm intra = {};
  intra.nRanks = 2;
  ncclComm inter = {};
  inter.nRanks = 2;
  const cocclPipelineStageContext context = {
      64, 256, 1024, ncclFloat32,
      cocclHierarchicalPolicy(cocclOperation::ReduceScatter), &owner};
  cocclPipelineEdge edge = {
      reinterpret_cast<void*>(0x100000), 1024, 256, ncclFloat32, 4};
  cocclPipelineStageOutput output = {
      reinterpret_cast<void*>(0x200000), 1024};

  const cocclPipelineStage compress = cocclPipelineCompress();
  EXPECT(cocclExecutePipelineStage(
             &context, &compress, &edge, &output, nullptr) == ncclSuccess);
  EXPECT(compressCalls == 1 && edge.bytes == 128 &&
         edge.totalElements == 128 && edge.logicalChunks == 4);

  const cocclPipelineStage intraAllToAll = cocclPipelineAllToAll(&intra);
  output = {reinterpret_cast<void*>(0x300000), 1024};
  EXPECT(cocclExecutePipelineStage(
             &context, &intraAllToAll, &edge, &output, nullptr) ==
         ncclSuccess);

  const cocclPipelineStage drc = cocclPipelineDecompReduceComp(2);
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
