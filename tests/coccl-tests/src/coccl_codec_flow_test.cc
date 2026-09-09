#include "core/pipeline/coccl_pipeline_internal.h"

#include "comm.h"
#include "core/compression/compress.h"

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
void* const kFramed = reinterpret_cast<void*>(0x33);
int drcCalls;
int drCalls;
int reduceScatterCalls;

}  // namespace

thread_local int ncclDebugNoWarn = 0;
void ncclDebugLog(ncclDebugLogLevel, unsigned long, const char*, int,
                  const char*, ...) {}

ncclResult_t ncclDecompReduceComp(
    void* decoder, void* encoder, ncclComm_t owner,
    const cocclCompressorView& input, cocclCompressorView* output,
    size_t reduceChunks, ncclDataType_t originalDatatype,
    size_t originalElements, cudaStream_t) {
  ++drcCalls;
  EXPECT(decoder == kIntra);
  EXPECT(encoder == kInter);
  EXPECT(owner->rank == 3);
  EXPECT(input.chunks == 8 && input.datatype == ncclInt8);
  EXPECT(reduceChunks == 4);
  EXPECT(originalDatatype == ncclFloat32 && originalElements == 32);
  output->bytes = 16;
  output->elements = 16;
  output->chunks = 2;
  output->datatype = ncclInt8;
  return ncclSuccess;
}

ncclResult_t ncclDecompressReduce(
    void* decoder, ncclComm_t owner, const cocclCompressorView& input,
    cocclCompressorView* output, size_t reduceChunks, cudaStream_t) {
  ++drCalls;
  EXPECT(decoder == kInter);
  EXPECT(owner->rank == 3);
  EXPECT(input.chunks == 2 && reduceChunks == 2);
  EXPECT(output->elements == 16 && output->chunks == 1);
  output->bytes = 64;
  return ncclSuccess;
}

ncclResult_t ncclReduceScatter(
    const void*, void*, size_t recvcount, ncclDataType_t datatype,
    ncclRedOp_t op, ncclComm_t comm, cudaStream_t) {
  ++reduceScatterCalls;
  EXPECT(recvcount == 32);
  EXPECT(datatype == ncclFloat32 && op == ncclSum && comm->nRanks == 4);
  return ncclSuccess;
}

ncclResult_t ncclCompress(
    void*, const cocclCompressorView&, cocclCompressorView*, int,
    cudaStream_t) {
  return ncclInternalError;
}

ncclResult_t ncclDecompress(
    void*, const cocclCompressorView&, cocclCompressorView*, cudaStream_t) {
  return ncclInternalError;
}

bool cocclCompressorSupports(void* compressor, cocclCompressorCapability capability) {
  return compressor == kFramed && capability == cocclCompressorCapabilityFramed;
}

bool cocclFrameMetadataValid(
    const cocclCompressorFrameMetadata&, size_t) {
  return true;
}

ncclResult_t ncclAllToAll(const void*, void*, size_t, ncclDataType_t,
                          ncclComm_t, cudaStream_t) {
  return ncclInternalError;
}

ncclResult_t ncclAllGather(const void*, void*, size_t, ncclDataType_t,
                           ncclComm_t, cudaStream_t) {
  return ncclInternalError;
}

ncclResult_t cocclLaunchPackSlice(
    const void*, size_t, void*, size_t, size_t, cocclPipelineInputLayout,
    int, int, cudaStream_t) {
  return ncclInternalError;
}

ncclResult_t cocclLaunchUnpackSlice(
    const void*, void*, size_t, size_t, size_t,
    cocclPipelineOutputLayout, int, int, cudaStream_t) {
  return ncclInternalError;
}

ncclResult_t cocclBuildAllToAllFrameExchanges(
    const void*, void*, size_t, size_t, ncclComm_t, cudaStream_t,
    const cocclCompressorFrameMetadata*,
    const cocclCompressorFrameMetadata*, cocclFrameExchange*, size_t,
    size_t*) {
  return ncclInternalError;
}

ncclResult_t cocclBuildAllGatherFrameExchanges(
    const void*, void*, size_t, size_t, ncclComm_t, cudaStream_t,
    const cocclCompressorFrameMetadata*,
    const cocclCompressorFrameMetadata*, cocclFrameExchange*, size_t,
    size_t*) {
  return ncclInternalError;
}

ncclResult_t cocclCommitFrameExchange(
    const cocclFrameExchange*, size_t) {
  return ncclInternalError;
}

int main() {
  ncclComm owner = {};
  owner.rank = 3;
  owner.nRanks = 8;
  ncclComm intra = {};
  intra.nRanks = 4;
  const cocclPipelineStageContext context = {
      16, 64, 256, ncclFloat32, &owner, nullptr,
      cocclPipelineInputContiguous, cocclPipelineOutputContiguous,
      2, 4};

  cocclPipelineEdge encoded = {
      reinterpret_cast<void*>(0x100000), 64, 64, ncclInt8, 8,
      kIntra, nullptr, 0};
  cocclPipelineStageOutput output = {
      reinterpret_cast<void*>(0x200000), 64, nullptr, 0};
  const cocclPipelineStage drc =
      cocclPipelineDecompReduceComp(4, kInter);
  EXPECT(cocclExecutePipelineStage(
             &context, &drc, &encoded, &output, nullptr) == ncclSuccess);
  EXPECT(drcCalls == 1);
  EXPECT(encoded.compressor == kInter && encoded.logicalChunks == 2);

  output = {reinterpret_cast<void*>(0x300000), 64, nullptr, 0};
  const cocclPipelineStage dr = cocclPipelineDecompressReduce(2);
  EXPECT(cocclExecutePipelineStage(
             &context, &dr, &encoded, &output, nullptr) == ncclSuccess);
  EXPECT(drCalls == 1);
  EXPECT(encoded.compressor == nullptr && encoded.logicalChunks == 1);
  EXPECT(encoded.datatype == ncclFloat32 && encoded.bytes == 64);

  cocclPipelineEdge raw = {
      reinterpret_cast<void*>(0x400000), 512, 128, ncclFloat32, 8,
      nullptr, nullptr, 0};
  output = {reinterpret_cast<void*>(0x500000), 128, nullptr, 0};
  const cocclPipelineStage native = cocclPipelineReduceScatter(&intra);
  EXPECT(cocclExecutePipelineStage(
             &context, &native, &raw, &output, nullptr) == ncclSuccess);
  EXPECT(reduceScatterCalls == 1);
  EXPECT(raw.logicalChunks == 2 && raw.totalElements == 32);
  EXPECT(raw.compressor == nullptr && raw.bytes == 128);

  // Full-size Encoded still enters the decoder (subAdd initialization).
  cocclPipelineStage recv = cocclPipelineSendRecv(&intra, 1, cocclPipelineRecv, kInter);
  cocclCompressorFrameMetadata metadata = {128, cocclCompressorFrameEncoded, 0};
  cocclPipelineEdge received = {};
  received.logicalChunks = 1;
  cocclPipelineApplyReceivedFrame(recv, metadata, &received, output);
  EXPECT(received.ptr == output.ptr && received.compressor == kInter);
  EXPECT(received.bytes == 128 && received.totalElements == 128);
  EXPECT(received.datatype == ncclInt8 && received.logicalChunks == 1);
  EXPECT(received.frameMetadata == nullptr && received.frameStrideBytes == 0);
  metadata = {64, cocclCompressorFrameRaw, 0};
  cocclPipelineApplyReceivedFrame(recv, metadata, &received, output);
  EXPECT(received.bytes == 64 && received.totalElements == 64);
  EXPECT(received.datatype == COCCL_COMPRESSOR_RAW_PASSTHROUGH);

  recv.compressor = kFramed;
  output.frameMetadata = &metadata;
  output.frameStrideBytes = output.capacityBytes;
  const uint32_t encodings[] = {cocclCompressorFrameRaw, cocclCompressorFrameEncoded};
  for (uint32_t encoding : encodings) {
    metadata = {64, encoding, 0};
    cocclPipelineApplyReceivedFrame(recv, metadata, &received, output);
    EXPECT(received.compressor == kFramed && received.datatype == ncclInt8);
    EXPECT(received.bytes == output.capacityBytes);
    EXPECT(received.totalElements == output.capacityBytes && received.logicalChunks == 1);
    EXPECT(received.frameMetadata == &metadata);
    EXPECT(received.frameStrideBytes == output.frameStrideBytes);
  }

  std::printf("coccl codec flow: PASS\n");
  return 0;
}
