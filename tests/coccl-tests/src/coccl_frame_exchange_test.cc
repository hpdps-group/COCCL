#include "core/pipeline/coccl_frame_exchange.h"

#include "comm.h"
#include "runtime/coccl_runtime.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

struct SubmittedCall {
  char kind;
  int peer;
  size_t bytes;
  const void* send;
  void* recv;
  ncclComm_t comm;
  cudaStream_t stream;
};

std::vector<SubmittedCall> submitted;

void fail(const char* message) {
  std::fprintf(stderr, "%s\n", message);
  std::exit(1);
}

#define EXPECT(expression) \
  do { if (!(expression)) fail(#expression); } while (0)

void testMetadata() {
  cocclCompressorFrameMetadata encoded = {
      17, cocclCompressorFrameEncoded, 0};
  cocclCompressorFrameMetadata raw = {
      64, cocclCompressorFrameRaw, 0};
  if (!cocclFrameMetadataValid(encoded, 64) ||
      !cocclFrameMetadataValid(raw, 64)) {
    fail("valid encoded/raw metadata was rejected");
  }
  raw.payloadBytes = 63;
  if (!cocclFrameMetadataValid(raw, 64)) {
    fail("raw payload smaller than its physical slot was rejected");
  }
  encoded.payloadBytes = 0;
  if (cocclFrameMetadataValid(encoded, 64)) {
    fail("empty encoded frame was accepted");
  }
  encoded = {17, cocclCompressorFrameEncoded, 99};
  if (!cocclFrameMetadataValid(encoded, 64)) {
    fail("reserved metadata field affected receive semantics");
  }
}

void testCommitOrder() {
  ncclComm_t comm0 = (ncclComm_t)std::calloc(1, sizeof(ncclComm));
  ncclComm_t comm1 = (ncclComm_t)std::calloc(1, sizeof(ncclComm));
  if (comm0 == nullptr || comm1 == nullptr) fail("comm allocation failed");
  comm0->nRanks = 4;
  comm1->nRanks = 4;
  cudaStream_t stream0 = reinterpret_cast<cudaStream_t>(1);
  cudaStream_t stream1 = reinterpret_cast<cudaStream_t>(2);
  unsigned char send[64] = {};
  unsigned char recv[64] = {};
  const cocclFrameExchange exchanges[] = {
      {1, send, recv, 10, 11, 64, comm0, stream0},
      {2, send, nullptr, 12, 0, 64, comm1, stream1},
  };

  submitted.clear();
  if (cocclCommitFrameExchange(exchanges, 2, nullptr, nullptr) !=
      ncclSuccess) {
    fail("frame batch submission failed");
  }
  const char expected[] = {'G', 'R', 'S', 'S', 'E'};
  if (submitted.size() != sizeof(expected)) fail("unexpected call count");
  for (size_t i = 0; i < sizeof(expected); ++i) {
    if (submitted[i].kind != expected[i]) fail("unexpected call order");
  }
  if (submitted[1].comm != comm0 || submitted[1].stream != stream0 ||
      submitted[2].comm != comm0 || submitted[2].stream != stream0 ||
      submitted[3].comm != comm1 || submitted[3].stream != stream1 ||
      submitted[1].bytes != 11 || submitted[2].bytes != 10 ||
      submitted[3].bytes != 12) {
    fail("exchange context or byte count was not preserved");
  }
  std::free(comm1);
  std::free(comm0);
}

void testAllToAllMapping() {
  unsigned char send[8 * 64] = {};
  unsigned char recv[8 * 64] = {};
  cocclCompressorFrameMetadata sendMetadata[8] = {};
  cocclCompressorFrameMetadata recvMetadata[8] = {};
  for (size_t frame = 0; frame < 8; ++frame) {
    sendMetadata[frame] = {
        frame + 1, cocclCompressorFrameEncoded, 0};
    recvMetadata[frame] = {
        frame + 11, cocclCompressorFrameEncoded, 0};
  }
  cocclFrameExchange exchanges[8] = {};
  size_t count = 0;
  EXPECT(cocclBuildAllToAllFrameExchanges(
             send, recv, 8, 64, 4, sendMetadata, recvMetadata,
             exchanges, 8, &count) == ncclSuccess);
  EXPECT(count == 8);
  for (size_t frame = 0; frame < count; ++frame) {
    EXPECT(exchanges[frame].peer == static_cast<int>(frame / 2));
    EXPECT(exchanges[frame].sendSlot == send + frame * 64);
    EXPECT(exchanges[frame].recvSlot == recv + frame * 64);
    EXPECT(exchanges[frame].sendBytes == sendMetadata[frame].payloadBytes);
    EXPECT(exchanges[frame].recvBytes == recvMetadata[frame].payloadBytes);
  }
}

void testAllGatherMapping() {
  unsigned char send[2 * 64] = {};
  unsigned char recv[8 * 64] = {};
  const cocclCompressorFrameMetadata sendMetadata[2] = {
      {20, cocclCompressorFrameEncoded, 0},
      {64, cocclCompressorFrameRaw, 0}};
  cocclCompressorFrameMetadata recvMetadata[8] = {};
  for (size_t frame = 0; frame < 8; ++frame) {
    recvMetadata[frame] = {
        frame + 1, cocclCompressorFrameEncoded, 0};
  }
  cocclFrameExchange exchanges[8] = {};
  size_t count = 0;
  EXPECT(cocclBuildAllGatherFrameExchanges(
             send, recv, 2, 64, 4, sendMetadata, recvMetadata,
             exchanges, 8, &count) == ncclSuccess);
  EXPECT(count == 8);
  for (size_t index = 0; index < count; ++index) {
    const size_t localFrame = index % 2;
    EXPECT(exchanges[index].peer == static_cast<int>(index / 2));
    EXPECT(exchanges[index].sendSlot == send + localFrame * 64);
    EXPECT(exchanges[index].recvSlot == recv + index * 64);
    EXPECT(exchanges[index].sendBytes ==
           sendMetadata[localFrame].payloadBytes);
    EXPECT(exchanges[index].recvBytes == recvMetadata[index].payloadBytes);
  }
}

void testAllGatherVCommitOrder() {
  ncclComm_t comm = (ncclComm_t)std::calloc(1, sizeof(ncclComm));
  if (comm == nullptr) fail("comm allocation failed");
  comm->nRanks = 4;
  unsigned char send[2 * 64] = {};
  unsigned char recv[8 * 64] = {};
  cocclFrameExchange exchanges[8] = {};
  for (int root = 0; root < comm->nRanks; ++root) {
    for (size_t frame = 0; frame < 2; ++frame) {
      const size_t index = (size_t)root * 2 + frame;
      exchanges[index] = {
          root, send + frame * 64, recv + index * 64,
          frame + 1, index + 11, 64, nullptr, nullptr};
    }
  }

  submitted.clear();
  EXPECT(cocclCommitAllGatherVFrameExchange(
             exchanges, 2, comm,
             reinterpret_cast<cudaStream_t>(3)) == ncclSuccess);
  EXPECT(submitted.size() == 12);
  for (size_t frame = 0; frame < 2; ++frame) {
    const size_t groupBase = frame * 6;
    EXPECT(submitted[groupBase].kind == 'G');
    EXPECT(submitted[groupBase + 5].kind == 'E');
    for (int root = 0; root < comm->nRanks; ++root) {
      const size_t index = (size_t)root * 2 + frame;
      const SubmittedCall& call = submitted[groupBase + 1 + root];
      EXPECT(call.kind == 'B');
      EXPECT(call.peer == root);
      EXPECT(call.bytes == exchanges[index].recvBytes);
      EXPECT(call.send == exchanges[index].sendSlot);
      EXPECT(call.recv == exchanges[index].recvSlot);
      EXPECT(call.comm == comm);
      EXPECT(call.stream == reinterpret_cast<cudaStream_t>(3));
    }
  }
  std::free(comm);
}

}  // namespace

ncclResult_t ncclGroupStart() {
  submitted.push_back({'G', -1, 0, nullptr, nullptr, nullptr, nullptr});
  return ncclSuccess;
}

ncclResult_t ncclGroupEnd() {
  submitted.push_back({'E', -1, 0, nullptr, nullptr, nullptr, nullptr});
  return ncclSuccess;
}

ncclResult_t ncclBroadcast(
    const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, int root, ncclComm_t comm,
    cudaStream_t stream) {
  if (datatype != ncclInt8) return ncclInvalidArgument;
  submitted.push_back(
      {'B', root, count, sendbuff, recvbuff, comm, stream});
  return ncclSuccess;
}

ncclResult_t cocclReplayNativeCall(const cocclInfo& info) {
  if (info.operation != cocclOperation::SendRecv ||
      info.datatype != ncclInt8 ||
      (info.func != ncclFuncRecv && info.func != ncclFuncSend)) {
    return ncclInvalidArgument;
  }
  submitted.push_back({info.func == ncclFuncRecv ? 'R' : 'S',
                       info.peer, info.count, info.sendbuff,
                       info.recvbuff, info.comm, info.stream});
  return ncclSuccess;
}

int main() {
  testMetadata();
  testCommitOrder();
  testAllToAllMapping();
  testAllGatherMapping();
  testAllGatherVCommitOrder();
  std::printf("COCCL frame exchange tests passed\n");
  return 0;
}
