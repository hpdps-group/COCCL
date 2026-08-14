#include "core/pipeline/coccl_frame_exchange.h"

#include "comm.h"
#include "runtime/coccl_runtime.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

namespace {

struct SubmittedCall {
  char kind;
  int peer;
  size_t bytes;
  ncclComm_t comm;
  cudaStream_t stream;
};

std::vector<SubmittedCall> submitted;

cocclCompressorFrameMetadata encoded(size_t bytes) {
  return {bytes, cocclCompressorFrameEncoded, 0};
}

cocclCompressorFrameMetadata raw(size_t bytes) {
  return {bytes, cocclCompressorFrameRaw, 0};
}

int testAllToAllMapping() {
  unsigned char send[8 * 64] = {};
  unsigned char recv[8 * 64] = {};
  cocclCompressorFrameMetadata sendMetadata[8];
  cocclCompressorFrameMetadata recvMetadata[8];
  for (size_t frame = 0; frame < 8; ++frame) {
    sendMetadata[frame] = encoded(frame + 1);
    recvMetadata[frame] = encoded(frame + 11);
  }
  sendMetadata[3] = raw(64);
  cocclFrameExchange exchanges[8] = {};
  size_t count = 0;
  if (cocclBuildAllToAllFrameExchanges(
          send, recv, 8, 64, 4, sendMetadata, recvMetadata,
          exchanges, 8, &count) != ncclSuccess || count != 8) {
    fprintf(stderr, "AllToAll frame mapping failed\n");
    return 1;
  }
  for (size_t frame = 0; frame < count; ++frame) {
    if (exchanges[frame].peer != (int)(frame / 2) ||
        exchanges[frame].sendSlot != send + frame * 64 ||
        exchanges[frame].recvSlot != recv + frame * 64 ||
        exchanges[frame].sendBytes != sendMetadata[frame].payloadBytes ||
        exchanges[frame].recvBytes != recvMetadata[frame].payloadBytes) {
      fprintf(stderr, "AllToAll descriptor %zu is incorrect\n", frame);
      return 1;
    }
  }
  sendMetadata[3] = raw(63);
  return cocclBuildAllToAllFrameExchanges(
             send, recv, 8, 64, 4, sendMetadata, recvMetadata,
             exchanges, 8, &count) == ncclInvalidUsage
      ? 0 : 1;
}

int testAllGatherMapping() {
  unsigned char send[2 * 64] = {};
  unsigned char recv[8 * 64] = {};
  const cocclCompressorFrameMetadata sendMetadata[2] = {
      encoded(20), raw(64)};
  cocclCompressorFrameMetadata recvMetadata[8];
  for (size_t frame = 0; frame < 8; ++frame) {
    recvMetadata[frame] = encoded(frame + 1);
  }
  cocclFrameExchange exchanges[8] = {};
  size_t count = 0;
  if (cocclBuildAllGatherFrameExchanges(
          send, recv, 2, 64, 4, sendMetadata, recvMetadata,
          exchanges, 8, &count) != ncclSuccess || count != 8) {
    fprintf(stderr, "AllGather frame mapping failed\n");
    return 1;
  }
  for (size_t index = 0; index < count; ++index) {
    const size_t localFrame = index % 2;
    if (exchanges[index].peer != (int)(index / 2) ||
        exchanges[index].sendSlot != send + localFrame * 64 ||
        exchanges[index].recvSlot != recv + index * 64 ||
        exchanges[index].sendBytes !=
            sendMetadata[localFrame].payloadBytes ||
        exchanges[index].recvBytes != recvMetadata[index].payloadBytes) {
      fprintf(stderr, "AllGather descriptor %zu is incorrect\n", index);
      return 1;
    }
  }
  return 0;
}

int testCommitOrder() {
  ncclComm_t comm0 = (ncclComm_t)calloc(1, sizeof(ncclComm));
  ncclComm_t comm1 = (ncclComm_t)calloc(1, sizeof(ncclComm));
  if (comm0 == nullptr || comm1 == nullptr) return 1;
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
  const ncclResult_t result =
      cocclCommitFrameExchange(exchanges, 2, nullptr, nullptr);
  const char expected[] = {'G', 'R', 'S', 'S', 'E'};
  if (result != ncclSuccess || submitted.size() != sizeof(expected)) {
    fprintf(stderr, "Batch frame submission failed\n");
    free(comm1);
    free(comm0);
    return 1;
  }
  for (size_t i = 0; i < sizeof(expected); ++i) {
    if (submitted[i].kind != expected[i]) {
      fprintf(stderr, "Batch frame order differs at %zu\n", i);
      free(comm1);
      free(comm0);
      return 1;
    }
  }
  if (submitted[1].comm != comm0 || submitted[1].stream != stream0 ||
      submitted[2].comm != comm0 || submitted[2].stream != stream0 ||
      submitted[3].comm != comm1 || submitted[3].stream != stream1) {
    fprintf(stderr, "Batch frame context was not preserved\n");
    free(comm1);
    free(comm0);
    return 1;
  }
  free(comm1);
  free(comm0);
  return 0;
}

int testValidation() {
  unsigned char send[64] = {};
  unsigned char recv[64] = {};
  cocclCompressorFrameMetadata sendMetadata[1] = {encoded(1)};
  cocclCompressorFrameMetadata recvMetadata[1] = {raw(64)};
  cocclFrameExchange exchange = {};
  size_t count = 0;

  sendMetadata[0].payloadBytes = 0;
  if (cocclBuildAllToAllFrameExchanges(
          send, recv, 1, 64, 1, sendMetadata, recvMetadata,
          &exchange, 1, &count) != ncclInvalidUsage) {
    fprintf(stderr, "zero frame payload was accepted\n");
    return 1;
  }
  sendMetadata[0] = encoded(65);
  if (cocclBuildAllToAllFrameExchanges(
          send, recv, 1, 64, 1, sendMetadata, recvMetadata,
          &exchange, 1, &count) != ncclInvalidUsage) {
    fprintf(stderr, "oversized frame payload was accepted\n");
    return 1;
  }
  sendMetadata[0] = raw(63);
  if (cocclBuildAllToAllFrameExchanges(
          send, recv, 1, 64, 1, sendMetadata, recvMetadata,
          &exchange, 1, &count) != ncclInvalidUsage) {
    fprintf(stderr, "short Raw frame was accepted\n");
    return 1;
  }
  ncclComm_t comm = (ncclComm_t)calloc(1, sizeof(ncclComm));
  if (comm == nullptr) return 1;
  comm->nRanks = 4;
  const cocclFrameExchange empty = {1, nullptr, nullptr, 0, 0, 64};
  const cocclFrameExchange invalidPeer = {4, send, recv, 1, 1, 64};
  const bool rejected =
      cocclCommitFrameExchange(&empty, 1, comm, nullptr) ==
          ncclInvalidArgument &&
      cocclCommitFrameExchange(&invalidPeer, 1, comm, nullptr) ==
          ncclInvalidArgument;
  free(comm);
  if (!rejected) {
    fprintf(stderr, "invalid Batch exchange descriptor was accepted\n");
    return 1;
  }
  return 0;
}

}  // namespace

ncclResult_t ncclGroupStart() {
  submitted.push_back({'G', -1, 0, nullptr, nullptr});
  return ncclSuccess;
}

ncclResult_t ncclGroupEnd() {
  submitted.push_back({'E', -1, 0, nullptr, nullptr});
  return ncclSuccess;
}

ncclResult_t cocclReplayNativeCall(const cocclInfo& info) {
  if (info.operation != cocclOperation::SendRecv ||
      info.datatype != ncclInt8 ||
      (info.func != ncclFuncRecv && info.func != ncclFuncSend)) {
    return ncclInvalidArgument;
  }
  submitted.push_back({info.func == ncclFuncRecv ? 'R' : 'S',
                       info.peer, info.count, info.comm, info.stream});
  return ncclSuccess;
}

int main() {
  if (testAllToAllMapping() || testAllGatherMapping() || testCommitOrder() ||
      testValidation()) {
    return 1;
  }
  printf("COCCL frame exchange tests passed\n");
  return 0;
}
