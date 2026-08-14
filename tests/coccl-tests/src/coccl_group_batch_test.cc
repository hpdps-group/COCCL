#include "runtime/coccl_group.h"
#include "runtime/coccl_runtime.h"

#include <stdio.h>
#include <vector>

namespace {

std::vector<int> replayed;
std::vector<int> executed;
std::vector<int> batched;

void ignoreState(cocclCompressorRuntimeState*) {}

cocclPreparedCall preparedCall(cocclOperation operation, int id) {
  cocclPreparedCall call;
  call.info.operation = operation;
  call.info.peer = id;
  call.compressor.state = std::shared_ptr<cocclCompressorRuntimeState>(
      reinterpret_cast<cocclCompressorRuntimeState*>(1), ignoreState);
  return call;
}

cocclInfo nativeCall(cocclOperation operation, int id) {
  cocclInfo info;
  info.operation = operation;
  info.peer = id;
  return info;
}

void reset() {
  cocclGroupAbort();
  replayed.clear();
  executed.clear();
  batched.clear();
}

bool equal(const std::vector<int>& values,
           std::initializer_list<int> expected) {
  return values == std::vector<int>(expected);
}

int testCompressedBatch() {
  reset();
  const cocclPreparedCall send =
      preparedCall(cocclOperation::SendRecv, 1);
  const cocclPreparedCall recv =
      preparedCall(cocclOperation::SendRecv, 2);
  cocclGroupEnqueue(&send);
  cocclGroupEnqueue(&recv);
  if (cocclGroupPrepareEnd(false) != ncclSuccess ||
      !cocclGroupHasPending() || !replayed.empty() ||
      cocclGroupDrain() != ncclSuccess ||
      !equal(batched, {1, 2}) || cocclGroupHasPending()) {
    fprintf(stderr, "pure Send/Recv group did not use one batch\n");
    return 1;
  }
  return 0;
}

int testIneligibleFallback() {
  reset();
  const cocclPreparedCall send =
      preparedCall(cocclOperation::SendRecv, 3);
  const cocclInfo recv = nativeCall(cocclOperation::SendRecv, 4);
  cocclGroupEnqueue(&send);
  cocclGroupEnqueueNative(&recv);
  if (cocclGroupPrepareEnd(false) != ncclSuccess ||
      !equal(replayed, {3, 4}) || cocclGroupHasPending() ||
      cocclGroupDrain() != ncclSuccess || !batched.empty()) {
    fprintf(stderr, "ineligible Send/Recv did not replay the whole group\n");
    return 1;
  }
  return 0;
}

int testMixedFallback() {
  reset();
  const cocclPreparedCall send =
      preparedCall(cocclOperation::SendRecv, 5);
  const cocclPreparedCall collective =
      preparedCall(cocclOperation::AllGather, 6);
  cocclGroupEnqueue(&send);
  cocclGroupEnqueue(&collective);
  if (cocclGroupPrepareEnd(false) != ncclSuccess ||
      !equal(replayed, {5, 6}) || cocclGroupHasPending()) {
    fprintf(stderr, "mixed P2P/collective group was not replayed natively\n");
    return 1;
  }
  return 0;
}

int testNativePendingFallback() {
  reset();
  const cocclPreparedCall collective =
      preparedCall(cocclOperation::AllGather, 7);
  cocclGroupEnqueue(&collective);
  if (cocclGroupPrepareEnd(true) != ncclSuccess ||
      !equal(replayed, {7}) || cocclGroupHasPending()) {
    fprintf(stderr, "native pending work did not force native replay\n");
    return 1;
  }
  return 0;
}

int testCollectiveDrain() {
  reset();
  const cocclPreparedCall collective =
      preparedCall(cocclOperation::AllGather, 8);
  cocclGroupEnqueue(&collective);
  if (cocclGroupPrepareEnd(false) != ncclSuccess ||
      cocclGroupDrain() != ncclSuccess || !equal(executed, {8})) {
    fprintf(stderr, "pure collective group did not keep deferred replay\n");
    return 1;
  }
  return 0;
}

}  // namespace

ncclResult_t cocclReplayNativeCall(const cocclInfo& info) {
  replayed.push_back(info.peer);
  return ncclSuccess;
}

ncclResult_t cocclExecutePreparedCall(const cocclPreparedCall* call) {
  executed.push_back(call->info.peer);
  return ncclSuccess;
}

ncclResult_t cocclExecuteSendRecvBatch(
    const cocclPreparedCall* calls, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    batched.push_back(calls[i].info.peer);
  }
  return ncclSuccess;
}

int main() {
  if (testCompressedBatch() || testIneligibleFallback() ||
      testMixedFallback() || testNativePendingFallback() ||
      testCollectiveDrain()) {
    return 1;
  }
  printf("COCCL group batch tests passed\n");
  return 0;
}
