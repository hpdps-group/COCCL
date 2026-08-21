#include "runtime/coccl_group.h"
#include "core/runtime/coccl_group_internal.h"
#include "core/runtime/coccl_prepared_call.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

std::vector<int> replayed;
std::vector<int> executed;
std::vector<int> batched;
bool detached = false;

void fail(const char* message) {
  std::fprintf(stderr, "%s\n", message);
  std::exit(1);
}

cocclPreparedCall compressed(cocclOperation operation, int id) {
  cocclPreparedCall call;
  call.info.operation = operation;
  call.info.peer = id;
  call.compressors.handles[static_cast<size_t>(
      cocclCompressionScope::Default)] = reinterpret_cast<void*>(1);
  return call;
}

cocclInfo native(cocclOperation operation, int id) {
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
  detached = false;
}

bool equal(const std::vector<int>& values,
           std::initializer_list<int> expected) {
  return values == std::vector<int>(expected);
}

void testCompressedBatch() {
  reset();
  cocclPreparedCall send = compressed(cocclOperation::SendRecv, 1);
  cocclPreparedCall recv = compressed(cocclOperation::SendRecv, 2);
  cocclGroupEnqueue(&send);
  cocclGroupEnqueue(&recv);
  if (cocclGroupPrepareEnd(false) != ncclSuccess ||
      !cocclGroupHasPending() || cocclGroupDrain() != ncclSuccess ||
      !equal(batched, {1, 2}) || !detached || cocclGroupHasPending()) {
    fail("pure Send/Recv group did not execute as one detached batch");
  }
}

void testIneligibleFallback() {
  reset();
  cocclPreparedCall send = compressed(cocclOperation::SendRecv, 3);
  cocclInfo recv = native(cocclOperation::SendRecv, 4);
  cocclGroupEnqueue(&send);
  cocclGroupEnqueueNative(&recv);
  if (cocclGroupPrepareEnd(false) != ncclSuccess ||
      !equal(replayed, {3, 4}) || cocclGroupHasPending()) {
    fail("ineligible Send/Recv did not replay the complete group");
  }
}

void testMixedAndNativeFallback() {
  reset();
  cocclPreparedCall send = compressed(cocclOperation::SendRecv, 5);
  cocclPreparedCall collective = compressed(cocclOperation::AllGather, 6);
  cocclGroupEnqueue(&send);
  cocclGroupEnqueue(&collective);
  if (cocclGroupPrepareEnd(false) != ncclSuccess ||
      !equal(replayed, {5, 6})) {
    fail("mixed P2P/collective group was not replayed natively");
  }

  reset();
  collective = compressed(cocclOperation::AllGather, 7);
  cocclGroupEnqueue(&collective);
  if (cocclGroupPrepareEnd(true) != ncclSuccess ||
      !equal(replayed, {7})) {
    fail("native pending work did not force replay");
  }
}

void testCollectiveDrain() {
  reset();
  cocclPreparedCall collective = compressed(cocclOperation::AllGather, 8);
  cocclGroupEnqueue(&collective);
  if (cocclGroupPrepareEnd(false) != ncclSuccess ||
      cocclGroupDrain() != ncclSuccess || !equal(executed, {8})) {
    fail("collective-only group did not retain deferred execution");
  }
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
  detached = !cocclGroupHasPending();
  for (size_t i = 0; i < count; ++i) batched.push_back(calls[i].info.peer);
  return ncclSuccess;
}

int main() {
  testCompressedBatch();
  testIneligibleFallback();
  testMixedAndNativeFallback();
  testCollectiveDrain();
  std::printf("COCCL M13 group batch tests passed\n");
  return 0;
}
