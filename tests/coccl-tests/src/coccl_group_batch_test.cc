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
std::vector<bool> batchCompressed;
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
  batchCompressed.clear();
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
      !replayed.empty() || !cocclGroupHasPending() ||
      cocclGroupDrain() != ncclSuccess || !equal(batched, {3, 4}) ||
      batchCompressed != std::vector<bool>({true, false})) {
    fail("ineligible Send/Recv changed another message's protocol");
  }
}

void testMixedAndNativeFallback() {
  reset();
  cocclPreparedCall send = compressed(cocclOperation::SendRecv, 5);
  cocclPreparedCall collective = compressed(cocclOperation::AllGather, 6);
  cocclGroupEnqueue(&send);
  cocclGroupEnqueue(&collective);
  if (cocclGroupPrepareEnd(false) != ncclSuccess ||
      !replayed.empty() || cocclGroupDrain() != ncclSuccess ||
      !equal(batched, {5}) || !equal(executed, {6})) {
    fail("mixed P2P/collective group changed per-call routing");
  }

  reset();
  collective = compressed(cocclOperation::AllGather, 7);
  cocclGroupEnqueue(&collective);
  if (cocclGroupPrepareEnd(true) != ncclSuccess ||
      !replayed.empty() || cocclGroupDrain() != ncclSuccess ||
      !equal(executed, {7})) {
    fail("native pending work changed collective routing");
  }
}

void testEntirelyNative() {
  reset();
  cocclInfo first = native(cocclOperation::SendRecv, 9);
  cocclInfo second = native(cocclOperation::AllReduce, 10);
  cocclGroupEnqueueNative(&first);
  cocclGroupEnqueueNative(&second);
  if (cocclGroupPrepareEnd(false) != ncclSuccess ||
      cocclGroupHasPending() || !equal(replayed, {9, 10}) ||
      cocclGroupDrain() != ncclSuccess || !batched.empty() ||
      !executed.empty()) {
    fail("entirely native group entered a COCCL executor");
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

void testInterleavedDrain() {
  reset();
  auto gather = compressed(cocclOperation::AllGather, 11);
  auto send = compressed(cocclOperation::SendRecv, 12);
  auto reduce = compressed(cocclOperation::AllReduce, 13);
  auto recv = native(cocclOperation::SendRecv, 14);
  auto secondSend = compressed(cocclOperation::SendRecv, 15);
  cocclGroupEnqueue(&gather);
  cocclGroupEnqueue(&send);
  cocclGroupEnqueue(&reduce);
  cocclGroupEnqueueNative(&recv);
  cocclGroupEnqueue(&secondSend);
  if (cocclGroupPrepareEnd(false) != ncclSuccess ||
      cocclGroupDrain() != ncclSuccess || !equal(executed, {11, 13}) ||
      !equal(batched, {12, 14, 15}) || !replayed.empty() || !detached ||
      batchCompressed != std::vector<bool>({true, false, true})) {
    fail("interleaved group compaction changed operation order or routing");
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
  for (size_t i = 0; i < count; ++i) {
    batched.push_back(calls[i].info.peer);
    batchCompressed.push_back(calls[i].compressors.anyEnabled());
  }
  return ncclSuccess;
}

int main() {
  testCompressedBatch();
  testIneligibleFallback();
  testMixedAndNativeFallback();
  testCollectiveDrain();
  testEntirelyNative();
  testInterleavedDrain();
  std::printf("COCCL group batch tests passed\n");
  return 0;
}
