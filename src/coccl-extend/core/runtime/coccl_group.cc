#include "runtime/coccl_group.h"

#include "core/runtime/coccl_group_internal.h"
#include "core/runtime/coccl_prepared_call.h"
#include "core/runtime/coccl_primitive_dispatch.h"
#include "runtime/coccl_runtime.h"

#include <vector>

namespace {

thread_local std::vector<cocclPreparedCall> pendingCalls;

}  // namespace

bool cocclGroupHasPending() {
  return !pendingCalls.empty();
}

ncclResult_t cocclGroupEnqueue(const cocclPreparedCall* prepared) {
  pendingCalls.push_back(*prepared);
  return ncclSuccess;
}

ncclResult_t cocclGroupEnqueueNative(const cocclInfo* info) {
  cocclPreparedCall pending;
  pending.info = *info;
  pendingCalls.push_back(pending);
  return ncclSuccess;
}

ncclResult_t cocclGroupPrepareEnd(bool nativePending) {
  bool hasSendRecv = false;
  bool hasCollective = false;
  bool replayNative = nativePending;
  for (const cocclPreparedCall& pending : pendingCalls) {
    hasSendRecv |= pending.info.operation == cocclOperation::SendRecv;
    hasCollective |= pending.info.operation != cocclOperation::SendRecv;
    replayNative |= !pending.compressors.anyEnabled();
  }
  replayNative |= hasSendRecv && hasCollective;
  if (!replayNative) return ncclSuccess;

  std::vector<cocclPreparedCall> batch;
  batch.swap(pendingCalls);
  for (const cocclPreparedCall& pending : batch) {
    const ncclResult_t result = cocclReplayNativeCall(pending.info);
    if (result != ncclSuccess) return result;
  }
  return ncclSuccess;
}

ncclResult_t cocclGroupDrain() {
  if (pendingCalls.empty()) return ncclSuccess;

  // Internal NCCL groups opened by an executor must not see this batch.
  std::vector<cocclPreparedCall> batch;
  batch.swap(pendingCalls);
  if (batch.front().info.operation == cocclOperation::SendRecv) {
    return cocclExecuteSendRecvBatch(batch.data(), batch.size());
  }
  for (const cocclPreparedCall& pending : batch) {
    const ncclResult_t result = cocclExecutePreparedCall(&pending);
    if (result != ncclSuccess) return result;
  }
  return ncclSuccess;
}

void cocclGroupAbort() {
  pendingCalls.clear();
}
