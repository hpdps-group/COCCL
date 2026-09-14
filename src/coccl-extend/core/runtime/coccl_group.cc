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

ncclResult_t cocclGroupPrepareEnd(bool /*nativePending*/) {
  for (const cocclPreparedCall& pending : pendingCalls) {
    if (pending.compressors.anyEnabled()) return ncclSuccess;
  }

  // An entirely native queue needs neither control messages nor GPU staging.
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
  size_t p2pCount = 0;
  for (size_t i = 0; i < batch.size(); ++i) {
    const cocclPreparedCall& pending = batch[i];
    if (pending.info.operation == cocclOperation::SendRecv) {
      // Keep P2P order in the detached batch, without a second allocation.
      if (p2pCount != i) batch[p2pCount] = pending;
      ++p2pCount;
      continue;
    }
    const ncclResult_t result = pending.compressors.anyEnabled()
        ? cocclExecutePreparedCall(&pending)
        : cocclReplayNativeCall(pending.info);
    if (result != ncclSuccess) return result;
  }
  return p2pCount == 0 ? ncclSuccess
      : cocclExecuteSendRecvBatch(batch.data(), p2pCount);
}

void cocclGroupAbort() {
  pendingCalls.clear();
}
