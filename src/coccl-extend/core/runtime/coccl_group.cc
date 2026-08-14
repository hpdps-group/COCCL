#include "runtime/coccl_group.h"

#include "core/runtime/coccl_group_internal.h"
#include "core/runtime/coccl_prepared_call.h"
#include "core/runtime/coccl_primitive_dispatch.h"
#include "runtime/coccl_runtime.h"

#include <vector>

namespace {

struct cocclDeferredGroupState {
  std::vector<cocclPreparedCall> pending;
};

thread_local cocclDeferredGroupState cocclGroupState;

}  // namespace

bool cocclGroupHasPending() {
  return !cocclGroupState.pending.empty();
}

ncclResult_t cocclGroupEnqueue(const cocclPreparedCall* prepared) {
  // Buffer pointers, communicator and stream follow NCCL group semantics: the
  // caller must keep them valid until ncclGroupEnd returns.
  cocclGroupState.pending.push_back(*prepared);
  return ncclSuccess;
}

ncclResult_t cocclGroupEnqueueNative(const cocclInfo* info) {
  cocclPreparedCall pending;
  pending.info = *info;
  cocclGroupState.pending.push_back(pending);
  return ncclSuccess;
}

ncclResult_t cocclGroupPrepareEnd(bool nativePending) {
  bool hasSendRecv = false;
  bool hasCollective = false;
  bool replayNative = nativePending;
  for (const cocclPreparedCall& pending : cocclGroupState.pending) {
    hasSendRecv |= pending.info.operation == cocclOperation::SendRecv;
    hasCollective |= pending.info.operation != cocclOperation::SendRecv;
    replayNative |= !pending.compressor;
  }
  replayNative |= hasSendRecv && hasCollective;
  if (!replayNative) return ncclSuccess;

  std::vector<cocclPreparedCall> batch;
  batch.swap(cocclGroupState.pending);
  for (const cocclPreparedCall& pending : batch) {
    ncclResult_t ret = cocclReplayNativeCall(pending.info);
    if (ret != ncclSuccess) return ret;
  }
  return ncclSuccess;
}

ncclResult_t cocclGroupDrain() {
  if (cocclGroupState.pending.empty()) return ncclSuccess;

  // Detach the batch before replay. Internal groups created by a primitive see
  // an empty pending queue and therefore cannot recursively drain this batch.
  std::vector<cocclPreparedCall> batch;
  batch.swap(cocclGroupState.pending);

  if (batch.front().info.operation == cocclOperation::SendRecv) {
    return cocclExecuteSendRecvBatch(batch.data(), batch.size());
  }

  ncclResult_t ret = ncclSuccess;
  for (const cocclPreparedCall& pending : batch) {
    ret = cocclExecutePreparedCall(&pending);
    if (ret != ncclSuccess) break;
  }

  return ret;
}

void cocclGroupAbort() {
  // Keep vector capacity for the next group; group-heavy applications should
  // not repeatedly allocate host metadata.
  cocclGroupState.pending.clear();
}
