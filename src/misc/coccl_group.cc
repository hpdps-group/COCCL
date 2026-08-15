#include "coccl_group.h"

#include "coccl_group_internal.h"
#include "coccl_prepared_call.h"
#include "coccl_runtime.h"

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

ncclResult_t cocclGroupPrepareEnd(bool nativePending) {
  if (!nativePending) return ncclSuccess;

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
  for (const cocclPreparedCall& pending : batch) {
    const ncclResult_t result = cocclExecutePreparedCall(&pending);
    if (result != ncclSuccess) return result;
  }
  return ncclSuccess;
}

void cocclGroupAbort() {
  pendingCalls.clear();
}
