#include "coccl_group.h"

#include "coccl_autotune.h"
#include "coccl_runtime.h"

#include <vector>

namespace {

struct cocclPendingPrimitive {
  cocclRuntimeArgs args;
  cocclAlgorithmDecision decision;
};

struct cocclDeferredGroupState {
  std::vector<cocclPendingPrimitive> pending;
  bool draining = false;
};

thread_local cocclDeferredGroupState cocclGroupState;

}  // namespace

bool cocclGroupHasPending() {
  return !cocclGroupState.pending.empty();
}

ncclResult_t cocclGroupEnqueue(const cocclRuntimeArgs* args,
                               const cocclAlgorithmDecision* decision) {
  if (args == nullptr || args->comm == nullptr || decision == nullptr) {
    return ncclInvalidArgument;
  }
  if (cocclGroupState.draining) return ncclInvalidUsage;

  // Buffer pointers, communicator and stream follow NCCL group semantics: the
  // caller must keep them valid until ncclGroupEnd returns.
  cocclGroupState.pending.push_back({*args, *decision});
  return ncclSuccess;
}

ncclResult_t cocclGroupDrain() {
  if (cocclGroupState.draining || cocclGroupState.pending.empty()) {
    return ncclSuccess;
  }

  // Detach the batch before replay. Internal groups created by a primitive see
  // an empty pending queue and therefore cannot recursively drain this batch.
  std::vector<cocclPendingPrimitive> batch;
  batch.swap(cocclGroupState.pending);
  cocclGroupState.draining = true;

  ncclResult_t ret = ncclSuccess;
  for (const cocclPendingPrimitive& pending : batch) {
    ret = cocclExecutePrimitive(&pending.args, &pending.decision);
    if (ret != ncclSuccess) break;
  }

  cocclGroupState.draining = false;
  return ret;
}

void cocclGroupAbort() {
  // Keep vector capacity for the next group; group-heavy applications should
  // not repeatedly allocate host metadata.
  cocclGroupState.pending.clear();
}
