#include "runtime/coccl_group.h"

#include "runtime/coccl_runtime.h"

#include "comm.h"
#include "debug.h"

#include <vector>

namespace {

struct cocclDeferredGroupState {
  std::vector<cocclPreparedCall> pending;
  bool draining = false;
};

thread_local cocclDeferredGroupState cocclGroupState;

}  // namespace

bool cocclGroupHasPending() {
  return !cocclGroupState.pending.empty();
}

ncclResult_t cocclGroupEnqueue(const cocclPreparedCall* prepared) {
  if (prepared == nullptr || prepared->info.comm == nullptr ||
      prepared->descriptor == nullptr || !prepared->compressor) {
    return ncclInvalidArgument;
  }
  if (cocclGroupState.draining) return ncclInvalidUsage;

  // COCCL's deferred replay contract is one host thread per rank. Reject the
  // multi-local-GPU pattern that would otherwise replay one blocking rank
  // before its peers have entered the internal collective.
  if (!cocclGroupState.pending.empty() &&
      cocclGroupState.pending.front().info.comm->cudaDev !=
          prepared->info.comm->cudaDev) {
    WARN("COCCL grouped calls require one host thread per rank");
    return ncclInvalidUsage;
  }

  // Buffer pointers, communicator and stream follow NCCL group semantics: the
  // caller must keep them valid until ncclGroupEnd returns.
  cocclGroupState.pending.push_back(*prepared);
  return ncclSuccess;
}

ncclResult_t cocclGroupDrain() {
  if (cocclGroupState.draining || cocclGroupState.pending.empty()) {
    return ncclSuccess;
  }

  // Detach the batch before replay. Internal groups created by a primitive see
  // an empty pending queue and therefore cannot recursively drain this batch.
  std::vector<cocclPreparedCall> batch;
  batch.swap(cocclGroupState.pending);
  cocclGroupState.draining = true;

  ncclResult_t ret = ncclSuccess;
  for (const cocclPreparedCall& pending : batch) {
    ret = cocclExecutePreparedCall(&pending);
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
