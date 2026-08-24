#ifndef COCCL_TRAINING_ASSIST_H_
#define COCCL_TRAINING_ASSIST_H_

#include "nccl.h"
#include "nccl_common.h"

#include <stddef.h>
#include <stdint.h>
struct cocclInfo;

constexpr size_t kCocclTrainingMinimumObservedBytes = 1ULL << 20;

// Training roles are COCCL-private and select role-specific compressor policy.
enum cocclTrainingRole {
  cocclTrainingRoleUnknown = 0,
  cocclTrainingRoleDataParallel,
  cocclTrainingRolePipelineParallel,
  cocclTrainingRoleTensorParallel,
  cocclTrainingRoleCount,
};

struct cocclTrainingClassification {
  cocclTrainingRole role = cocclTrainingRoleUnknown;
  cocclTrainingRole candidateRole = cocclTrainingRoleUnknown;
  double confidence = 0.0;
  double sizeConsistency = 0.0;
  double cycleSupport = 0.0;
  double overlapPatternSupport = 0.0;
  double orderSupport = 0.0;
  double agToRsRatio = 0.0;
  double callsPerIteration = 0.0;
  size_t medianBytes = 0;
  uint64_t observedCalls = 0;
  bool committed = false;
};

const char* cocclTrainingRoleName(cocclTrainingRole role);

// The assist registry is independent of compressor policy state.
bool cocclTrainingAssistEnabled();
void cocclTrainingAssistRegister(ncclComm_t comm);
void cocclTrainingAssistUnregister(ncclComm_t comm);

// A uniquely matching configured parallel size commits the role on the first
// corresponding call. Ambiguous user-visible calls of at least 1 MiB are
// observed before routing. Their role is activated at a common absolute call
// boundary so asynchronous pipeline stages cannot switch protocol separately.
void cocclTrainingAssistObserve(const cocclInfo* args, int groupDepth);
bool cocclTrainingAssistQuery(
    ncclComm_t comm, cocclTrainingClassification* classification);

#endif
