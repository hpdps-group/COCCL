#ifndef COCCL_TRAINING_ASSIST_H_
#define COCCL_TRAINING_ASSIST_H_

#include "nccl.h"
#include "nccl_common.h"

#include <stddef.h>
#include <stdint.h>
struct cocclInfo;

// Training roles are intentionally COCCL-private. The first implementation
// only observes and reports a role; it never changes the wire protocol.
enum cocclTrainingRole {
  cocclTrainingRoleUnknown = 0,
  cocclTrainingRoleDataParallel,
  cocclTrainingRolePipelineParallel,
  cocclTrainingRoleTensorParallel,
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

// The assist registry is independent of cocclComm. Registration is idempotent
// and allocation failure only disables classification for the affected comm.
bool cocclTrainingAssistEnabled();
void cocclTrainingAssistRegister(ncclComm_t comm);
void cocclTrainingAssistUnregister(ncclComm_t comm);

// User-visible calls are observed before COCCL routing predicates are applied,
// so disabled roles and native fallbacks still contribute to classification.
void cocclTrainingAssistObserve(const cocclInfo* args, int groupDepth);
bool cocclTrainingAssistQuery(
    ncclComm_t comm, cocclTrainingClassification* classification);

#endif
