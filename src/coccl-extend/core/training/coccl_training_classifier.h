#ifndef COCCL_TRAINING_CLASSIFIER_H_
#define COCCL_TRAINING_CLASSIFIER_H_

#include "core/training/coccl_training_assist.h"

#include <stddef.h>
#include <stdint.h>
#include <vector>
struct cocclTrainingConfig;

// Trace types deliberately contain no runtime-owned pointers. This keeps the
// classifier deterministic and directly testable with synthetic schedules.
struct cocclTrainingTraceComm {
  uint64_t communicatorId = 0;
  uint64_t commHash = 0;
  int rank = 0;
  int nRanks = 0;
  int nNodes = 0;
  int localRanks = 0;
};

struct cocclTrainingTraceEvent {
  uint64_t sequence = 0;
  uint64_t communicatorId = 0;
  ncclFunc_t operation = ncclFuncAllReduce;
  size_t logicalBytes = 0;
  ncclDataType_t datatype = ncclFloat32;
  int peer = -1;
  uint64_t timestampNs = 0;
  int groupDepth = 0;
};

struct cocclTrainingIterationRange {
  size_t begin = 0;
  size_t end = 0;
};

struct cocclTrainingTraceResult {
  uint64_t communicatorId = 0;
  cocclTrainingClassification classification;
};

// A unique configured communicator size is enough to classify a collective
// as DP or TP. P2P is PP when its communicator matches the configured PP size.
cocclTrainingRole cocclTrainingTopologyRole(
    const cocclTrainingTraceComm& communicator,
    ncclFunc_t operation,
    const cocclTrainingConfig& config);

// Finds the longest repeated suffix with at least 90 percent token agreement.
bool cocclTrainingDetectIterations(
    const std::vector<cocclTrainingTraceEvent>& events,
    int targetIterations,
    std::vector<cocclTrainingIterationRange>* iterations);

// Pure CPU classifier shared by runtime observation and host-only tests.
void cocclTrainingClassifyTrace(
    const std::vector<cocclTrainingTraceComm>& communicators,
    const std::vector<cocclTrainingTraceEvent>& events,
    const std::vector<cocclTrainingIterationRange>& iterations,
    int targetIterations, const cocclTrainingConfig& config,
    std::vector<cocclTrainingTraceResult>* results);

#endif
