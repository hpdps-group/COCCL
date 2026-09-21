#ifndef COCCL_TRAINING_CLASSIFIER_H_
#define COCCL_TRAINING_CLASSIFIER_H_

#include "core/training/coccl_training_assist.h"

#include <algorithm>
#include <cmath>
#include <stddef.h>
#include <stdint.h>
#include <vector>
struct cocclTrainingConfig;

inline bool cocclTrainingIsCollective(ncclFunc_t operation) {
  return operation == ncclFuncAllGather ||
         operation == ncclFuncReduceScatter ||
         operation == ncclFuncAllReduce;
}

inline bool cocclTrainingIsP2p(ncclFunc_t operation) {
  return operation == ncclFuncSend || operation == ncclFuncRecv;
}

inline bool cocclTrainingIsObservedOperation(ncclFunc_t operation) {
  return cocclTrainingIsCollective(operation) || cocclTrainingIsP2p(operation);
}

template <typename T>
inline T cocclTrainingMedian(std::vector<T> values) {
  if (values.empty()) return T{};
  size_t middle = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + middle, values.end());
  T result = values[middle];
  if ((values.size() & 1) == 0) {
    std::nth_element(values.begin(), values.begin() + middle - 1, values.end());
    result = (T)((values[middle - 1] + result) / 2);
  }
  return result;
}

inline bool cocclTrainingRatioNear(double value, double target) {
  return target > 0.0 && std::fabs(value - target) / target <= 0.20;
}


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
  uint64_t communicatorId = 0;
  ncclFunc_t operation = ncclFuncAllReduce;
  size_t logicalBytes = 0;
  int peer = -1;
  uint64_t timestampNs = 0;
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

// Time boundaries identify training iterations in the combined trace. A
// communicator-local trace instead uses its shortest repeated operation cycle,
// so rank-local timing cannot change the call at which compression activates.
bool cocclTrainingDetectIterations(
    const std::vector<cocclTrainingTraceEvent>& events,
    int targetIterations,
    std::vector<cocclTrainingIterationRange>* iterations,
    bool useTimeBoundaries = true);

// Pure CPU classifier shared by runtime observation and host-only tests.
void cocclTrainingClassifyTrace(
    const std::vector<cocclTrainingTraceComm>& communicators,
    const std::vector<cocclTrainingTraceEvent>& events,
    const std::vector<cocclTrainingIterationRange>& iterations,
    int targetIterations, const cocclTrainingConfig& config,
    std::vector<cocclTrainingTraceResult>* results);

#endif
