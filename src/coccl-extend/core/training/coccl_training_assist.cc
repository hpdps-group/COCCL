#include "core/training/coccl_training_assist.h"

#include "core/config/coccl_config.h"
#include "runtime/coccl_runtime.h"
#include "core/training/coccl_training_classifier.h"
#include "collectives.h"
#include "comm.h"
#include "debug.h"

#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <pthread.h>
#include <string>
#include <time.h>
#include <vector>

namespace {

constexpr size_t kMinimumCycleEvents = 2;
constexpr size_t kLoggedEventsPerIteration = 32;

struct cocclTrainingAssistCommState {
  ncclComm_t comm = nullptr;
  cocclTrainingTraceComm descriptor;
  cocclTrainingClassification classification;
  uint64_t observedCalls = 0;
  uint64_t routingActivationCall = 0;
};

pthread_mutex_t cocclTrainingAssistLock = PTHREAD_MUTEX_INITIALIZER;
uint64_t cocclTrainingNextCommId = 1;
uint64_t cocclTrainingNextSequence = 1;

// The detector owns communicator state independently of cocclComm. Events use
// the stable numeric ID, while the pointer is only a live-registry lookup key.
std::map<ncclComm_t, std::unique_ptr<cocclTrainingAssistCommState>>
    cocclTrainingAssistComms;
std::vector<cocclTrainingTraceEvent> cocclTrainingEvents;

static bool isCollective(ncclFunc_t operation) {
  return operation == ncclFuncAllGather ||
         operation == ncclFuncReduceScatter ||
         operation == ncclFuncAllReduce;
}

static bool isP2p(ncclFunc_t operation) {
  return operation == ncclFuncSend || operation == ncclFuncRecv;
}

static bool isObservedOperation(ncclFunc_t operation) {
  return isCollective(operation) || isP2p(operation);
}

static const char* operationName(ncclFunc_t operation) {
  switch (operation) {
    case ncclFuncAllGather:
      return "AG";
    case ncclFuncReduceScatter:
      return "RS";
    case ncclFuncAllReduce:
      return "AR";
    case ncclFuncSend:
      return "Send";
    case ncclFuncRecv:
      return "Recv";
    default:
      return "Other";
  }
}

static bool checkedLogicalBytes(const cocclInfo* args, size_t* bytes) {
  if (args == nullptr || args->comm == nullptr || bytes == nullptr) return false;
  const cocclOperationDescriptor* descriptor =
      cocclGetOperationDescriptor(args->operation);
  if (descriptor == nullptr) return false;
  int datatypeBytes = ncclTypeSize(args->datatype);
  if (datatypeBytes <= 0 ||
      args->count > std::numeric_limits<size_t>::max() /
                        (size_t)datatypeBytes) {
    return false;
  }

  size_t result = args->count * (size_t)datatypeBytes;
  if (cocclOperationHasTrait(
          descriptor, cocclOperationTraitScaleBytesByRanks)) {
    if (args->comm->nRanks <= 0 ||
        result > std::numeric_limits<size_t>::max() /
                     (size_t)args->comm->nRanks) {
      return false;
    }
    result *= (size_t)args->comm->nRanks;
  }
  *bytes = result;
  return true;
}

static bool allCommsCommittedLocked() {
  for (const auto& entry : cocclTrainingAssistComms) {
    if (!entry.second->classification.committed) return false;
  }
  return true;
}

static void releaseTraceLocked() {
  std::vector<cocclTrainingTraceEvent>().swap(cocclTrainingEvents);
}

static void logIterationSequencesLocked(
    const std::vector<cocclTrainingIterationRange>& iterations) {
  for (const auto& commEntry : cocclTrainingAssistComms) {
    const cocclTrainingAssistCommState* state = commEntry.second.get();
    for (size_t iterationIndex = 0; iterationIndex < iterations.size();
         ++iterationIndex) {
      const cocclTrainingIterationRange& iteration = iterations[iterationIndex];
      std::string sequence;
      size_t matchingEvents = 0;
      size_t loggedEvents = 0;
      for (size_t eventIndex = iteration.begin;
           eventIndex < iteration.end && eventIndex < cocclTrainingEvents.size();
           ++eventIndex) {
        const cocclTrainingTraceEvent& event = cocclTrainingEvents[eventIndex];
        if (event.communicatorId != state->descriptor.communicatorId) continue;
        matchingEvents++;
        if (loggedEvents >= kLoggedEventsPerIteration) continue;
        if (!sequence.empty()) sequence += ',';
        sequence += operationName(event.operation);
        sequence += ':';
        sequence += std::to_string(event.logicalBytes);
        loggedEvents++;
      }
      if (matchingEvents == 0) continue;
      if (matchingEvents > loggedEvents) {
        sequence += ",...+";
        sequence += std::to_string(matchingEvents - loggedEvents);
      }
      INFO(NCCL_TUNING,
           "COCCL training comm=%p hash=%llu iteration=%zu/%zu sequence=[%s]",
           state->comm,
           (unsigned long long)state->descriptor.commHash,
           iterationIndex + 1, iterations.size(), sequence.c_str());
    }
  }
}

static void commitTraceLocked(
    const std::vector<cocclTrainingIterationRange>& iterations,
    int targetIterations) {
  logIterationSequencesLocked(iterations);

  std::vector<cocclTrainingTraceComm> communicators;
  communicators.reserve(cocclTrainingAssistComms.size());
  for (const auto& entry : cocclTrainingAssistComms) {
    communicators.push_back(entry.second->descriptor);
  }

  std::vector<cocclTrainingTraceResult> results;
  cocclTrainingClassifyTrace(communicators, cocclTrainingEvents, iterations,
                             targetIterations, cocclGetConfig().training,
                             &results);
  std::map<uint64_t, cocclTrainingClassification> byId;
  for (const cocclTrainingTraceResult& result : results) {
    byId[result.communicatorId] = result.classification;
  }

  for (const auto& entry : cocclTrainingAssistComms) {
    cocclTrainingAssistCommState* state = entry.second.get();
    if (state->classification.committed) continue;
    auto result = byId.find(state->descriptor.communicatorId);
    if (result != byId.end()) state->classification = result->second;
    state->classification.committed = true;
    if (state->classification.role != cocclTrainingRoleUnknown) {
      const uint64_t callsPerIteration = std::max<uint64_t>(
          1, (uint64_t)state->classification.callsPerIteration);
      state->routingActivationCall =
          2ULL * (uint64_t)targetIterations * callsPerIteration;
    }
    INFO(NCCL_TUNING,
         "COCCL training comm=%p hash=%llu rank=%d nodes=%d ranks=%d role=%s "
         "confidence=%.3f calls=%llu medianBytes=%zu sizeConsistency=%.3f "
         "cycleSupport=%.3f overlapSupport=%.3f orderSupport=%.3f AG/RS=%.3f "
         "activateCall=%llu",
         state->comm, (unsigned long long)state->descriptor.commHash,
         state->descriptor.rank, state->descriptor.nNodes,
         state->descriptor.nRanks,
         cocclTrainingRoleName(state->classification.role),
         state->classification.confidence,
         (unsigned long long)state->classification.observedCalls,
         state->classification.medianBytes,
         state->classification.sizeConsistency,
         state->classification.cycleSupport,
         state->classification.overlapPatternSupport,
         state->classification.orderSupport,
         state->classification.agToRsRatio,
         (unsigned long long)state->routingActivationCall);
  }
  if (allCommsCommittedLocked()) releaseTraceLocked();
}

static int configuredIterations() {
  return cocclGetConfig().training.observationIterations;
}

static size_t configuredMaxEvents() {
  return std::clamp(cocclGetConfig().training.maxEvents,
                    (size_t)256, (size_t)1024 * 1024);
}

static uint64_t monotonicTimeNs() {
  struct timespec now = {};
  if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) return 0;
  return (uint64_t)now.tv_sec * 1000000000ULL + (uint64_t)now.tv_nsec;
}

}  // namespace

const char* cocclTrainingRoleName(cocclTrainingRole role) {
  switch (role) {
    case cocclTrainingRoleDataParallel:
      return "DP";
    case cocclTrainingRolePipelineParallel:
      return "PP";
    case cocclTrainingRoleTensorParallel:
      return "TP";
    case cocclTrainingRoleUnknown:
    default:
      return "Unknown";
  }
}

bool cocclTrainingAssistEnabled() {
  return cocclGetConfig().runtime.mode == cocclRuntimeMode::Training;
}

void cocclTrainingAssistRegister(ncclComm_t comm) {
  if (!cocclTrainingAssistEnabled() || comm == nullptr) return;

  std::unique_ptr<cocclTrainingAssistCommState> state(
      new cocclTrainingAssistCommState());

  state->comm = comm;
  state->descriptor.commHash = comm->commHash;
  state->descriptor.rank = comm->rank;
  state->descriptor.nRanks = comm->nRanks;
  state->descriptor.nNodes = comm->nNodes;
  state->descriptor.localRanks = comm->localRanks;

  bool inserted = false;
  cocclTrainingTraceComm descriptor;
  pthread_mutex_lock(&cocclTrainingAssistLock);
  if (cocclTrainingAssistComms.find(comm) == cocclTrainingAssistComms.end()) {
    state->descriptor.communicatorId = cocclTrainingNextCommId++;
    descriptor = state->descriptor;
    cocclTrainingAssistComms.emplace(comm, std::move(state));
    inserted = true;
    if (cocclTrainingEvents.empty()) {
      cocclTrainingEvents.reserve(configuredMaxEvents());
    }
  }
  pthread_mutex_unlock(&cocclTrainingAssistLock);

  if (inserted) {
    INFO(NCCL_TUNING,
         "COCCL training registered comm=%p id=%llu hash=%llu rank=%d ranks=%d nodes=%d localRanks=%d",
         comm, (unsigned long long)descriptor.communicatorId,
         (unsigned long long)descriptor.commHash, descriptor.rank,
         descriptor.nRanks, descriptor.nNodes, descriptor.localRanks);
  }
}

void cocclTrainingAssistUnregister(ncclComm_t comm) {
  if (comm == nullptr) return;
  pthread_mutex_lock(&cocclTrainingAssistLock);
  cocclTrainingAssistComms.erase(comm);
  if (cocclTrainingAssistComms.empty()) releaseTraceLocked();
  pthread_mutex_unlock(&cocclTrainingAssistLock);
}

void cocclTrainingAssistObserve(
    const cocclInfo* args, int groupDepth) {
  if (!cocclTrainingAssistEnabled() || args == nullptr ||
      args->comm == nullptr || !isObservedOperation(args->func)) {
    return;
  }

  cocclTrainingTraceComm descriptor;
  descriptor.nRanks = args->comm->nRanks;
  const cocclTrainingRole topologyRole = cocclTrainingTopologyRole(
      descriptor, args->func, cocclGetConfig().training);

  size_t logicalBytes = 0;
  if (topologyRole == cocclTrainingRoleUnknown &&
      (!checkedLogicalBytes(args, &logicalBytes) ||
       logicalBytes < kCocclTrainingMinimumObservedBytes)) {
    return;
  }

  cocclTrainingTraceEvent event;
  event.operation = args->func;
  event.logicalBytes = logicalBytes;
  event.datatype = args->datatype;
  event.peer = args->peer;
  event.timestampNs = monotonicTimeNs();
  event.groupDepth = groupDepth;

  pthread_mutex_lock(&cocclTrainingAssistLock);
  auto state = cocclTrainingAssistComms.find(args->comm);
  if (state == cocclTrainingAssistComms.end()) {
    pthread_mutex_unlock(&cocclTrainingAssistLock);
    return;
  }
  if (topologyRole == cocclTrainingRoleUnknown) {
    state->second->observedCalls++;
  }
  if (state->second->classification.committed) {
    if (state->second->routingActivationCall != 0 &&
        state->second->observedCalls ==
            state->second->routingActivationCall) {
      INFO(NCCL_TUNING,
           "COCCL training comm=%p hash=%llu role=%s routing-ready call=%llu",
           state->second->comm,
           (unsigned long long)state->second->descriptor.commHash,
           cocclTrainingRoleName(state->second->classification.role),
           (unsigned long long)state->second->observedCalls);
    }
    pthread_mutex_unlock(&cocclTrainingAssistLock);
    return;
  }

  if (topologyRole != cocclTrainingRoleUnknown) {
    state->second->classification.role = topologyRole;
    state->second->classification.candidateRole = topologyRole;
    state->second->classification.confidence = 1.0;
    state->second->classification.committed = true;
    INFO(NCCL_TUNING,
         "COCCL training comm=%p hash=%llu rank=%d nodes=%d ranks=%d "
         "role=%s confidence=1.000 source=configured-size",
         state->second->comm,
         (unsigned long long)state->second->descriptor.commHash,
         state->second->descriptor.rank, state->second->descriptor.nNodes,
         state->second->descriptor.nRanks,
         cocclTrainingRoleName(topologyRole));
    if (allCommsCommittedLocked()) releaseTraceLocked();
    pthread_mutex_unlock(&cocclTrainingAssistLock);
    return;
  }

  event.communicatorId = state->second->descriptor.communicatorId;
  event.sequence = cocclTrainingNextSequence++;
  cocclTrainingEvents.push_back(event);

  int targetIterations = configuredIterations();
  size_t maxEvents = configuredMaxEvents();
  bool reachedLimit = cocclTrainingEvents.size() >= maxEvents;
  size_t firstDetectionPoint =
      (size_t)targetIterations * kMinimumCycleEvents;
  bool detectionPoint = cocclTrainingEvents.size() >= firstDetectionPoint;
  std::vector<cocclTrainingIterationRange> iterations;
  bool detected = detectionPoint && cocclTrainingDetectIterations(
      cocclTrainingEvents, targetIterations, &iterations);

  if (detected) {
    commitTraceLocked(iterations, targetIterations);
  } else if (reachedLimit) {
    int fallbackIterations = std::min(3, targetIterations);
    if (!cocclTrainingDetectIterations(cocclTrainingEvents,
                                       fallbackIterations, &iterations)) {
      iterations.push_back({0, cocclTrainingEvents.size()});
    }
    commitTraceLocked(iterations, targetIterations);
  }
  pthread_mutex_unlock(&cocclTrainingAssistLock);
}

bool cocclTrainingAssistQuery(
    ncclComm_t comm, cocclTrainingClassification* classification) {
  if (!cocclTrainingAssistEnabled() || comm == nullptr ||
      classification == nullptr) {
    return false;
  }

  bool committed = false;
  pthread_mutex_lock(&cocclTrainingAssistLock);
  auto state = cocclTrainingAssistComms.find(comm);
  if (state != cocclTrainingAssistComms.end() &&
      state->second->classification.committed &&
      (state->second->routingActivationCall == 0 ||
       state->second->observedCalls >=
           state->second->routingActivationCall)) {
    *classification = state->second->classification;
    committed = true;
  }
  pthread_mutex_unlock(&cocclTrainingAssistLock);
  return committed;
}
