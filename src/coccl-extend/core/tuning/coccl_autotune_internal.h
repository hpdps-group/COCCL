#ifndef COCCL_AUTOTUNE_INTERNAL_H_
#define COCCL_AUTOTUNE_INTERNAL_H_

#include "core/tuning/coccl_autotune.h"
#include "core/runtime/coccl_prepared_call.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

struct cocclAutotuneProfilePoint {
  double bytes = 0.0;
  double timeUs = 0.0;
};

struct cocclSelectionPerformanceModel {
  cocclLinearModel intraP2p;
  cocclLinearModel interP2p;
  // Native communication stages used by flat AllReduce. AllGather is keyed
  // by bytes sent per rank; AllToAll by total bytes sent per rank.
  cocclLinearModel allGather;
  cocclLinearModel allToAll;
};

enum class cocclAutotuneCostKind : uint8_t {
  ReduceScatterOneShot,
  ReduceScatterTwoShot,
  AllReduceOneShot,
  AllReduceTwoShot,
  AllReduceTripleShot,
};

enum cocclAutotuneCandidateRequirement : uint32_t {
  cocclAutotuneRequiresNone = 0,
  cocclAutotuneRequiresHierarchical = 1u << 0,
  cocclAutotuneRequiresDivisibleByRanks = 1u << 1,
};

enum class cocclAutotuneScoreSlot : uint8_t {
  OneShot,
  TwoShot,
  TripleShot,
};

struct cocclAutotuneCandidateSpec {
  cocclOperation operation;
  cocclAlgorithmKind algorithm;
  uint32_t requirements;
  cocclAlgorithmKind unavailableFallback;
  int heuristicPriority;
  cocclAutotuneCostKind costKind;
  cocclAutotuneScoreSlot scoreSlot;
  const char* name;
};

// Table order is the stable tie break. OneShot deliberately precedes the
// other variants.
inline constexpr std::array<cocclAutotuneCandidateSpec, 5>
    kCocclAutotuneCandidateSpecs = {{
        {cocclOperation::ReduceScatter,
         cocclAlgorithmReduceScatterOneShot,
         cocclAutotuneRequiresNone,
         cocclAlgorithmReduceScatterOneShot,
         1,
         cocclAutotuneCostKind::ReduceScatterOneShot,
         cocclAutotuneScoreSlot::OneShot,
         "oneshot"},
        {cocclOperation::ReduceScatter,
         cocclAlgorithmReduceScatterTwoShot,
         cocclAutotuneRequiresHierarchical,
         cocclAlgorithmReduceScatterOneShot,
         0,
         cocclAutotuneCostKind::ReduceScatterTwoShot,
         cocclAutotuneScoreSlot::TwoShot,
         "twoshot"},
        {cocclOperation::AllReduce,
         cocclAlgorithmAllReduceOneShot,
         cocclAutotuneRequiresDivisibleByRanks,
         cocclAlgorithmAllReduceOneShot,
         1,
         cocclAutotuneCostKind::AllReduceOneShot,
         cocclAutotuneScoreSlot::OneShot,
         "oneshot"},
        {cocclOperation::AllReduce,
         cocclAlgorithmAllReduceTwoShot,
         cocclAutotuneRequiresDivisibleByRanks,
         cocclAlgorithmAllReduceTwoShot,
         0,
         cocclAutotuneCostKind::AllReduceTwoShot,
         cocclAutotuneScoreSlot::TwoShot,
         "twoshot"},
        {cocclOperation::AllReduce,
         cocclAlgorithmAllReduceTripleShot,
         cocclAutotuneRequiresHierarchical |
             cocclAutotuneRequiresDivisibleByRanks,
         cocclAlgorithmAllReduceTwoShot,
         2,
         cocclAutotuneCostKind::AllReduceTripleShot,
         cocclAutotuneScoreSlot::TripleShot,
         "tripleshot"},
    }};

struct cocclAutotuneEligibility {
  bool hierarchical = false;
  bool divisibleByRanks = false;
};

struct cocclAutotuneCandidate {
  const cocclAutotuneCandidateSpec* spec = nullptr;
  double scoreUs = std::numeric_limits<double>::infinity();
};

struct cocclAutotuneCandidateSet {
  std::array<cocclAutotuneCandidate, 3> candidates = {};
  size_t count = 0;
};

struct cocclAutotuneDecision {
  const cocclAutotuneCandidateSpec* candidate = nullptr;
  bool usedModel = false;
  bool forcedFallback = false;
};

struct cocclAutotunePhaseCodec {
  bool compressed = false;
  const cocclCodecModel* model = nullptr;
};

struct cocclAutotuneCodecSet {
  cocclAutotunePhaseCodec defaultScope;
  cocclAutotunePhaseCodec intra;
  cocclAutotunePhaseCodec inter;
  bool fusedIntraToInter = false;
  bool fusedInterToDefault = false;
  bool fusedFlatDecompressReduce = false;
};

enum class cocclAutotuneTopologyOperation : uint8_t {
  AllGather,
  AllToAll,
  ReduceScatter,
};

inline const cocclAutotuneCandidateSpec* cocclAutotuneFindCandidateSpec(
    cocclOperation operation, cocclAlgorithmKind algorithm) {
  for (const cocclAutotuneCandidateSpec& spec :
       kCocclAutotuneCandidateSpecs) {
    if (spec.operation == operation && spec.algorithm == algorithm) {
      return &spec;
    }
  }
  return nullptr;
}

inline cocclAutotuneCandidate* cocclAutotuneFindCandidate(
    cocclAutotuneCandidateSet* candidates, cocclAlgorithmKind algorithm) {
  for (size_t i = 0; i < candidates->count; ++i) {
    if (candidates->candidates[i].spec->algorithm == algorithm) {
      return &candidates->candidates[i];
    }
  }
  return nullptr;
}

inline const cocclAutotuneCandidate* cocclAutotuneFindCandidate(
    const cocclAutotuneCandidateSet& candidates,
    cocclAlgorithmKind algorithm) {
  for (size_t i = 0; i < candidates.count; ++i) {
    if (candidates.candidates[i].spec->algorithm == algorithm) {
      return &candidates.candidates[i];
    }
  }
  return nullptr;
}

inline cocclAutotuneCandidateSet cocclAutotuneBuildCandidates(
    cocclOperation operation, const cocclAutotuneEligibility& eligibility) {
  cocclAutotuneCandidateSet result;
  for (const cocclAutotuneCandidateSpec& spec :
       kCocclAutotuneCandidateSpecs) {
    if (spec.operation != operation) continue;
    if ((spec.requirements & cocclAutotuneRequiresHierarchical) != 0 &&
        !eligibility.hierarchical) {
      continue;
    }
    if ((spec.requirements & cocclAutotuneRequiresDivisibleByRanks) != 0 &&
        !eligibility.divisibleByRanks) {
      continue;
    }
    result.candidates[result.count++].spec = &spec;
  }
  return result;
}

inline const cocclAutotuneCandidate* cocclAutotuneHeuristicCandidate(
    const cocclAutotuneCandidateSet& candidates) {
  if (candidates.count == 0) return nullptr;
  const cocclAutotuneCandidate* selected = &candidates.candidates[0];
  for (size_t i = 1; i < candidates.count; ++i) {
    if (candidates.candidates[i].spec->heuristicPriority <
        selected->spec->heuristicPriority) {
      selected = &candidates.candidates[i];
    }
  }
  return selected;
}

inline cocclAutotuneDecision cocclAutotuneChooseCandidate(
    const cocclAutotuneCandidateSet& candidates,
    cocclAlgorithmKind requested, bool autotuneEnabled) {
  cocclAutotuneDecision decision;
  if (requested != cocclAlgorithmNone) {
    const cocclAutotuneCandidate* selected =
        cocclAutotuneFindCandidate(candidates, requested);
    if (selected != nullptr) {
      decision.candidate = selected->spec;
      return decision;
    }
    const cocclAutotuneCandidateSpec* requestedSpec =
        cocclAutotuneFindCandidateSpec(
            candidates.count == 0 ? cocclOperation::Count
                                  : candidates.candidates[0].spec->operation,
            requested);
    if (requestedSpec == nullptr) return decision;
    selected = cocclAutotuneFindCandidate(
        candidates, requestedSpec->unavailableFallback);
    if (selected != nullptr) {
      decision.candidate = selected->spec;
      decision.forcedFallback = true;
    }
    return decision;
  }

  if (autotuneEnabled && candidates.count > 0) {
    bool allScoresValid = true;
    const cocclAutotuneCandidate* selected = &candidates.candidates[0];
    for (size_t i = 0; i < candidates.count; ++i) {
      const cocclAutotuneCandidate& candidate = candidates.candidates[i];
      if (!std::isfinite(candidate.scoreUs)) allScoresValid = false;
      if (candidate.scoreUs < selected->scoreUs) selected = &candidate;
    }
    if (allScoresValid) {
      decision.candidate = selected->spec;
      decision.usedModel = true;
      return decision;
    }
  }

  const cocclAutotuneCandidate* heuristic =
      cocclAutotuneHeuristicCandidate(candidates);
  if (heuristic != nullptr) decision.candidate = heuristic->spec;
  return decision;
}

inline const char* cocclAutotuneAlgorithmName(
    cocclAlgorithmKind algorithm) {
  for (const cocclAutotuneCandidateSpec& spec :
       kCocclAutotuneCandidateSpecs) {
    if (spec.algorithm == algorithm) return spec.name;
  }
  return "none";
}

cocclLinearModel cocclAutotuneFitLinearModel(
    const std::vector<cocclAutotuneProfilePoint>& points);
double cocclAutotunePredict(const cocclLinearModel& model, double bytes);
double cocclAutotuneEvaluateCost(
    cocclAutotuneCostKind costKind,
    const cocclSelectionPerformanceModel& model,
    const cocclAutotuneCodecSet& codecs, double messageBytes, int localRanks,
    int nodes);

cocclSelectionPerformanceModel cocclAutotuneSnapshotPerformanceModel(
    ncclComm_t ownerComm, ncclComm_t intraComm, ncclComm_t interComm,
    ncclComm_t gatherComm);
void cocclAutotuneSnapshotCodecModels(
    void* defaultCompressor, void* intraCompressor, void* interCompressor,
    ncclDataType_t datatype, cocclCodecModel* defaultModel,
    cocclCodecModel* intraModel, cocclCodecModel* interModel);
cocclCodecModel cocclAutotuneSnapshotCodecModel(
    void* compressor, ncclDataType_t datatype);
cocclLinearModel cocclAutotuneSnapshotTopologyStageModel(
    ncclComm_t comm, cocclAutotuneTopologyOperation operation);
void cocclAutotuneTopologyCommDestroy(ncclComm_t comm);

#endif
