#ifndef COCCL_AUTOTUNE_INTERNAL_H_
#define COCCL_AUTOTUNE_INTERNAL_H_

#include "coccl_autotune.h"
#include "coccl_prepared_call.h"

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
  cocclPolicyVariant policyVariant;
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
         cocclPolicyVariant::Default,
         cocclAutotuneRequiresNone,
         cocclAlgorithmReduceScatterOneShot,
         1,
         cocclAutotuneCostKind::ReduceScatterOneShot,
         cocclAutotuneScoreSlot::OneShot,
         "oneshot"},
        {cocclOperation::ReduceScatter,
         cocclAlgorithmReduceScatterTwoShot,
         cocclPolicyVariant::Hierarchical,
         cocclAutotuneRequiresHierarchical,
         cocclAlgorithmReduceScatterOneShot,
         0,
         cocclAutotuneCostKind::ReduceScatterTwoShot,
         cocclAutotuneScoreSlot::TwoShot,
         "twoshot"},
        {cocclOperation::AllReduce,
         cocclAlgorithmAllReduceOneShot,
         cocclPolicyVariant::Default,
         cocclAutotuneRequiresDivisibleByRanks,
         cocclAlgorithmAllReduceOneShot,
         1,
         cocclAutotuneCostKind::AllReduceOneShot,
         cocclAutotuneScoreSlot::OneShot,
         "oneshot"},
        {cocclOperation::AllReduce,
         cocclAlgorithmAllReduceTwoShot,
         cocclPolicyVariant::Default,
         cocclAutotuneRequiresDivisibleByRanks,
         cocclAlgorithmAllReduceTwoShot,
         0,
         cocclAutotuneCostKind::AllReduceTwoShot,
         cocclAutotuneScoreSlot::TwoShot,
         "twoshot"},
        {cocclOperation::AllReduce,
         cocclAlgorithmAllReduceTripleShot,
         cocclPolicyVariant::Hierarchical,
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
    const cocclCodecModel* codec, double messageBytes, int localRanks,
    int nodes);

cocclSelectionPerformanceModel cocclAutotuneSnapshotPerformanceModel(
    void* primary, void* secondary, cocclCodecModel* primaryModel,
    cocclCodecModel* secondaryModel);

#endif
