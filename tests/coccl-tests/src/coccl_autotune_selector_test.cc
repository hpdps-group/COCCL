#include "tuning/coccl_autotune_internal.h"

#include <cmath>
#include <limits>
#include <stdio.h>
#include <vector>

namespace {

int expectDecision(const char* scenario,
                   const cocclAutotuneCandidateSet& candidates,
                   cocclAlgorithmKind requested, bool autotuneEnabled,
                   cocclAlgorithmKind expected, bool expectedUsedModel,
                   bool expectedFallback) {
  cocclAutotuneDecision decision = cocclAutotuneChooseCandidate(
      candidates, requested, autotuneEnabled);
  if (decision.candidate == nullptr ||
      decision.candidate->algorithm != expected ||
      decision.usedModel != expectedUsedModel ||
      decision.forcedFallback != expectedFallback) {
    fprintf(stderr,
            "%s: expected algorithm=%d model=%d fallback=%d, got "
            "algorithm=%d model=%d fallback=%d\n",
            scenario, (int)expected, (int)expectedUsedModel,
            (int)expectedFallback,
            decision.candidate == nullptr ? -1
                                          : (int)decision.candidate->algorithm,
            (int)decision.usedModel, (int)decision.forcedFallback);
    return 1;
  }
  return 0;
}

int setScore(cocclAutotuneCandidateSet* candidates,
             cocclAlgorithmKind algorithm, double scoreUs) {
  cocclAutotuneCandidate* candidate =
      cocclAutotuneFindCandidate(candidates, algorithm);
  if (candidate == nullptr) return 1;
  candidate->scoreUs = scoreUs;
  return 0;
}

int testEligibilityAndHeuristics() {
  const cocclAutotuneEligibility flat = {false, true};
  const cocclAutotuneEligibility hierarchical = {true, true};

  cocclAutotuneCandidateSet rsFlat =
      cocclAutotuneBuildCandidates(cocclOperation::ReduceScatter, flat);
  if (rsFlat.count != 1 ||
      expectDecision("flat ReduceScatter heuristic", rsFlat,
                     cocclAlgorithmNone, false,
                     cocclAlgorithmReduceScatterOneShot, false, false)) {
    return 1;
  }

  cocclAutotuneCandidateSet rsHierarchical =
      cocclAutotuneBuildCandidates(
          cocclOperation::ReduceScatter, hierarchical);
  if (rsHierarchical.count != 2 ||
      expectDecision("hierarchical ReduceScatter heuristic",
                     rsHierarchical, cocclAlgorithmNone, false,
                     cocclAlgorithmReduceScatterTwoShot, false, false)) {
    return 1;
  }

  cocclAutotuneCandidateSet arFlat =
      cocclAutotuneBuildCandidates(cocclOperation::AllReduce, flat);
  cocclAutotuneCandidateSet arHierarchical =
      cocclAutotuneBuildCandidates(
          cocclOperation::AllReduce, hierarchical);
  const cocclAutotuneEligibility invalidShape = {true, false};
  cocclAutotuneCandidateSet arInvalid =
      cocclAutotuneBuildCandidates(
          cocclOperation::AllReduce, invalidShape);
  if (arFlat.count != 2 || arHierarchical.count != 3 ||
      arInvalid.count != 0) {
    fprintf(stderr, "candidate eligibility produced unexpected counts\n");
    return 1;
  }
  return expectDecision("AllReduce heuristic", arHierarchical,
                        cocclAlgorithmNone, false,
                        cocclAlgorithmAllReduceTwoShot, false, false);
}

int testForcedFallbacks() {
  const cocclAutotuneEligibility flat = {false, true};
  cocclAutotuneCandidateSet rs =
      cocclAutotuneBuildCandidates(cocclOperation::ReduceScatter, flat);
  if (expectDecision("forced ReduceScatter fallback", rs,
                     cocclAlgorithmReduceScatterTwoShot, true,
                     cocclAlgorithmReduceScatterOneShot, false, true)) {
    return 1;
  }

  cocclAutotuneCandidateSet ar =
      cocclAutotuneBuildCandidates(cocclOperation::AllReduce, flat);
  return expectDecision("forced AllReduce fallback", ar,
                        cocclAlgorithmAllReduceTripleShot, true,
                        cocclAlgorithmAllReduceTwoShot, false, true);
}

int testModelSelection() {
  const cocclAutotuneEligibility hierarchical = {true, true};
  cocclAutotuneCandidateSet candidates =
      cocclAutotuneBuildCandidates(
          cocclOperation::AllReduce, hierarchical);
  if (setScore(&candidates, cocclAlgorithmAllReduceOneShot, 30.0) ||
      setScore(&candidates, cocclAlgorithmAllReduceTwoShot, 20.0) ||
      setScore(&candidates, cocclAlgorithmAllReduceTripleShot, 40.0) ||
      expectDecision("minimum modeled cost", candidates,
                     cocclAlgorithmNone, true,
                     cocclAlgorithmAllReduceTwoShot, true, false)) {
    return 1;
  }

  if (setScore(&candidates, cocclAlgorithmAllReduceOneShot, 20.0) ||
      expectDecision("stable modeled tie", candidates,
                     cocclAlgorithmNone, true,
                     cocclAlgorithmAllReduceOneShot, true, false)) {
    return 1;
  }

  if (setScore(&candidates, cocclAlgorithmAllReduceOneShot, 10.0) ||
      setScore(&candidates, cocclAlgorithmAllReduceTwoShot, 20.0) ||
      setScore(&candidates, cocclAlgorithmAllReduceTripleShot,
               std::numeric_limits<double>::infinity())) {
    return 1;
  }
  return expectDecision("partial model fallback", candidates,
                        cocclAlgorithmNone, true,
                        cocclAlgorithmAllReduceTwoShot, false, false);
}

int testCostModel() {
  std::vector<cocclAutotuneProfilePoint> points = {
      {100.0, 3.0},
      {200.0, 5.0},
      {300.0, 7.0},
  };
  cocclLinearModel fitted = cocclAutotuneFitLinearModel(points);
  if (!fitted.valid || std::abs(fitted.alphaUs - 1.0) > 1e-12 ||
      std::abs(fitted.betaUsPerByte - 0.02) > 1e-12 ||
      std::abs(cocclAutotunePredict(fitted, 400.0) - 9.0) > 1e-12) {
    fprintf(stderr, "linear model fitting or prediction failed\n");
    return 1;
  }

  cocclSelectionPerformanceModel performance = {
      {1.0, 0.001, true},
      {2.0, 0.002, true},
  };
  cocclCodecModel codec = {
      {3.0, 0.003, true},
      2.0,
      true,
  };
  for (const cocclAutotuneCandidateSpec& spec :
       kCocclAutotuneCandidateSpecs) {
    double cost = cocclAutotuneEvaluateCost(
        spec.costKind, performance, &codec, 1024.0, 4, 2);
    if (!std::isfinite(cost) || cost <= 0.0) {
      fprintf(stderr, "cost model failed for %s\n", spec.name);
      return 1;
    }
  }

  cocclCodecModel invalidCodec;
  double invalidCost = cocclAutotuneEvaluateCost(
      cocclAutotuneCostKind::AllReduceOneShot, performance,
      &invalidCodec, 1024.0, 4, 2);
  if (std::isfinite(invalidCost)) {
    fprintf(stderr, "invalid codec unexpectedly produced a finite cost\n");
    return 1;
  }
  return 0;
}

}  // namespace

int main() {
  if (testEligibilityAndHeuristics() || testForcedFallbacks() ||
      testModelSelection() || testCostModel()) {
    return 1;
  }
  printf("coccl autotune selector tests passed\n");
  return 0;
}
