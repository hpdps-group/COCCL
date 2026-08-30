#include "core/tuning/coccl_autotune_internal.h"
#include "core/pipeline/coccl_pipeline_depth.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

namespace {

struct ResultRow {
  std::string scenario;
  std::string eligible;
  std::string selected;
  int usedModel;
  int forcedFallback;
  std::string status;
};

std::vector<ResultRow> rows;

void fail(const char* message) {
  std::fprintf(stderr, "%s\n", message);
  std::exit(1);
}

std::string candidateNames(const cocclAutotuneCandidateSet& candidates) {
  std::string result;
  for (size_t i = 0; i < candidates.count; ++i) {
    if (!result.empty()) result += ';';
    result += candidates.candidates[i].spec->name;
  }
  return result;
}

void expectDecision(const char* scenario,
                    const cocclAutotuneCandidateSet& candidates,
                    cocclAlgorithmKind requested, bool autotuneEnabled,
                    cocclAlgorithmKind expected, bool expectedUsedModel,
                    bool expectedFallback) {
  const cocclAutotuneDecision decision = cocclAutotuneChooseCandidate(
      candidates, requested, autotuneEnabled);
  if (decision.candidate == nullptr ||
      decision.candidate->algorithm != expected ||
      decision.usedModel != expectedUsedModel ||
      decision.forcedFallback != expectedFallback) {
    fail(scenario);
  }
  rows.push_back({scenario, candidateNames(candidates),
                  decision.candidate->name, (int)decision.usedModel,
                  (int)decision.forcedFallback, "PASS"});
}

void setScore(cocclAutotuneCandidateSet* candidates,
              cocclAlgorithmKind algorithm, double scoreUs) {
  cocclAutotuneCandidate* candidate =
      cocclAutotuneFindCandidate(candidates, algorithm);
  if (candidate == nullptr) fail("missing candidate while setting score");
  candidate->scoreUs = scoreUs;
}

void testEligibilityAndFallback() {
  const cocclAutotuneEligibility flat = {false, true};
  const cocclAutotuneEligibility hierarchical = {true, true};
  const cocclAutotuneEligibility tail = {true, false};

  const cocclAutotuneCandidateSet rsFlat =
      cocclAutotuneBuildCandidates(cocclOperation::ReduceScatter, flat);
  const cocclAutotuneCandidateSet rsHierarchical =
      cocclAutotuneBuildCandidates(
          cocclOperation::ReduceScatter, hierarchical);
  const cocclAutotuneCandidateSet arFlat =
      cocclAutotuneBuildCandidates(cocclOperation::AllReduce, flat);
  const cocclAutotuneCandidateSet arHierarchical =
      cocclAutotuneBuildCandidates(cocclOperation::AllReduce, hierarchical);
  const cocclAutotuneCandidateSet arTail =
      cocclAutotuneBuildCandidates(cocclOperation::AllReduce, tail);

  if (rsFlat.count != 1 || rsHierarchical.count != 2 ||
      arFlat.count != 2 || arHierarchical.count != 3 ||
      arTail.count != 0) {
    fail("candidate eligibility count mismatch");
  }
  rows.push_back({"allreduce_tail_rejects_oneshot", "", "none", 0, 0,
                  "PASS"});

  expectDecision("reducescatter_flat_heuristic", rsFlat,
                 cocclAlgorithmNone, false,
                 cocclAlgorithmReduceScatterOneShot, false, false);
  expectDecision("reducescatter_hierarchical_heuristic", rsHierarchical,
                 cocclAlgorithmNone, false,
                 cocclAlgorithmReduceScatterTwoShot, false, false);
  expectDecision("allreduce_disabled_heuristic", arHierarchical,
                 cocclAlgorithmNone, false,
                 cocclAlgorithmAllReduceTwoShot, false, false);
  expectDecision("forced_reducescatter_fallback", rsFlat,
                 cocclAlgorithmReduceScatterTwoShot, true,
                 cocclAlgorithmReduceScatterOneShot, false, true);
  expectDecision("forced_allreduce_fallback", arFlat,
                 cocclAlgorithmAllReduceTripleShot, true,
                 cocclAlgorithmAllReduceTwoShot, false, true);

  // Framed compressors use the generic decode/reduce/recompress path, so lack
  // of fused DR/DRC capabilities does not remove these communication recipes.
  expectDecision("framed_generic_reduction_candidates", rsHierarchical,
                 cocclAlgorithmNone, false,
                 cocclAlgorithmReduceScatterTwoShot, false, false);
}

void testModelSelection() {
  const cocclAutotuneEligibility hierarchical = {true, true};
  cocclAutotuneCandidateSet candidates =
      cocclAutotuneBuildCandidates(
          cocclOperation::AllReduce, hierarchical);
  setScore(&candidates, cocclAlgorithmAllReduceOneShot, 30.0);
  setScore(&candidates, cocclAlgorithmAllReduceTwoShot, 20.0);
  setScore(&candidates, cocclAlgorithmAllReduceTripleShot, 40.0);
  expectDecision("minimum_finite_cost", candidates, cocclAlgorithmNone, true,
                 cocclAlgorithmAllReduceTwoShot, true, false);

  setScore(&candidates, cocclAlgorithmAllReduceOneShot, 20.0);
  expectDecision("stable_table_order_tie", candidates,
                 cocclAlgorithmNone, true,
                 cocclAlgorithmAllReduceOneShot, true, false);

  setScore(&candidates, cocclAlgorithmAllReduceOneShot, 10.0);
  setScore(&candidates, cocclAlgorithmAllReduceTwoShot, 20.0);
  setScore(&candidates, cocclAlgorithmAllReduceTripleShot,
           std::numeric_limits<double>::infinity());
  expectDecision("partial_model_whole_fallback", candidates,
                 cocclAlgorithmNone, true,
                 cocclAlgorithmAllReduceTwoShot, false, false);
}

void testCostModel() {
  const std::vector<cocclAutotuneProfilePoint> points = {
      {100.0, 3.0}, {200.0, 5.0}, {300.0, 7.0}};
  const cocclLinearModel fitted = cocclAutotuneFitLinearModel(points);
  if (!fitted.valid || std::abs(fitted.alphaUs - 1.0) > 1e-12 ||
      std::abs(fitted.betaUsPerByte - 0.02) > 1e-12 ||
      std::abs(cocclAutotunePredict(fitted, 400.0) - 9.0) > 1e-12) {
    fail("linear model mismatch");
  }

  const cocclLinearModel sampled = cocclAutotuneFitLinearModel(
      {{1.0, 1.0}, {2.0, 2.0}, {4.0, 10.0}});
  if (!sampled.valid || sampled.sampleCount != 3 ||
      std::abs(cocclAutotunePredict(sampled, 3.0) - 6.0) > 1e-12) {
    fail("sampled latency interpolation mismatch");
  }

  const cocclSelectionPerformanceModel performance = {
      {1.0, 0.001, true}, {2.0, 0.002, true},
      {5.0, 0.005, true}, {7.0, 0.007, true}};
  const cocclCodecModel codec = {
      {3.0, 0.003, true}, 2.0, true};
  const cocclAutotuneCodecSet codecs = {
      {true, &codec}, {true, &codec}, {true, &codec}};
  for (const cocclAutotuneCandidateSpec& spec :
       kCocclAutotuneCandidateSpecs) {
    const double cost = cocclAutotuneEvaluateCost(
        spec.costKind, performance, codecs, 1024.0, 4, 2);
    if (!std::isfinite(cost) || cost <= 0.0) {
      fail("valid model returned invalid cost");
    }
  }

  const double oneShot = cocclAutotuneEvaluateCost(
      cocclAutotuneCostKind::AllReduceOneShot, performance,
      codecs, 1024.0, 4, 2);
  if (std::abs(oneShot - 34.9072) > 1e-9) {
    fail("allreduce oneshot must include intra-node and inter-node phases");
  }

  const double flatOneRank = cocclAutotuneEvaluateCost(
      cocclAutotuneCostKind::AllReduceOneShot, performance,
      codecs, 1024.0, 1, 1);
  const double flatFourRanks = cocclAutotuneEvaluateCost(
      cocclAutotuneCostKind::AllReduceOneShot, performance,
      codecs, 1024.0, 4, 1);
  if (std::abs((flatFourRanks - flatOneRank) - 4.608) > 1e-9) {
    fail("flat allreduce oneshot must decode one message per rank");
  }

  const cocclAutotuneCodecSet fusedCodecs = {
      {true, &codec}, {true, &codec}, {true, &codec}, true, true};
  const double regularTwoShot = cocclAutotuneEvaluateCost(
      cocclAutotuneCostKind::ReduceScatterTwoShot, performance,
      codecs, 1024.0, 4, 2);
  const double fusedTwoShot = cocclAutotuneEvaluateCost(
      cocclAutotuneCostKind::ReduceScatterTwoShot, performance,
      fusedCodecs, 1024.0, 4, 2);
  if (std::abs((regularTwoShot - fusedTwoShot) - 3.768) > 1e-9) {
    fail("fused DRC must not charge a second full codec pass");
  }

  const cocclCodecModel invalidCodec;
  const cocclAutotuneCodecSet invalidCodecs = {
      {true, &invalidCodec}, {true, &codec}, {true, &codec}};
  if (std::isfinite(cocclAutotuneEvaluateCost(
          cocclAutotuneCostKind::AllReduceOneShot, performance,
          invalidCodecs, 1024.0, 4, 2))) {
    fail("invalid model returned finite cost");
  }

  cocclSelectionPerformanceModel sampledPerformance = performance;
  sampledPerformance.intraP2p.sampleCount = 2;
  sampledPerformance.intraP2p.sampleBytes[0] = 256.0;
  sampledPerformance.intraP2p.sampleBytes[1] = 1024.0;
  sampledPerformance.intraP2p.sampleTimeUs[0] = 40.0;
  sampledPerformance.intraP2p.sampleTimeUs[1] = 80.0;
  sampledPerformance.interP2p = sampledPerformance.intraP2p;
  sampledPerformance.allGather = {5.0, 0.02, true};
  sampledPerformance.allToAll = {7.0, 0.001, true};
  cocclCodecModel sampledCodec = codec;
  sampledCodec.drcTime = {1.0, 0.0005, true};
  const cocclAutotuneCodecSet sampledCodecs = {
      {true, &sampledCodec}, {true, &sampledCodec},
      {true, &sampledCodec}};

  const double smallOne = cocclAutotuneEvaluateCost(
      cocclAutotuneCostKind::AllReduceOneShot, sampledPerformance,
      sampledCodecs, 512.0, 4, 1);
  const double smallTwo = cocclAutotuneEvaluateCost(
      cocclAutotuneCostKind::AllReduceTwoShot, sampledPerformance,
      sampledCodecs, 512.0, 4, 1);
  const double largeOne = cocclAutotuneEvaluateCost(
      cocclAutotuneCostKind::AllReduceOneShot, sampledPerformance,
      sampledCodecs, 4096.0, 4, 1);
  const double largeTwo = cocclAutotuneEvaluateCost(
      cocclAutotuneCostKind::AllReduceTwoShot, sampledPerformance,
      sampledCodecs, 4096.0, 4, 1);
  if (!(smallOne < smallTwo && largeTwo < largeOne)) {
    fail("single-node crossover model mismatch");
  }

  const double rsOne = cocclAutotuneEvaluateCost(
      cocclAutotuneCostKind::ReduceScatterOneShot, sampledPerformance,
      codecs, 1024.0, 4, 2);
  const double rsTwo = cocclAutotuneEvaluateCost(
      cocclAutotuneCostKind::ReduceScatterTwoShot, sampledPerformance,
      codecs, 1024.0, 4, 2);
  if (!(rsTwo < rsOne)) fail("hierarchical reducescatter model mismatch");

  const double triple = cocclAutotuneEvaluateCost(
      cocclAutotuneCostKind::AllReduceTripleShot, sampledPerformance,
      codecs, 1024.0, 4, 2);
  const double one = cocclAutotuneEvaluateCost(
      cocclAutotuneCostKind::AllReduceOneShot, sampledPerformance,
      codecs, 1024.0, 4, 2);
  if (!(triple < one)) {
    fail("hierarchical allreduce model mismatch");
  }
  rows.push_back({"cost_model_and_invalid_infinity", "all", "all", 1, 0,
                  "PASS"});
}

void testPipelineDepthSelection() {
  if (cocclChoosePipelineDepthForBytes(32ULL << 20) != 1 ||
      cocclChoosePipelineDepthForBytes(64ULL << 20) != 2 ||
      cocclChoosePipelineDepthForBytes(128ULL << 20) != 4 ||
      cocclChoosePipelineDepthForBytes(256ULL << 20) != 8 ||
      cocclChoosePipelineDepthForBytes(1ULL << 30, 4) != 4) {
    fail("pipeline depth working-set knee mismatch");
  }
  rows.push_back({"pipeline_depth_candidates", "1;2;4;8", "working-set",
                  1, 0, "PASS"});
}

}  // namespace

int main() {
  testEligibilityAndFallback();
  testModelSelection();
  testCostModel();
  testPipelineDepthSelection();

  std::printf("scenario,eligible_candidates,selected_algorithm,used_model,forced_fallback,status\n");
  for (const ResultRow& row : rows) {
    std::printf("%s,%s,%s,%d,%d,%s\n", row.scenario.c_str(),
                row.eligible.c_str(), row.selected.c_str(), row.usedModel,
                row.forcedFallback, row.status.c_str());
  }
  return 0;
}
