#include "core/training/coccl_training_classifier.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <utility>
#include <vector>

namespace {

constexpr size_t kMinimumCycleEvents = 8;
constexpr double kCycleMatchThreshold = 0.90;
constexpr double kRoleEvidenceThreshold = 0.60;
constexpr double kOverlapPatternThreshold = 0.80;
constexpr double kRoleConflictMargin = 0.15;

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

static uint64_t mixToken(uint64_t value) {
  value ^= value >> 30;
  value *= 0xbf58476d1ce4e5b9ULL;
  value ^= value >> 27;
  value *= 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

static uint64_t canonicalToken(const cocclTrainingTraceEvent& event) {
  uint64_t token = mixToken(event.communicatorId);
  token ^= mixToken((uint64_t)(unsigned int)event.operation + 0x9e3779b9U);
  token ^= mixToken((uint64_t)event.logicalBytes + 0x9e3779b97f4a7c15ULL);
  return token;
}

static bool sameCanonicalEvent(const cocclTrainingTraceEvent& lhs,
                               const cocclTrainingTraceEvent& rhs) {
  return lhs.communicatorId == rhs.communicatorId &&
         lhs.operation == rhs.operation &&
         lhs.logicalBytes == rhs.logicalBytes;
}

template <typename T>
static T median(std::vector<T> values) {
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

static double boundaryGapScore(
    const std::vector<cocclTrainingTraceEvent>& events,
    size_t start, size_t period, int iterations) {
  if (events.size() < 2 || period == 0) return 1.0;

  std::vector<uint64_t> allGaps;
  allGaps.reserve(events.size() - 1);
  for (size_t i = 1; i < events.size(); ++i) {
    if (events[i].timestampNs >= events[i - 1].timestampNs) {
      allGaps.push_back(events[i].timestampNs - events[i - 1].timestampNs);
    }
  }
  uint64_t normalGap = median(allGaps);
  if (normalGap == 0) normalGap = 1;

  std::vector<uint64_t> boundaryGaps;
  for (int iteration = 1; iteration < iterations; ++iteration) {
    size_t boundary = start + (size_t)iteration * period;
    if (boundary < events.size() &&
        events[boundary].timestampNs >= events[boundary - 1].timestampNs) {
      boundaryGaps.push_back(events[boundary].timestampNs -
                             events[boundary - 1].timestampNs);
    }
  }
  if (boundaryGaps.empty()) return 1.0;
  return (double)median(boundaryGaps) / (double)normalGap;
}

static bool candidateHasWork(
    const std::vector<cocclTrainingTraceEvent>& events,
    size_t begin, size_t end) {
  for (size_t i = begin; i < end; ++i) {
    if (isObservedOperation(events[i].operation)) return true;
  }
  return false;
}

static double candidateSimilarity(
    const std::vector<cocclTrainingTraceEvent>& events,
    size_t start, size_t period, int iterations) {
  uint64_t matches = 0;
  uint64_t comparisons = 0;
  for (int iteration = 1; iteration < iterations; ++iteration) {
    size_t current = start + (size_t)iteration * period;
    for (size_t offset = 0; offset < period; ++offset) {
      matches += sameCanonicalEvent(events[start + offset],
                                    events[current + offset]);
      comparisons++;
    }
  }
  return comparisons == 0 ? 0.0 : (double)matches / (double)comparisons;
}

using EventSignature = std::pair<int, size_t>;
using IterationPattern = std::vector<EventSignature>;

struct cocclTrainingCommStats {
  cocclTrainingTraceComm descriptor;
  uint64_t calls = 0;
  uint64_t collectiveCalls = 0;
  uint64_t p2pCalls = 0;
  uint64_t allGatherCalls = 0;
  uint64_t reduceScatterCalls = 0;
  uint64_t allReduceCalls = 0;
  size_t medianBytes = 0;
  double callsPerIteration = 0.0;
  double p2pShare = 0.0;
  double peerConcentration = 0.0;
  double sizeConsistency = 0.0;
  double patternSupport = 0.0;
  double overlapPatternSupport = 0.0;
  double ratioSupport = 0.0;
  double representativeAgToRsRatio = 0.0;
  bool tpSizeConsistent = false;
};

struct cocclTrainingEvidence {
  double dp = 0.0;
  double pp = 0.0;
  double tp = 0.0;
  double orderAsDp = 0.0;
  double orderAsTp = 0.0;
};

static bool ratioNear(double value, double target) {
  return target > 0.0 && std::fabs(value - target) / target <= 0.20;
}

static void buildCommStats(
    const cocclTrainingTraceComm& descriptor,
    const std::vector<cocclTrainingTraceEvent>& events,
    const std::vector<cocclTrainingIterationRange>& iterations,
    cocclTrainingCommStats* stats) {
  *stats = {};
  stats->descriptor = descriptor;

  std::map<size_t, uint64_t> byteCounts;
  std::map<int, uint64_t> peerCounts;
  std::vector<size_t> collectiveBytes;
  std::map<IterationPattern, uint64_t> patternCounts;
  std::map<IterationPattern, uint64_t> multiSizePatternCounts;
  uint64_t ratioMatches = 0;
  std::vector<double> matchingRatios;

  for (const cocclTrainingTraceEvent& event : events) {
    if (event.communicatorId != descriptor.communicatorId) continue;
    stats->calls++;
    if (isP2p(event.operation)) {
      stats->p2pCalls++;
      peerCounts[event.peer]++;
      continue;
    }
    if (!isCollective(event.operation)) continue;

    stats->collectiveCalls++;
    stats->allGatherCalls += event.operation == ncclFuncAllGather;
    stats->reduceScatterCalls += event.operation == ncclFuncReduceScatter;
    stats->allReduceCalls += event.operation == ncclFuncAllReduce;
    byteCounts[event.logicalBytes]++;
    collectiveBytes.push_back(event.logicalBytes);
  }

  stats->callsPerIteration = iterations.empty()
      ? 0.0 : (double)stats->calls / (double)iterations.size();
  stats->p2pShare = stats->calls == 0
      ? 0.0 : (double)stats->p2pCalls / (double)stats->calls;

  std::vector<uint64_t> peerFrequencies;
  for (const auto& peer : peerCounts) peerFrequencies.push_back(peer.second);
  std::sort(peerFrequencies.begin(), peerFrequencies.end(), std::greater<uint64_t>());
  uint64_t topTwoPeers = 0;
  for (size_t i = 0; i < std::min<size_t>(2, peerFrequencies.size()); ++i) {
    topTwoPeers += peerFrequencies[i];
  }
  stats->peerConcentration = stats->p2pCalls == 0
      ? 0.0 : (double)topTwoPeers / (double)stats->p2pCalls;
  stats->medianBytes = median(collectiveBytes);

  uint64_t modeCount = 0;
  size_t modeBytes = 0;
  for (const auto& entry : byteCounts) {
    if (entry.second > modeCount) {
      modeBytes = entry.first;
      modeCount = entry.second;
    }
  }
  double modeCoverage = stats->collectiveCalls == 0
      ? 0.0 : (double)modeCount / (double)stats->collectiveCalls;
  bool withinFivePercent = modeBytes != 0;
  for (size_t bytes : collectiveBytes) {
    double deviation = std::fabs((double)bytes - (double)modeBytes) /
                       (double)modeBytes;
    if (deviation > 0.05) {
      withinFivePercent = false;
      break;
    }
  }
  stats->sizeConsistency = withinFivePercent ? modeCoverage : modeCoverage * 0.5;

  for (const cocclTrainingIterationRange& iteration : iterations) {
    IterationPattern pattern;
    std::set<size_t> distinctSizes;
    long double allGatherBytes = 0.0;
    long double reduceScatterBytes = 0.0;
    for (size_t i = iteration.begin; i < iteration.end && i < events.size(); ++i) {
      const cocclTrainingTraceEvent& event = events[i];
      if (event.communicatorId != descriptor.communicatorId ||
          !isCollective(event.operation)) {
        continue;
      }
      pattern.push_back({(int)event.operation, event.logicalBytes});
      distinctSizes.insert(event.logicalBytes);
      if (event.operation == ncclFuncAllGather) {
        allGatherBytes += (long double)event.logicalBytes;
      } else if (event.operation == ncclFuncReduceScatter) {
        reduceScatterBytes += (long double)event.logicalBytes;
      }
    }

    if (!pattern.empty()) {
      patternCounts[pattern]++;
      if (distinctSizes.size() >= 2) multiSizePatternCounts[pattern]++;
    }
    if (allGatherBytes > 0.0 && reduceScatterBytes > 0.0) {
      double ratio = (double)(allGatherBytes / reduceScatterBytes);
      if (ratioNear(ratio, 0.5) || ratioNear(ratio, 0.25)) {
        ratioMatches++;
        matchingRatios.push_back(ratio);
      }
    }
  }

  uint64_t bestPatternCount = 0;
  for (const auto& pattern : patternCounts) {
    bestPatternCount = std::max(bestPatternCount, pattern.second);
  }
  uint64_t bestMultiSizeCount = 0;
  for (const auto& pattern : multiSizePatternCounts) {
    bestMultiSizeCount = std::max(bestMultiSizeCount, pattern.second);
  }
  double iterationCount = (double)std::max<size_t>(1, iterations.size());
  stats->patternSupport = (double)bestPatternCount / iterationCount;
  stats->overlapPatternSupport = (double)bestMultiSizeCount / iterationCount;
  stats->ratioSupport = (double)ratioMatches / iterationCount;
  stats->representativeAgToRsRatio = matchingRatios.empty()
      ? 0.0 : median(matchingRatios);
  stats->tpSizeConsistent = modeCoverage >= 0.90 && withinFivePercent &&
                            stats->patternSupport >= 0.60;
}

static double orderRelationSupport(
    uint64_t dpComm, uint64_t tpComm,
    const std::vector<cocclTrainingTraceEvent>& events,
    const std::vector<cocclTrainingIterationRange>& iterations) {
  if (iterations.empty()) return 0.0;
  uint64_t matchingIterations = 0;

  for (const cocclTrainingIterationRange& iteration : iterations) {
    bool bracket = false;
    bool tail = false;

    for (size_t i = iteration.begin; i < iteration.end && i < events.size(); ++i) {
      const cocclTrainingTraceEvent& event = events[i];
      if (event.communicatorId != dpComm ||
          event.operation != ncclFuncAllGather) {
        continue;
      }
      int tpCollectives = 0;
      for (size_t j = i + 1; j < iteration.end && j < events.size(); ++j) {
        const cocclTrainingTraceEvent& next = events[j];
        if (next.communicatorId == tpComm && isCollective(next.operation)) {
          tpCollectives++;
        }
        if (next.communicatorId == dpComm &&
            next.operation == ncclFuncReduceScatter) {
          bracket = tpCollectives >= 2;
          break;
        }
      }
      if (bracket) break;
    }

    for (size_t i = iteration.begin; !tail && i < iteration.end && i < events.size(); ++i) {
      const cocclTrainingTraceEvent& event = events[i];
      if (event.communicatorId != dpComm ||
          (event.operation != ncclFuncReduceScatter &&
           event.operation != ncclFuncAllReduce)) {
        continue;
      }
      int tpCollectives = 0;
      int examinedCollectives = 0;
      for (size_t j = i; j > iteration.begin && examinedCollectives < 8;) {
        --j;
        const cocclTrainingTraceEvent& previous = events[j];
        if (!isCollective(previous.operation)) continue;
        examinedCollectives++;
        if (previous.communicatorId == tpComm) tpCollectives++;
      }
      tail = tpCollectives >= 2;
    }

    matchingIterations += bracket || tail;
  }
  return (double)matchingIterations / (double)iterations.size();
}

static double ppInteriorSupport(
    uint64_t candidateComm,
    const std::set<uint64_t>& ppComms,
    const std::vector<cocclTrainingTraceEvent>& events,
    const std::vector<cocclTrainingIterationRange>& iterations) {
  if (ppComms.empty() || iterations.empty()) return 0.0;
  uint64_t matchingIterations = 0;

  for (const cocclTrainingIterationRange& iteration : iterations) {
    bool haveBoundary = false;
    int candidateCollectives = 0;
    bool foundBurst = false;
    for (size_t i = iteration.begin; i < iteration.end && i < events.size(); ++i) {
      const cocclTrainingTraceEvent& event = events[i];
      if (ppComms.count(event.communicatorId) != 0 && isP2p(event.operation)) {
        if (haveBoundary && candidateCollectives >= 2) foundBurst = true;
        haveBoundary = true;
        candidateCollectives = 0;
      } else if (haveBoundary && event.communicatorId == candidateComm &&
                 isCollective(event.operation)) {
        candidateCollectives++;
      }
    }
    matchingIterations += foundBurst;
  }
  return (double)matchingIterations / (double)iterations.size();
}

}  // namespace

bool cocclTrainingDetectIterations(
    const std::vector<cocclTrainingTraceEvent>& events,
    int targetIterations,
    std::vector<cocclTrainingIterationRange>* iterations) {
  if (iterations == nullptr) return false;
  iterations->clear();
  if (targetIterations < 2 ||
      events.size() < (size_t)targetIterations * kMinimumCycleEvents) {
    return false;
  }

  size_t eventCount = events.size();
  size_t maximumPeriod = eventCount / (size_t)targetIterations;
  constexpr uint64_t base = 0x9e3779b185ebca87ULL;
  std::vector<uint64_t> powers(eventCount + 1, 1);
  std::vector<uint64_t> prefixes(eventCount + 1, 0);
  for (size_t i = 0; i < eventCount; ++i) {
    powers[i + 1] = powers[i] * base;
    prefixes[i + 1] = prefixes[i] * base + canonicalToken(events[i]);
  }
  auto blockHash = [&](size_t begin, size_t length) {
    return prefixes[begin + length] - prefixes[begin] * powers[length];
  };

  size_t bestPeriod = 0;
  size_t bestStart = 0;
  double bestGapScore = -1.0;

  // Exact repeated schedules are common in training. Rolling hashes make the
  // normal path O(number of candidate periods) rather than quadratic.
  for (size_t period = kMinimumCycleEvents; period <= maximumPeriod; ++period) {
    size_t start = eventCount - (size_t)targetIterations * period;
    if (!candidateHasWork(events, start, start + period)) continue;
    uint64_t reference = blockHash(start, period);
    bool exact = true;
    for (int iteration = 1; iteration < targetIterations; ++iteration) {
      if (blockHash(start + (size_t)iteration * period, period) != reference) {
        exact = false;
        break;
      }
    }
    if (!exact) continue;
    double gapScore = boundaryGapScore(events, start, period, targetIterations);
    if (gapScore > bestGapScore + 1e-9 ||
        (std::fabs(gapScore - bestGapScore) <= 1e-9 && period > bestPeriod)) {
      bestPeriod = period;
      bestStart = start;
      bestGapScore = gapScore;
    }
  }

  // Real frameworks can insert an occasional bookkeeping collective. If no
  // exact cycle exists, inspect candidates with matching boundaries and accept
  // a schedule whose canonical events agree by at least 90 percent.
  if (bestPeriod == 0) {
    for (size_t period = kMinimumCycleEvents; period <= maximumPeriod; ++period) {
      size_t start = eventCount - (size_t)targetIterations * period;
      bool boundariesMatch = true;
      for (int iteration = 1; iteration < targetIterations; ++iteration) {
        size_t current = start + (size_t)iteration * period;
        if (!sameCanonicalEvent(events[start], events[current]) ||
            !sameCanonicalEvent(events[start + period - 1],
                                events[current + period - 1])) {
          boundariesMatch = false;
          break;
        }
      }
      if (!boundariesMatch) continue;

      size_t sampleCount = std::min<size_t>(16, period);
      uint64_t sampleMatches = 0;
      uint64_t sampleComparisons = 0;
      for (int iteration = 1; iteration < targetIterations; ++iteration) {
        size_t current = start + (size_t)iteration * period;
        for (size_t sample = 0; sample < sampleCount; ++sample) {
          size_t offset = sample * period / sampleCount;
          sampleMatches += sameCanonicalEvent(events[start + offset],
                                              events[current + offset]);
          sampleComparisons++;
        }
      }
      if (sampleComparisons == 0 ||
          (double)sampleMatches / (double)sampleComparisons < 0.85) {
        continue;
      }
      double similarity = candidateSimilarity(events, start, period,
                                               targetIterations);
      if (similarity < kCycleMatchThreshold) continue;
      double gapScore = boundaryGapScore(events, start, period, targetIterations);
      if (gapScore > bestGapScore + 1e-9 ||
          (std::fabs(gapScore - bestGapScore) <= 1e-9 && period > bestPeriod)) {
        bestPeriod = period;
        bestStart = start;
        bestGapScore = gapScore;
      }
    }
  }

  if (bestPeriod == 0) return false;
  iterations->reserve((size_t)targetIterations);
  for (int iteration = 0; iteration < targetIterations; ++iteration) {
    size_t begin = bestStart + (size_t)iteration * bestPeriod;
    iterations->push_back({begin, begin + bestPeriod});
  }
  return true;
}

void cocclTrainingClassifyTrace(
    const std::vector<cocclTrainingTraceComm>& communicators,
    const std::vector<cocclTrainingTraceEvent>& events,
    const std::vector<cocclTrainingIterationRange>& iterations,
    int targetIterations,
    std::vector<cocclTrainingTraceResult>* results) {
  if (results == nullptr) return;
  results->clear();

  std::map<uint64_t, cocclTrainingCommStats> statsById;
  std::map<uint64_t, cocclTrainingEvidence> evidenceById;
  double calibrationCoverage = targetIterations <= 0
      ? 0.0 : std::min(1.0, (double)iterations.size() /
                                (double)targetIterations);

  for (const cocclTrainingTraceComm& communicator : communicators) {
    cocclTrainingCommStats stats;
    buildCommStats(communicator, events, iterations, &stats);
    statsById[communicator.communicatorId] = stats;
    cocclTrainingEvidence& evidence = evidenceById[communicator.communicatorId];

    if (stats.p2pShare >= 0.70 && stats.peerConcentration >= 0.80) {
      evidence.pp = std::min(stats.p2pShare, stats.peerConcentration);
      continue;
    }
    if (stats.collectiveCalls != 0 && communicator.nNodes > 1) {
      // User-provided topology contract: TP never spans multiple nodes.
      evidence.dp = 1.0;
      continue;
    }
    // Overlap buckets need stronger repetition than the AG/RS precision-ratio
    // signal: the contract requires 8/10 matching iterations versus 6/10.
    double overlapEvidence = stats.overlapPatternSupport * calibrationCoverage;
    double ratioEvidence = stats.ratioSupport * calibrationCoverage;
    if (overlapEvidence < kOverlapPatternThreshold) overlapEvidence = 0.0;
    if (ratioEvidence < kRoleEvidenceThreshold) ratioEvidence = 0.0;
    evidence.dp = std::max(overlapEvidence, ratioEvidence);
  }

  std::set<uint64_t> ppComms;
  for (const auto& entry : evidenceById) {
    if (entry.second.pp >= 0.70) ppComms.insert(entry.first);
  }

  // Compare communicator pairs so order, rather than an absolute message-size
  // threshold, decides between node-local DP and TP.
  for (const auto& dpEntry : statsById) {
    if (evidenceById[dpEntry.first].pp >= 0.70) continue;
    for (const auto& tpEntry : statsById) {
      if (dpEntry.first == tpEntry.first ||
          evidenceById[tpEntry.first].pp >= 0.70 ||
          tpEntry.second.descriptor.nNodes > 1) {
        continue;
      }
      double relation = orderRelationSupport(dpEntry.first, tpEntry.first,
                                             events, iterations) *
                        calibrationCoverage;
      if (relation < kRoleEvidenceThreshold) continue;

      evidenceById[dpEntry.first].dp =
          std::max(evidenceById[dpEntry.first].dp, relation);
      evidenceById[dpEntry.first].orderAsDp =
          std::max(evidenceById[dpEntry.first].orderAsDp, relation);

      if (tpEntry.second.tpSizeConsistent) {
        double tpEvidence = relation;
        // The expected non-overlap pattern is fewer, larger DP calls. It is a
        // tie-breaker only; a stable order relation remains the primary proof.
        if (dpEntry.second.medianBytes >=
                (size_t)((double)tpEntry.second.medianBytes * 1.25) &&
            dpEntry.second.callsPerIteration <
                tpEntry.second.callsPerIteration) {
          tpEvidence = std::min(1.0, tpEvidence + 0.10);
        }
        evidenceById[tpEntry.first].tp =
            std::max(evidenceById[tpEntry.first].tp, tpEvidence);
        evidenceById[tpEntry.first].orderAsTp =
            std::max(evidenceById[tpEntry.first].orderAsTp, relation);
      }
    }
  }

  // PP boundaries provide a second way to recognize a dense, constant-size
  // TP burst when the DP communicator is absent or uses a different schedule.
  for (const auto& entry : statsById) {
    if (entry.second.descriptor.nNodes > 1 ||
        evidenceById[entry.first].pp >= 0.70 ||
        !entry.second.tpSizeConsistent) {
      continue;
    }
    double interior = ppInteriorSupport(entry.first, ppComms, events,
                                        iterations) * calibrationCoverage;
    evidenceById[entry.first].tp =
        std::max(evidenceById[entry.first].tp, interior);
    evidenceById[entry.first].orderAsTp =
        std::max(evidenceById[entry.first].orderAsTp, interior);
  }

  // With a known DP communicator, a much denser constant-size node-local flow
  // is a TP candidate even when framework scheduling obscures a strict tail.
  double maximumDpCallsPerIteration = 0.0;
  bool haveDpEvidence = false;
  for (const auto& entry : statsById) {
    if (evidenceById[entry.first].dp >= kRoleEvidenceThreshold) {
      haveDpEvidence = true;
      maximumDpCallsPerIteration = std::max(maximumDpCallsPerIteration,
                                            entry.second.callsPerIteration);
    }
  }
  if (haveDpEvidence) {
    for (const auto& entry : statsById) {
      if (entry.second.descriptor.nNodes == 1 &&
          entry.second.tpSizeConsistent &&
          entry.second.callsPerIteration >= 2.0 &&
          entry.second.callsPerIteration >=
              maximumDpCallsPerIteration * 1.25) {
        evidenceById[entry.first].tp = std::max(
            evidenceById[entry.first].tp, 0.60 * calibrationCoverage);
      }
    }
  }

  for (const auto& entry : statsById) {
    const cocclTrainingCommStats& stats = entry.second;
    const cocclTrainingEvidence& evidence = evidenceById[entry.first];
    cocclTrainingClassification classification;
    classification.observedCalls = stats.calls;
    classification.sizeConsistency = stats.sizeConsistency;
    classification.cycleSupport = stats.patternSupport;
    classification.overlapPatternSupport = stats.overlapPatternSupport;
    classification.orderSupport = std::max(evidence.orderAsDp,
                                           evidence.orderAsTp);
    classification.agToRsRatio = stats.representativeAgToRsRatio;
    classification.callsPerIteration = stats.callsPerIteration;
    classification.medianBytes = stats.medianBytes;
    classification.committed = true;

    if (evidence.pp >= 0.70) {
      classification.role = cocclTrainingRolePipelineParallel;
      classification.candidateRole = classification.role;
      classification.confidence = evidence.pp;
    } else {
      double dp = evidence.dp;
      double tp = stats.descriptor.nNodes > 1 ? 0.0 : evidence.tp;
      classification.candidateRole = dp >= tp
          ? (dp > 0.0 ? cocclTrainingRoleDataParallel
                      : cocclTrainingRoleUnknown)
          : cocclTrainingRoleTensorParallel;

      if (dp >= kRoleEvidenceThreshold &&
          (tp < kRoleEvidenceThreshold || dp - tp >= kRoleConflictMargin)) {
        classification.role = cocclTrainingRoleDataParallel;
        classification.confidence = dp;
      } else if (tp >= kRoleEvidenceThreshold &&
                 (dp < kRoleEvidenceThreshold ||
                  tp - dp >= kRoleConflictMargin)) {
        classification.role = cocclTrainingRoleTensorParallel;
        classification.confidence = tp;
      } else {
        classification.role = cocclTrainingRoleUnknown;
        classification.confidence = std::max(dp, tp);
      }
    }

    results->push_back({entry.first, classification});
  }
}
