#include "core/training/coccl_training_classifier.h"

#include <algorithm>
#include <cmath>
#include <set>

namespace {

constexpr size_t kMinimumCycleEvents = 2;
constexpr double kCycleMatchThreshold = 0.90;
constexpr double kMinimumBoundaryGapScore = 2.0;

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
  uint64_t normalGap = cocclTrainingMedian(allGaps);
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
  return (double)cocclTrainingMedian(boundaryGaps) / (double)normalGap;
}

static bool candidateHasWork(
    const std::vector<cocclTrainingTraceEvent>& events,
    size_t begin, size_t end) {
  for (size_t i = begin; i < end; ++i) {
    if (cocclTrainingIsObservedOperation(events[i].operation)) return true;
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

static bool candidateHasStableAgRsRatio(
    const std::vector<cocclTrainingTraceEvent>& events,
    size_t start, size_t period, int iterations) {
  std::set<uint64_t> communicators;
  for (size_t i = start; i < start + period; ++i) {
    if (events[i].operation == ncclFuncAllGather ||
        events[i].operation == ncclFuncReduceScatter) {
      communicators.insert(events[i].communicatorId);
    }
  }

  for (uint64_t communicator : communicators) {
    bool stable = true;
    for (int iteration = 0; iteration < iterations; ++iteration) {
      long double allGatherBytes = 0.0;
      long double reduceScatterBytes = 0.0;
      size_t begin = start + (size_t)iteration * period;
      for (size_t i = begin; i < begin + period; ++i) {
        const cocclTrainingTraceEvent& event = events[i];
        if (event.communicatorId != communicator) continue;
        if (event.operation == ncclFuncAllGather) {
          allGatherBytes += (long double)event.logicalBytes;
        } else if (event.operation == ncclFuncReduceScatter) {
          reduceScatterBytes += (long double)event.logicalBytes;
        }
      }
      if (allGatherBytes == 0.0 || reduceScatterBytes == 0.0) {
        stable = false;
        break;
      }
      const double ratio = (double)(allGatherBytes / reduceScatterBytes);
      if (!cocclTrainingRatioNear(ratio, 0.5) && !cocclTrainingRatioNear(ratio, 0.25)) {
        stable = false;
        break;
      }
    }
    if (stable) return true;
  }
  return false;
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

  if (bestPeriod == 0 ||
      (bestGapScore < kMinimumBoundaryGapScore &&
       !candidateHasStableAgRsRatio(
           events, bestStart, bestPeriod, targetIterations))) {
    return false;
  }
  iterations->reserve((size_t)targetIterations);
  for (int iteration = 0; iteration < targetIterations; ++iteration) {
    size_t begin = bestStart + (size_t)iteration * bestPeriod;
    iterations->push_back({begin, begin + bestPeriod});
  }
  return true;
}

