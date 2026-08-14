#include "coccl_autotune_internal.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

// PCCA overlaps communication with codec work, so one phase is represented by
// the measured transfer cost plus one compress/decompress model evaluation.
double pccaCost(const cocclLinearModel& p2p, const cocclCodecModel* codec,
                double messageBytes, int ranks) {
  if (ranks <= 1) return 0.0;
  if (codec == nullptr || !codec->valid || codec->compressionRatio <= 0.0 ||
      !p2p.valid || messageBytes < 0.0) {
    return std::numeric_limits<double>::infinity();
  }
  double wireBytes = messageBytes * (double)(ranks - 1) /
                     (codec->compressionRatio * (double)ranks);
  return cocclAutotunePredict(p2p, wireBytes) +
         cocclAutotunePredict(codec->time, messageBytes);
}

double globalPccaCost(const cocclSelectionPerformanceModel& model,
                      const cocclCodecModel* codec, double messageBytes,
                      int localRanks, int nodes) {
  double cost = 0.0;
  bool hasBranch = false;
  if (localRanks > 1) {
    cost = pccaCost(model.intraP2p, codec,
                    messageBytes / (double)nodes, localRanks);
    hasBranch = true;
  }
  if (nodes > 1) {
    double interBytes = messageBytes * (double)(nodes - 1) / (double)nodes +
                        messageBytes /
                            ((double)localRanks * (double)nodes);
    int interRanks = localRanks * (nodes - 1) + 1;
    double interCost =
        pccaCost(model.interP2p, codec, interBytes, interRanks);
    // Intra- and inter-node branches overlap in the global stage.
    cost = hasBranch ? std::max(cost, interCost) : interCost;
    hasBranch = true;
  }
  return hasBranch ? cost : 0.0;
}

double reduceScatterTwoShotCost(
    const cocclSelectionPerformanceModel& model,
    const cocclCodecModel* codec, double messageBytes, int localRanks,
    int nodes) {
  return pccaCost(model.intraP2p, codec, messageBytes, localRanks) +
         pccaCost(model.interP2p, codec,
                  messageBytes / (double)localRanks, nodes);
}

double allReduceOneShotCost(const cocclSelectionPerformanceModel& model,
                            const cocclCodecModel* codec,
                            double messageBytes, int localRanks, int nodes) {
  double cost = 0.0;
  bool hasBranch = false;
  if (localRanks > 1) {
    cost = pccaCost(model.intraP2p, codec,
                    messageBytes * (double)localRanks, localRanks);
    hasBranch = true;
  }
  if (nodes > 1) {
    double interBytes =
        messageBytes * (double)localRanks * (double)(nodes - 1);
    int interRanks = localRanks * (nodes - 1) + 1;
    double interCost =
        pccaCost(model.interP2p, codec, interBytes, interRanks);
    cost = hasBranch ? std::max(cost, interCost) : interCost;
    hasBranch = true;
  }
  return hasBranch ? cost : 0.0;
}

}  // namespace

cocclLinearModel cocclAutotuneFitLinearModel(
    const std::vector<cocclAutotuneProfilePoint>& points) {
  cocclLinearModel model = {};
  if (points.size() < 2) return model;

  double meanX = 0.0;
  double meanY = 0.0;
  for (const cocclAutotuneProfilePoint& point : points) {
    if (!std::isfinite(point.bytes) || !std::isfinite(point.timeUs) ||
        point.bytes <= 0.0 || point.timeUs <= 0.0) {
      return model;
    }
    meanX += point.bytes;
    meanY += point.timeUs;
  }
  meanX /= (double)points.size();
  meanY /= (double)points.size();

  double covariance = 0.0;
  double variance = 0.0;
  for (const cocclAutotuneProfilePoint& point : points) {
    double deltaX = point.bytes - meanX;
    covariance += deltaX * (point.timeUs - meanY);
    variance += deltaX * deltaX;
  }
  if (!(variance > 0.0) || !std::isfinite(variance)) return model;

  // Negative latency or bandwidth coefficients are not physically meaningful
  // and make extrapolation unstable outside the sampled byte range.
  model.betaUsPerByte = std::max(0.0, covariance / variance);
  model.alphaUs = std::max(0.0, meanY - model.betaUsPerByte * meanX);
  model.valid = std::isfinite(model.alphaUs) &&
                std::isfinite(model.betaUsPerByte);
  return model;
}

double cocclAutotunePredict(const cocclLinearModel& model, double bytes) {
  if (!model.valid || !std::isfinite(bytes) || bytes < 0.0) {
    return std::numeric_limits<double>::infinity();
  }
  return std::max(0.0, model.alphaUs + model.betaUsPerByte * bytes);
}

double cocclAutotuneEvaluateCost(
    cocclAutotuneCostKind costKind,
    const cocclSelectionPerformanceModel& model,
    const cocclCodecModel* codec, double messageBytes, int localRanks,
    int nodes) {
  switch (costKind) {
    case cocclAutotuneCostKind::ReduceScatterOneShot:
      return globalPccaCost(model, codec, messageBytes, localRanks, nodes);
    case cocclAutotuneCostKind::ReduceScatterTwoShot:
      return reduceScatterTwoShotCost(
          model, codec, messageBytes, localRanks, nodes);
    case cocclAutotuneCostKind::AllReduceOneShot:
      return allReduceOneShotCost(
          model, codec, messageBytes, localRanks, nodes);
    case cocclAutotuneCostKind::AllReduceTwoShot: {
      // TwoShot is a global ReduceScatter followed by a global AllGather.
      double reduceScatter = globalPccaCost(
          model, codec, messageBytes, localRanks, nodes);
      double allGather = globalPccaCost(
          model, codec, messageBytes, localRanks, nodes);
      return reduceScatter + allGather;
    }
    case cocclAutotuneCostKind::AllReduceTripleShot:
      // TripleShot replaces the first global phase with hierarchical
      // ReduceScatter and retains the final global AllGather.
      return reduceScatterTwoShotCost(
                 model, codec, messageBytes, localRanks, nodes) +
             globalPccaCost(
                 model, codec, messageBytes, localRanks, nodes);
  }
  return std::numeric_limits<double>::infinity();
}
