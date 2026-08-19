#include "coccl_autotune_internal.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

double codecTime(const cocclAutotunePhaseCodec& codec,
                 double messageBytes) {
  if (!codec.compressed) return 0.0;
  if (codec.model == nullptr || !codec.model->valid ||
      codec.model->compressionRatio <= 0.0) {
    return std::numeric_limits<double>::infinity();
  }
  return cocclAutotunePredict(codec.model->time, messageBytes);
}

double compressionRatio(const cocclAutotunePhaseCodec& codec) {
  return codec.compressed ? codec.model->compressionRatio : 1.0;
}

double pccaCost(const cocclLinearModel& p2p,
                const cocclAutotunePhaseCodec& codec,
                double messageBytes, int ranks) {
  const double codecUs = codecTime(codec, messageBytes);
  if (!std::isfinite(codecUs) || messageBytes < 0.0) {
    return std::numeric_limits<double>::infinity();
  }
  if (ranks <= 1) return codecUs;
  const double wireBytes = messageBytes /
      (compressionRatio(codec) * (double)ranks);
  return codecUs + cocclAutotunePredict(p2p, wireBytes);
}

double globalPccaCost(const cocclSelectionPerformanceModel& model,
                      const cocclAutotunePhaseCodec& codec,
                      double messageBytes,
                      int localRanks, int nodes) {
  const double codecUs = codecTime(codec, messageBytes);
  if (!std::isfinite(codecUs)) {
    return std::numeric_limits<double>::infinity();
  }
  double communicationUs = 0.0;
  bool hasBranch = false;
  if (localRanks > 1) {
    const double wireBytes = messageBytes /
        ((double)nodes * (double)localRanks * compressionRatio(codec));
    communicationUs = cocclAutotunePredict(model.intraP2p, wireBytes);
    hasBranch = true;
  }
  if (nodes > 1) {
    const double wireBytes = messageBytes * (double)(nodes - 1) /
        ((double)nodes * compressionRatio(codec));
    const double interUs = cocclAutotunePredict(model.interP2p, wireBytes);
    communicationUs = hasBranch
        ? std::max(communicationUs, interUs) : interUs;
    hasBranch = true;
  }
  return codecUs + (hasBranch ? communicationUs : 0.0);
}

double reduceScatterTwoShotCost(
    const cocclSelectionPerformanceModel& model,
    const cocclAutotuneCodecSet& codecs, double messageBytes,
    int localRanks, int nodes) {
  return pccaCost(model.intraP2p, codecs.intra,
                  messageBytes, localRanks) +
         pccaCost(model.interP2p, codecs.inter,
                  messageBytes / (double)localRanks, nodes);
}

double allReduceOneShotCost(const cocclSelectionPerformanceModel& model,
                            const cocclAutotunePhaseCodec& codec,
                            double messageBytes, int localRanks, int nodes) {
  double cost = 0.0;
  bool hasBranch = false;
  if (localRanks > 1) {
    cost = pccaCost(model.intraP2p, codec,
                    messageBytes * (double)localRanks, localRanks);
    hasBranch = true;
  }
  if (nodes > 1) {
    const double interBytes =
        messageBytes * (double)localRanks * (double)(nodes - 1);
    const int interRanks = localRanks * (nodes - 1) + 1;
    const double interCost =
        pccaCost(model.interP2p, codec, interBytes, interRanks);
    // A global AllGather traverses both the intra-node and inter-node phases;
    // unlike the independent branches in globalPccaCost, they do not overlap.
    cost = hasBranch ? cost + interCost : interCost;
    hasBranch = true;
  }
  return hasBranch ? cost : 0.0;
}

}  // namespace

cocclLinearModel cocclAutotuneFitLinearModel(
    const std::vector<cocclAutotuneProfilePoint>& points) {
  cocclLinearModel model;
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
    const double deltaX = point.bytes - meanX;
    covariance += deltaX * (point.timeUs - meanY);
    variance += deltaX * deltaX;
  }
  if (!(variance > 0.0) || !std::isfinite(variance)) return model;

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
    const cocclAutotuneCodecSet& codecs, double messageBytes, int localRanks,
    int nodes) {
  const cocclAutotunePhaseCodec& flat =
      nodes == 1 ? codecs.intra : codecs.defaultScope;
  switch (costKind) {
    case cocclAutotuneCostKind::ReduceScatterOneShot:
      return globalPccaCost(model, flat, messageBytes, localRanks, nodes);
    case cocclAutotuneCostKind::ReduceScatterTwoShot:
      return reduceScatterTwoShotCost(
          model, codecs, messageBytes, localRanks, nodes);
    case cocclAutotuneCostKind::AllReduceOneShot:
      return allReduceOneShotCost(
          model, flat, messageBytes, localRanks, nodes);
    case cocclAutotuneCostKind::AllReduceTwoShot:
      return 2.0 * globalPccaCost(
                       model, flat, messageBytes, localRanks, nodes);
    case cocclAutotuneCostKind::AllReduceTripleShot:
      return reduceScatterTwoShotCost(
                 model, codecs, messageBytes, localRanks, nodes) +
             globalPccaCost(
                 model, codecs.defaultScope, messageBytes,
                 localRanks, nodes);
  }
  return std::numeric_limits<double>::infinity();
}
