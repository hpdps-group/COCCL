#include "coccl_autotune_internal.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

constexpr double kLayoutAlphaUs = 4.0;
constexpr double kLayoutBetaUsPerByte = 1.16e-6;

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

double encodeTime(const cocclAutotunePhaseCodec& codec,
                  double messageBytes) {
  if (!codec.compressed) return 0.0;
  if (codec.model == nullptr || !codec.model->valid) {
    return std::numeric_limits<double>::infinity();
  }
  if (codec.model->encodeTime.valid) {
    return cocclAutotunePredict(codec.model->encodeTime, messageBytes);
  }
  return codecTime(codec, messageBytes) * 0.5;
}

double decodeTime(const cocclAutotunePhaseCodec& codec,
                  double messageBytes) {
  if (!codec.compressed) return 0.0;
  if (codec.model == nullptr || !codec.model->valid) {
    return std::numeric_limits<double>::infinity();
  }
  if (codec.model->decodeTime.valid) {
    return cocclAutotunePredict(codec.model->decodeTime, messageBytes);
  }
  return codecTime(codec, messageBytes) * 0.5;
}

double drcTime(const cocclAutotunePhaseCodec& codec,
               double messageBytes) {
  if (!codec.compressed) return 0.0;
  if (codec.model == nullptr || !codec.model->valid) {
    return std::numeric_limits<double>::infinity();
  }
  if (codec.model->drcTime.valid) {
    return cocclAutotunePredict(codec.model->drcTime, messageBytes);
  }
  return codecTime(codec, messageBytes);
}

double pccaCost(const cocclLinearModel& p2p,
                const cocclAutotunePhaseCodec& codec,
                double messageBytes, int ranks,
                bool includeCodec = true) {
  double codecUs = codecTime(codec, messageBytes);
  if (!std::isfinite(codecUs) || messageBytes < 0.0) {
    return std::numeric_limits<double>::infinity();
  }
  if (!includeCodec) codecUs = 0.0;
  if (ranks <= 1) return codecUs;
  const double wireBytes = messageBytes /
      (compressionRatio(codec) * (double)ranks);
  return codecUs + cocclAutotunePredict(p2p, wireBytes);
}

double globalPccaCost(const cocclSelectionPerformanceModel& model,
                      const cocclAutotunePhaseCodec& codec,
                      double messageBytes,
                      int localRanks, int nodes,
                      bool includeCodec = true) {
  double codecUs = codecTime(codec, messageBytes);
  if (!std::isfinite(codecUs)) {
    return std::numeric_limits<double>::infinity();
  }
  if (!includeCodec) codecUs = 0.0;
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

double allGatherOneShotCost(
    const cocclSelectionPerformanceModel& model,
    const cocclAutotunePhaseCodec& codec, double messageBytes,
    int ranks) {
  const double encodeUs = encodeTime(codec, messageBytes);
  const double decodeUs = decodeTime(
      codec, messageBytes * (double)ranks);
  const double communicationUs = cocclAutotunePredict(
      model.allGather, messageBytes / compressionRatio(codec));
  if (!std::isfinite(encodeUs) || !std::isfinite(decodeUs) ||
      !std::isfinite(communicationUs)) {
    return std::numeric_limits<double>::infinity();
  }
  return encodeUs + communicationUs + decodeUs;
}

double allGatherTwoShotCost(
    const cocclSelectionPerformanceModel& model,
    const cocclAutotunePhaseCodec& codec, double messageBytes,
    int localRanks, int nodes) {
  const int ranks = localRanks * nodes;
  const double encodedBytes = messageBytes / compressionRatio(codec);
  const double encodeUs = encodeTime(codec, messageBytes);
  const double interUs = cocclAutotunePredict(
      model.allGatherInter, encodedBytes);
  const double intraUs = cocclAutotunePredict(
      model.allGatherIntra, encodedBytes * (double)nodes);
  const double decodeUs = decodeTime(
      codec, messageBytes * (double)ranks);
  const double transposeUs = kLayoutAlphaUs + kLayoutBetaUsPerByte *
      messageBytes * (double)ranks;
  if (!std::isfinite(encodeUs) || !std::isfinite(interUs) ||
      !std::isfinite(intraUs) || !std::isfinite(decodeUs)) {
    return std::numeric_limits<double>::infinity();
  }
  return encodeUs + interUs + intraUs + decodeUs + transposeUs;
}

double reduceScatterTwoShotCost(
    const cocclSelectionPerformanceModel& model,
    const cocclAutotuneCodecSet& codecs, double messageBytes,
    int localRanks, int nodes) {
  // Fused DRC replaces the intermediate full decode/encode pass. The first
  // phase still accounts for the codec work at the raw pipeline boundaries.
  return pccaCost(model.intraP2p, codecs.intra,
                  messageBytes, localRanks) +
         pccaCost(model.interP2p, codecs.inter,
                  messageBytes / (double)localRanks, nodes,
                  !codecs.fusedIntraToInter);
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

double flatAllReduceOneShotCost(
    const cocclSelectionPerformanceModel& model,
    const cocclAutotunePhaseCodec& codec, double messageBytes, int ranks,
    bool fusedDecompressReduce) {
  // Every rank contributes one complete encoded message. DecompressReduce
  // therefore decodes ranks copies even though AllGather sends one per rank.
  const double codecUs = codec.model != nullptr && codec.model->drTime.valid
      ? encodeTime(codec, messageBytes) + cocclAutotunePredict(
            codec.model->drTime, messageBytes * (double)ranks)
      : (fusedDecompressReduce
             ? codecTime(codec, messageBytes)
             : encodeTime(codec, messageBytes) +
                   decodeTime(codec, messageBytes * (double)ranks));
  if (!std::isfinite(codecUs)) {
    return std::numeric_limits<double>::infinity();
  }
  const double encodedBytes = messageBytes / compressionRatio(codec);
  return codecUs + cocclAutotunePredict(model.allGather, encodedBytes);
}

double flatAllReduceTwoShotCost(
    const cocclSelectionPerformanceModel& model,
    const cocclAutotunePhaseCodec& codec, double messageBytes,
    int ranks) {
  const double codecUs = codecTime(codec, messageBytes);
  const double drcUs = drcTime(codec, messageBytes);
  if (!std::isfinite(codecUs) || !std::isfinite(drcUs)) {
    return std::numeric_limits<double>::infinity();
  }
  const double encodedBytes = messageBytes / compressionRatio(codec);
  const double allToAllUs =
      cocclAutotunePredict(model.allToAll, encodedBytes);
  const double allGatherUs = cocclAutotunePredict(
      model.allGather, encodedBytes / (double)ranks);
  if (!std::isfinite(allToAllUs) || !std::isfinite(allGatherUs)) {
    return std::numeric_limits<double>::infinity();
  }
  return codecUs + drcUs + allToAllUs + allGatherUs;
}

double communicationKnee(const cocclLinearModel& model) {
  if (model.alphaUs > 0.0 && model.betaUsPerByte > 0.0) {
    return model.alphaUs / model.betaUsPerByte;
  }
  if (model.sampleCount < 2) {
    return model.betaUsPerByte > 0.0
        ? model.alphaUs / model.betaUsPerByte
        : std::numeric_limits<double>::infinity();
  }
  const double deltaBytes = model.sampleBytes[1] - model.sampleBytes[0];
  const double slope =
      (model.sampleTimeUs[1] - model.sampleTimeUs[0]) / deltaBytes;
  const double startupUs = model.sampleTimeUs[0] -
      slope * model.sampleBytes[0];
  return slope > 0.0 && startupUs > 0.0
      ? startupUs / slope
      : std::numeric_limits<double>::infinity();
}

bool uniformCodecRatio(const cocclAutotuneCodecSet& codecs, int nodes,
                       double* ratio) {
  const cocclAutotunePhaseCodec* phases[] = {
      nodes == 1 ? &codecs.intra : &codecs.defaultScope,
      &codecs.intra,
      &codecs.inter,
  };
  const int count = nodes == 1 ? 1 : 3;
  if (!phases[0]->compressed || phases[0]->model == nullptr ||
      !phases[0]->model->valid) {
    return false;
  }
  *ratio = phases[0]->model->compressionRatio;
  for (int i = 1; i < count; ++i) {
    if (!phases[i]->compressed || phases[i]->model == nullptr ||
        !phases[i]->model->valid ||
        std::abs(phases[i]->model->compressionRatio - *ratio) >
            1e-6 * *ratio) {
      return false;
    }
  }
  return true;
}

double crossoverCost(cocclAutotuneCostKind costKind,
                     const cocclSelectionPerformanceModel& model,
                     const cocclAutotuneCodecSet& codecs,
                     double messageBytes, int localRanks, int nodes) {
  if (costKind == cocclAutotuneCostKind::AllGatherOneShot ||
      costKind == cocclAutotuneCostKind::AllGatherTwoShot) {
    return std::numeric_limits<double>::infinity();
  }
  if (nodes <= 1) {
    if (costKind == cocclAutotuneCostKind::AllReduceOneShot) {
      return flatAllReduceOneShotCost(
          model, codecs.intra, messageBytes, localRanks,
          codecs.fusedFlatDecompressReduce);
    }
    if (costKind == cocclAutotuneCostKind::AllReduceTwoShot) {
      return flatAllReduceTwoShotCost(
          model, codecs.intra, messageBytes, localRanks);
    }
    return std::numeric_limits<double>::infinity();
  }

  double ratio = 0.0;
  if (!uniformCodecRatio(codecs, nodes, &ratio)) {
    return std::numeric_limits<double>::infinity();
  }
  const cocclLinearModel& p2p =
      nodes == 1 ? model.intraP2p : model.interP2p;
  if (p2p.sampleCount < 2) {
    return std::numeric_limits<double>::infinity();
  }

  if (costKind == cocclAutotuneCostKind::ReduceScatterOneShot ||
      costKind == cocclAutotuneCostKind::ReduceScatterTwoShot) {
    const double divisor = costKind ==
            cocclAutotuneCostKind::ReduceScatterOneShot
        ? (double)nodes / (double)(nodes - 1)
        : (double)localRanks * (double)nodes;
    double cost = cocclAutotunePredict(
        model.interP2p, messageBytes / (ratio * divisor));
    if (costKind == cocclAutotuneCostKind::ReduceScatterTwoShot) {
      cost += (model.allToAll.alphaUs + model.interP2p.alphaUs) /
          (double)localRanks;
    }
    return cost;
  }

  const double knee = communicationKnee(p2p);
  if (!std::isfinite(knee)) {
    return std::numeric_limits<double>::infinity();
  }
  const double wireBytes = messageBytes / ratio;
  if (costKind == cocclAutotuneCostKind::AllReduceOneShot) {
    return cocclAutotunePredict(p2p, wireBytes);
  }

  double threshold = knee * 0.5;
  if (costKind == cocclAutotuneCostKind::AllReduceTwoShot) {
    threshold *= 2.0;
  }
  return cocclAutotunePredict(p2p, threshold);
}

}  // namespace

cocclLinearModel cocclAutotuneFitLinearModel(
    const std::vector<cocclAutotuneProfilePoint>& points) {
  cocclLinearModel model;
  if (points.size() < 2) return model;

  model.sampleCount = std::min(
      points.size(), (size_t)kCocclAutotuneMaxProfilePoints);

  double meanX = 0.0;
  double meanY = 0.0;
  for (size_t i = 0; i < model.sampleCount; ++i) {
    const cocclAutotuneProfilePoint& point = points[i];
    if (!std::isfinite(point.bytes) || !std::isfinite(point.timeUs) ||
        point.bytes <= 0.0 || point.timeUs <= 0.0) {
      return model;
    }
    model.sampleBytes[i] = point.bytes;
    model.sampleTimeUs[i] = point.timeUs;
    meanX += point.bytes;
    meanY += point.timeUs;
  }
  meanX /= (double)model.sampleCount;
  meanY /= (double)model.sampleCount;

  double covariance = 0.0;
  double variance = 0.0;
  for (size_t i = 0; i < model.sampleCount; ++i) {
    const cocclAutotuneProfilePoint& point = points[i];
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
  if (model.sampleCount >= 2) {
    size_t upper = 1;
    while (upper + 1 < model.sampleCount &&
           bytes > model.sampleBytes[upper]) {
      ++upper;
    }
    const size_t lower = upper - 1;
    const double span =
        model.sampleBytes[upper] - model.sampleBytes[lower];
    const double slope =
        (model.sampleTimeUs[upper] - model.sampleTimeUs[lower]) / span;
    return std::max(
        0.0, model.sampleTimeUs[lower] +
                 slope * (bytes - model.sampleBytes[lower]));
  }
  return std::max(0.0, model.alphaUs + model.betaUsPerByte * bytes);
}

double cocclAutotuneEvaluateCost(
    cocclAutotuneCostKind costKind,
    const cocclSelectionPerformanceModel& model,
    const cocclAutotuneCodecSet& codecs, double messageBytes, int localRanks,
    int nodes) {
  const double calibrated = crossoverCost(
      costKind, model, codecs, messageBytes, localRanks, nodes);
  if (std::isfinite(calibrated)) return calibrated;

  const cocclAutotunePhaseCodec& flat =
      nodes == 1 ? codecs.intra : codecs.defaultScope;
  const int ranks = localRanks * nodes;
  switch (costKind) {
    case cocclAutotuneCostKind::AllGatherOneShot:
      return allGatherOneShotCost(model, flat, messageBytes, ranks);
    case cocclAutotuneCostKind::AllGatherTwoShot:
      return allGatherTwoShotCost(
          model, codecs.inter, messageBytes, localRanks, nodes);
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
                 localRanks, nodes, !codecs.fusedInterToDefault);
  }
  return std::numeric_limits<double>::infinity();
}
