#include "core/tuning/coccl_autotune_pipeline.h"

#include "comm.h"
#include "core/compression/coccl_compressor_runtime.h"
#include "core/pipeline/coccl_pipeline_internal.h"
#include "core/tuning/coccl_autotune_internal.h"
#include "debug.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <limits>
#include <map>
#include <tuple>

namespace {

constexpr int kMaxStageTerms = 3;
constexpr double kLayoutAlphaUs = 4.0;
constexpr double kLayoutBetaUsPerByte = 1.16e-6;
constexpr double kReductionBetaUsPerByte = 5.0e-7;
constexpr size_t kEfficientElements = size_t{16} << 20;
constexpr size_t kFramedTargetSliceBytes = size_t{32} << 20;

enum class StageResource : uint8_t {
  Layout,
  Codec,
  NcclSm,
};

struct StageCostTerm {
  cocclLinearModel model;
  double bytesPerRawByte = 0.0;
};

struct cocclAutotuneStageModel {
  cocclPipelineStageKind kind;
  StageResource resource;
  std::array<StageCostTerm, kMaxStageTerms> terms = {};
  int termCount = 0;
};

struct StageGraph {
  std::array<cocclAutotuneStageModel, kCocclPipelinePhysicalStages>
      stages = {};
  int stageCount = 0;
  bool framed = false;
  bool valid = true;
};

struct PipelineTuningKey {
  ncclComm_t comm;
  const char* recipe;
  size_t rawChunkCount;
  size_t inputChunks;
  ncclDataType_t datatype;
  int stageCount;
  std::array<cocclPipelineStageKind, kCocclPipelineExplicitStages>
      kinds = {};
  std::array<ncclComm_t, kCocclPipelineExplicitStages> comms = {};
  std::array<void*, kCocclPipelineExplicitStages> compressors = {};

  bool operator<(const PipelineTuningKey& other) const {
    return std::tie(comm, recipe, rawChunkCount, inputChunks, datatype,
                    stageCount, kinds, comms, compressors) <
        std::tie(other.comm, other.recipe, other.rawChunkCount,
                 other.inputChunks, other.datatype, other.stageCount,
                 other.kinds, other.comms, other.compressors);
  }
};

thread_local std::map<PipelineTuningKey, cocclPipelineTuningDecision>
    pipelineTuningCache;

cocclLinearModel fixedModel(double alphaUs, double betaUsPerByte) {
  cocclLinearModel model;
  model.alphaUs = alphaUs;
  model.betaUsPerByte = betaUsPerByte;
  model.valid = true;
  return model;
}

double modelCost(const cocclLinearModel& model, double bytes) {
  return cocclAutotunePredict(model, bytes);
}

double stageCost(const cocclAutotuneStageModel& stage,
                 double rawBytes) {
  double result = 0.0;
  for (int term = 0; term < stage.termCount; ++term) {
    const StageCostTerm& cost = stage.terms[term];
    const double predicted = modelCost(
        cost.model, rawBytes * cost.bytesPerRawByte);
    if (!std::isfinite(predicted)) {
      return std::numeric_limits<double>::infinity();
    }
    result += predicted;
  }
  return result;
}

void addTerm(cocclAutotuneStageModel* stage,
             const cocclLinearModel& model, double bytesPerRawByte,
             StageGraph* graph) {
  if (!model.valid || bytesPerRawByte <= 0.0) {
    graph->valid = false;
    return;
  }
  stage->terms[stage->termCount++] = {model, bytesPerRawByte};
}

void addStage(const cocclAutotuneStageModel& stage, StageGraph* graph) {
  graph->stages[graph->stageCount++] = stage;
}

double efficientModelBytes(const cocclLinearModel& model) {
  if (!model.valid) return std::numeric_limits<double>::infinity();
  if (model.sampleCount >= 2) {
    double bestThroughput = 0.0;
    for (size_t i = 0; i < model.sampleCount; ++i) {
      bestThroughput = std::max(
          bestThroughput, model.sampleBytes[i] / model.sampleTimeUs[i]);
    }
    for (size_t i = 0; i < model.sampleCount; ++i) {
      if (model.sampleBytes[i] / model.sampleTimeUs[i] >=
          0.9 * bestThroughput) {
        return model.sampleBytes[i];
      }
    }
  }
  if (model.alphaUs > 0.0 && model.betaUsPerByte > 0.0) {
    return 9.0 * model.alphaUs / model.betaUsPerByte;
  }
  return 0.0;
}

double stageEfficientRawBytes(const cocclAutotuneStageModel& stage) {
  double result = 0.0;
  for (int term = 0; term < stage.termCount; ++term) {
    const StageCostTerm& cost = stage.terms[term];
    result = std::max(
        result, efficientModelBytes(cost.model) / cost.bytesPerRawByte);
  }
  return result;
}

StageResource communicationResource() {
  return StageResource::NcclSm;
}

cocclLinearModel communicationModel(const cocclPipelineStage& stage) {
  switch (stage.kind) {
    case cocclPipelineStageAllGather:
      return cocclAutotuneSnapshotTopologyStageModel(
          stage.comm, cocclAutotuneTopologyOperation::AllGather);
    case cocclPipelineStageAllToAll:
      return cocclAutotuneSnapshotTopologyStageModel(
          stage.comm, cocclAutotuneTopologyOperation::AllToAll);
    case cocclPipelineStageReduceScatter:
      return cocclAutotuneSnapshotTopologyStageModel(
          stage.comm, cocclAutotuneTopologyOperation::ReduceScatter);
    default:
      __builtin_unreachable();
  }
}

StageGraph buildStageGraph(const cocclPipelineSpec* spec) {
  StageGraph graph;
  cocclAutotuneStageModel pack = {
      cocclPipelineStagePack, StageResource::Layout};
  addTerm(&pack, fixedModel(kLayoutAlphaUs, kLayoutBetaUsPerByte), 1.0,
          &graph);
  addStage(pack, &graph);

  double rawScale = 1.0;
  double wireScale = 1.0;
  void* currentCompressor = nullptr;
  cocclCodecModel currentCodec;

  for (int index = 0; index < spec->stageCount; ++index) {
    const cocclPipelineStage& pipelineStage = spec->stages[index];
    cocclAutotuneStageModel stage = {
        pipelineStage.kind, StageResource::Codec};

    switch (pipelineStage.kind) {
      case cocclPipelineStageCompress: {
        currentCompressor = pipelineStage.compressor;
        currentCodec = cocclAutotuneSnapshotCodecModel(
            currentCompressor, spec->datatype);
        graph.framed |= cocclCompressorSupports(
            currentCompressor, cocclCompressorCapabilityFramed);
        addTerm(&stage, currentCodec.encodeTime.valid
                            ? currentCodec.encodeTime : currentCodec.time,
                rawScale, &graph);
        wireScale = rawScale / currentCodec.compressionRatio;
        break;
      }
      case cocclPipelineStageAllToAll:
      case cocclPipelineStageAllGather:
      case cocclPipelineStageReduceScatter: {
        stage.resource = communicationResource();
        addTerm(&stage, communicationModel(pipelineStage), wireScale, &graph);
        if (pipelineStage.kind == cocclPipelineStageAllGather) {
          rawScale *= (double)pipelineStage.comm->nRanks;
          wireScale *= (double)pipelineStage.comm->nRanks;
        } else if (pipelineStage.kind ==
                   cocclPipelineStageReduceScatter) {
          rawScale /= (double)pipelineStage.comm->nRanks;
          wireScale /= (double)pipelineStage.comm->nRanks;
          currentCompressor = nullptr;
          currentCodec = {};
        }
        break;
      }
      case cocclPipelineStageDecompReduceComp: {
        const cocclCodecModel outputCodec =
            cocclAutotuneSnapshotCodecModel(
                pipelineStage.compressor, spec->datatype);
        const bool fused = currentCompressor != nullptr &&
            cocclCompressorSupports(
                currentCompressor,
                cocclCompressorCapabilityDecompressReduceCompress);
        if (fused) {
          addTerm(&stage, currentCodec.drcTime.valid
                              ? currentCodec.drcTime : currentCodec.time,
                  rawScale, &graph);
        } else {
          addTerm(&stage, currentCodec.decodeTime.valid
                              ? currentCodec.decodeTime : currentCodec.time,
                  rawScale, &graph);
          addTerm(&stage, outputCodec.encodeTime.valid
                              ? outputCodec.encodeTime : outputCodec.time,
                  rawScale / (double)pipelineStage.reduceChunks, &graph);
        }
        rawScale /= (double)pipelineStage.reduceChunks;
        currentCompressor = pipelineStage.compressor;
        currentCodec = outputCodec;
        graph.framed |= cocclCompressorSupports(
            currentCompressor, cocclCompressorCapabilityFramed);
        wireScale = rawScale / currentCodec.compressionRatio;
        break;
      }
      case cocclPipelineStageDecompressReduce: {
        addTerm(&stage, currentCodec.drTime.valid
                            ? currentCodec.drTime
                            : (currentCodec.decodeTime.valid
                                   ? currentCodec.decodeTime
                                   : currentCodec.time),
                rawScale, &graph);
        if (!currentCodec.drTime.valid) {
          addTerm(&stage, fixedModel(2.0, kReductionBetaUsPerByte),
                  rawScale, &graph);
        }
        rawScale /= (double)pipelineStage.reduceChunks;
        wireScale = rawScale;
        currentCompressor = nullptr;
        currentCodec = {};
        break;
      }
      case cocclPipelineStageDecompress:
        addTerm(&stage, currentCodec.decodeTime.valid
                            ? currentCodec.decodeTime : currentCodec.time,
                rawScale, &graph);
        wireScale = rawScale;
        currentCompressor = nullptr;
        currentCodec = {};
        break;
      case cocclPipelineStagePack:
      case cocclPipelineStageUnpack:
        __builtin_unreachable();
    }
    addStage(stage, &graph);
  }

  cocclAutotuneStageModel unpack = {
      cocclPipelineStageUnpack, StageResource::Layout};
  addTerm(&unpack, fixedModel(kLayoutAlphaUs, kLayoutBetaUsPerByte),
          wireScale, &graph);
  addStage(unpack, &graph);
  return graph;
}

double scoreGraph(const StageGraph& graph, size_t totalBytes,
                  size_t targetSliceBytes) {
  const int depth = cocclPipelineDepthForSlice(
      totalBytes, targetSliceBytes, kCocclAutotuneMaxPipelineDepth);
  const size_t regularBytes = depth == kCocclAutotuneMaxPipelineDepth &&
          totalBytes / (size_t)depth > targetSliceBytes
      ? totalBytes / (size_t)depth +
            (totalBytes % (size_t)depth != 0)
      : targetSliceBytes;

  std::array<double, kCocclPipelinePhysicalStages> streamReady = {};
  double codecWork = 0.0;
  double ncclSmWork = 0.0;
  double layoutWork = 0.0;
  double completion = 0.0;
  size_t consumed = 0;
  for (int slice = 0; slice < depth; ++slice) {
    const size_t bytes = std::min(regularBytes, totalBytes - consumed);
    consumed += bytes;
    double dependency = 0.0;
    for (int index = 0; index < graph.stageCount; ++index) {
      const cocclAutotuneStageModel& stage = graph.stages[index];
      if (depth == 1 && (stage.kind == cocclPipelineStagePack ||
                         stage.kind == cocclPipelineStageUnpack)) {
        continue;
      }
      const double duration = stageCost(stage, (double)bytes);
      if (!std::isfinite(duration)) return duration;
      const double finish =
          std::max(dependency, streamReady[index]) + duration;
      dependency = finish;
      streamReady[index] = finish;
      if (stage.resource == StageResource::Codec) codecWork += duration;
      if (stage.resource == StageResource::NcclSm) ncclSmWork += duration;
      if (stage.resource == StageResource::Layout) layoutWork += duration;
    }
    completion = dependency;
  }

  const double smContention = 0.08 * std::min(codecWork, ncclSmWork);
  const double memoryContention = 0.05 * std::min(layoutWork, codecWork);
  return completion + smContention + memoryContention;
}

size_t quantizeUp(size_t bytes, size_t step) {
  return bytes / step * step + (bytes % step != 0 ? step : 0);
}

void addCandidate(size_t bytes, size_t minimumBytes, size_t maximumBytes,
                  std::array<size_t, 24>* candidates, int* count) {
  bytes = std::min(maximumBytes, std::max(
      minimumBytes, quantizeUp(bytes, kCocclAutotuneSliceStepBytes)));
  for (int i = 0; i < *count; ++i) {
    if ((*candidates)[i] == bytes) return;
  }
  (*candidates)[(*count)++] = bytes;
}

PipelineTuningKey cacheKey(const cocclPipelineSpec* spec) {
  PipelineTuningKey key = {
      spec->ownerComm, spec->name, spec->rawChunkCount,
      spec->inputChunks, spec->datatype, spec->stageCount};
  for (int stage = 0; stage < spec->stageCount; ++stage) {
    key.kinds[stage] = spec->stages[stage].kind;
    key.comms[stage] = spec->stages[stage].comm;
    key.compressors[stage] = spec->stages[stage].compressor;
  }
  return key;
}

cocclPipelineTuningDecision chooseLayout(const cocclPipelineSpec* spec) {
  size_t totalBytes = spec->rawChunkCount * spec->inputChunks *
      (size_t)ncclTypeSize(spec->datatype);
  const StageGraph graph = buildStageGraph(spec);
  if (graph.framed) {
    bool hasAllGather = false;
    for (int stage = 0; stage < spec->stageCount; ++stage) {
      hasAllGather |=
          spec->stages[stage].kind == cocclPipelineStageAllGather;
    }
    const int maxDepth = hasAllGather
        ? (totalBytes < (size_t{4} << 30) ? 4 : 8)
        : (totalBytes < (size_t{2} << 30) ? 8 : 16);
    return {std::min(totalBytes, kFramedTargetSliceBytes), maxDepth};
  }
  if (!graph.valid || spec->stageCount < 4) {
    return {totalBytes, 1};
  }

  const size_t layoutFloor = kEfficientElements *
      (size_t)ncclTypeSize(spec->datatype);
  size_t efficientBytes = layoutFloor;
  for (int stage = 0; stage < graph.stageCount; ++stage) {
    const double knee = stageEfficientRawBytes(graph.stages[stage]);
    if (std::isfinite(knee) && knee > 0.0 &&
        (graph.stages[stage].resource == StageResource::Codec ||
         graph.stages[stage].resource == StageResource::Layout)) {
      efficientBytes = std::max(
          efficientBytes,
          std::min((size_t)knee, 4 * layoutFloor));
    }
  }

  const double seed = (double)std::min(totalBytes, efficientBytes);
  int bottleneck = 0;
  double bottleneckCost = 0.0;
  double otherSlope = 0.0;
  for (int stage = 0; stage < graph.stageCount; ++stage) {
    const double first = stageCost(graph.stages[stage], seed);
    const double second = stageCost(graph.stages[stage], 2.0 * seed);
    const double slope = std::max(0.0, (second - first) / seed);
    if (first > bottleneckCost) {
      bottleneckCost = first;
      bottleneck = stage;
    }
    otherSlope += slope;
  }
  const double first = stageCost(graph.stages[bottleneck], seed);
  const double second = stageCost(graph.stages[bottleneck], 2.0 * seed);
  const double bottleneckSlope = std::max(0.0, (second - first) / seed);
  const double bottleneckAlpha =
      std::max(0.0, first - bottleneckSlope * seed);
  otherSlope = std::max(0.0, otherSlope - bottleneckSlope);
  const double ideal = bottleneckAlpha > 0.0 && otherSlope > 0.0
      ? std::sqrt((double)totalBytes * bottleneckAlpha / otherSlope)
      : seed;

  std::array<size_t, 24> candidates = {};
  int candidateCount = 0;
  const size_t minimumBytes = layoutFloor;
  const size_t minimumParallelSlices = std::max<size_t>(
      2, ((size_t)spec->stageCount + 2) / 3);
  if (totalBytes / minimumParallelSlices < minimumBytes) {
    return {totalBytes, 1};
  }
  const size_t maximumBytes = totalBytes / minimumParallelSlices;
  addCandidate(efficientBytes / 2, minimumBytes, maximumBytes,
               &candidates, &candidateCount);
  addCandidate(efficientBytes, minimumBytes, maximumBytes,
               &candidates, &candidateCount);
  addCandidate(efficientBytes * 2, minimumBytes, maximumBytes,
               &candidates, &candidateCount);
  addCandidate((size_t)ideal, minimumBytes, maximumBytes,
               &candidates, &candidateCount);
  addCandidate((size_t)ideal + kCocclAutotuneSliceStepBytes, minimumBytes,
               maximumBytes, &candidates, &candidateCount);
  if ((size_t)ideal > kCocclAutotuneSliceStepBytes) {
    addCandidate((size_t)ideal - kCocclAutotuneSliceStepBytes, minimumBytes,
                 maximumBytes, &candidates, &candidateCount);
  }
  for (int stage = 0; stage < graph.stageCount; ++stage) {
    const double knee = stageEfficientRawBytes(graph.stages[stage]);
    if (std::isfinite(knee) && knee > 0.0) {
      addCandidate((size_t)knee, minimumBytes, maximumBytes,
                   &candidates, &candidateCount);
    }
  }

  std::array<double, 24> scores = {};
  double bestScore = std::numeric_limits<double>::infinity();
  for (int candidate = 0; candidate < candidateCount; ++candidate) {
    scores[candidate] = scoreGraph(
        graph, totalBytes, candidates[candidate]);
    bestScore = std::min(bestScore, scores[candidate]);
  }
  size_t selected = 0;
  for (int candidate = 0; candidate < candidateCount; ++candidate) {
    if (scores[candidate] <= bestScore * 1.03) {
      selected = std::max(selected, candidates[candidate]);
    }
  }
  return {selected, kCocclAutotuneMaxPipelineDepth};
}

}  // namespace

cocclPipelineTuningDecision cocclAutotunePipelineLayout(
    const cocclPipelineSpec* spec) {
  const PipelineTuningKey key = cacheKey(spec);
  auto found = pipelineTuningCache.find(key);
  bool inserted = false;
  if (found == pipelineTuningCache.end()) {
    found = pipelineTuningCache.emplace(key, chooseLayout(spec)).first;
    inserted = true;
  }
  if (inserted && spec->ownerComm->rank == 0) {
    const size_t totalBytes = spec->rawChunkCount * spec->inputChunks *
        (size_t)ncclTypeSize(spec->datatype);
    INFO(COCCL_TUNING,
         "COCCL pipeline recipe=%s bytes=%zu target_slice=%zu depth=%d",
         spec->name, totalBytes, found->second.targetSliceBytes,
         cocclPipelineDepthForSlice(
             totalBytes, found->second.targetSliceBytes,
             found->second.maxDepth));
  }
  return found->second;
}

void cocclAutotunePipelineCommDestroy(ncclComm_t comm) {
  for (auto item = pipelineTuningCache.begin();
       item != pipelineTuningCache.end();) {
    item = item->first.comm == comm ? pipelineTuningCache.erase(item)
                                    : std::next(item);
  }
}
