#include "core/pipeline/coccl_pipeline_depth.h"

#include "comm.h"
#include "core/compression/coccl_compressor_runtime.h"
#include "core/pipeline/coccl_pipeline_internal.h"
#include "debug.h"

#include <array>
#include <iterator>
#include <map>
#include <tuple>

namespace {

constexpr std::array<int, 4> kAutoDepths = {1, 2, 4, 8};

struct DepthCacheKey {
  ncclComm_t comm;
  const char* recipe;
  size_t rawChunkCount;
  size_t inputChunks;
  ncclDataType_t datatype;
  int stageCount;
  std::array<cocclPipelineStageKind, kCocclPipelineExplicitStages> kinds = {};
  std::array<ncclComm_t, kCocclPipelineExplicitStages> stageComms = {};
  std::array<void*, kCocclPipelineExplicitStages> compressors = {};

  bool operator<(const DepthCacheKey& other) const {
    return std::tie(comm, recipe, rawChunkCount, inputChunks, datatype,
                    stageCount, kinds, stageComms, compressors) <
        std::tie(other.comm, other.recipe, other.rawChunkCount,
                 other.inputChunks, other.datatype, other.stageCount,
                 other.kinds, other.stageComms, other.compressors);
  }
};

thread_local std::map<DepthCacheKey, int> depthCache;

DepthCacheKey cacheKey(const cocclPipelineSpec* spec) {
  DepthCacheKey key = {
      spec->ownerComm, spec->name, spec->rawChunkCount,
      spec->inputChunks, spec->datatype, spec->stageCount};
  for (int stage = 0; stage < spec->stageCount; ++stage) {
    key.kinds[stage] = spec->stages[stage].kind;
    key.stageComms[stage] = spec->stages[stage].comm;
    key.compressors[stage] = spec->stages[stage].compressor;
  }
  return key;
}

bool usesFramedCompressor(const cocclPipelineSpec* spec) {
  for (int stage = 0; stage < spec->stageCount; ++stage) {
    void* compressor = spec->stages[stage].compressor;
    if (compressor != nullptr && cocclCompressorSupports(
            compressor, cocclCompressorCapabilityFramed)) {
      return true;
    }
  }
  return false;
}

bool hasStage(const cocclPipelineSpec* spec, cocclPipelineStageKind kind) {
  for (int stage = 0; stage < spec->stageCount; ++stage) {
    if (spec->stages[stage].kind == kind) return true;
  }
  return false;
}

}  // namespace

int cocclAutotunePipelineDepth(const cocclPipelineSpec* spec) {
  const DepthCacheKey key = cacheKey(spec);
  const auto found = depthCache.find(key);
  if (found != depthCache.end()) return found->second;

  const double rawBytes = (double)spec->rawChunkCount *
      (double)spec->inputChunks * (double)ncclTypeSize(spec->datatype);
  int selected = 1;
  const bool framed = usesFramedCompressor(spec);
  if (spec->stageCount >= 4 || framed) {
    int maxDepth = 8;
    if (framed && spec->stageCount >= 4 &&
        hasStage(spec, cocclPipelineStageAllGather) &&
        rawBytes < (double)(size_t{4} << 30)) {
      maxDepth = rawBytes < (double)(size_t{64} << 20)
          ? 1 : (rawBytes < (double)(size_t{128} << 20) ? 2 : 4);
    }
    int lastEffectiveDepth = 0;
    for (int requested : kAutoDepths) {
      if (requested > maxDepth) break;
      cocclPipelineContext context;
      if (cocclPreparePipeline(spec, requested, &context) != ncclSuccess ||
          context.depth == lastEffectiveDepth) {
        continue;
      }
      lastEffectiveDepth = context.depth;
      if (rawBytes / (double)context.depth >=
          (double)kCocclPipelineTargetSliceBytes) {
        selected = context.depth;
      }
    }
  }
  depthCache.emplace(key, selected);

  if (spec->ownerComm->rank == 0) {
    INFO(COCCL_TUNING,
         "COCCL depth recipe=%s bytes=%g target_slice=%zu -> %d",
         spec->name, rawBytes, kCocclPipelineTargetSliceBytes, selected);
  }
  return selected;
}

void cocclPipelineDepthCommDestroy(ncclComm_t comm) {
  for (auto item = depthCache.begin(); item != depthCache.end();) {
    item = item->first.comm == comm ? depthCache.erase(item)
                                    : std::next(item);
  }
}
