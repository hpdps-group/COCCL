#include "coccl_autotune_internal.h"

#include "core/config/coccl_config.h"
#include "core/pipeline/coccl_pipeline.h"
#include "core/runtime/coccl_comm.h"
#include "core/runtime/coccl_primitive_dispatch.h"
#include "comm.h"
#include "core/compression/compress.h"
#include "core/tuning/coccl_autotune_pipeline.h"
#include "debug.h"

#include <limits>

namespace {

bool uniformNodeRanks(ncclComm_t comm) {
  for (int node = 0; node < comm->nNodes; ++node) {
    if (comm->nodeRanks[node].localRanks != comm->localRanks) return false;
  }
  return comm->nRanks == comm->nNodes * comm->localRanks;
}

cocclAlgorithmKind configuredAlgorithm(cocclOperation operation) {
  const cocclAutotuneConfig& config = cocclGetConfig().autotune;
  if (operation == cocclOperation::AllGather) {
    switch (config.allGatherAlgorithm) {
      case cocclAllGatherAlgorithmPolicy::Auto:
        return cocclAlgorithmNone;
      case cocclAllGatherAlgorithmPolicy::OneShot:
        return cocclAlgorithmAllGatherOneShot;
      case cocclAllGatherAlgorithmPolicy::TwoShot:
        return cocclAlgorithmAllGatherTwoShot;
    }
  }
  if (operation == cocclOperation::ReduceScatter) {
    switch (config.reduceScatterAlgorithm) {
      case cocclReduceScatterAlgorithmPolicy::Auto:
        return cocclAlgorithmNone;
      case cocclReduceScatterAlgorithmPolicy::OneShot:
        return cocclAlgorithmReduceScatterOneShot;
      case cocclReduceScatterAlgorithmPolicy::TwoShot:
        return cocclAlgorithmReduceScatterTwoShot;
    }
  }
  if (operation == cocclOperation::AllReduce) {
    switch (config.allReduceAlgorithm) {
      case cocclAllReduceAlgorithmPolicy::Auto:
        return cocclAlgorithmNone;
      case cocclAllReduceAlgorithmPolicy::OneShot:
        return cocclAlgorithmAllReduceOneShot;
      case cocclAllReduceAlgorithmPolicy::TwoShot:
        return cocclAlgorithmAllReduceTwoShot;
      case cocclAllReduceAlgorithmPolicy::TripleShot:
        return cocclAlgorithmAllReduceTripleShot;
    }
  }
  return cocclAlgorithmNone;
}

void warnForcedFallback(cocclOperation operation) {
  if (operation == cocclOperation::AllGather) {
    WARN("COCCL forced AllGather twoshot is unavailable for this topology; using oneshot");
  } else if (operation == cocclOperation::ReduceScatter) {
    WARN("COCCL forced ReduceScatter twoshot is unavailable for this topology; using oneshot");
  } else {
    WARN("COCCL forced AllReduce tripleshot is unavailable; using twoshot");
  }
}

double messageBytes(const cocclPreparedCall& prepared) {
  const cocclInfo& info = prepared.info;
  double bytes = (double)info.count * (double)ncclTypeSize(info.datatype);
  if (info.operation == cocclOperation::ReduceScatter) {
    bytes *= (double)info.comm->nRanks;
  }
  return bytes;
}

bool hasFusedDrc(void* decoder, void* encoder) {
  return decoder != nullptr && encoder != nullptr &&
      cocclCompressorDescriptor(decoder) ==
          cocclCompressorDescriptor(encoder) &&
      cocclCompressorSupports(
          encoder, cocclCompressorCapabilityDecompressReduceCompress);
}

bool hasFusedDr(void* compressor) {
  return compressor != nullptr && cocclCompressorSupports(
      compressor, cocclCompressorCapabilityDecompressReduce);
}

bool usesFramedCompressor(const cocclPreparedCall& prepared) {
  for (void* compressor : prepared.compressors.handles) {
    if (compressor != nullptr && cocclCompressorSupports(
            compressor, cocclCompressorCapabilityFramed)) {
      return true;
    }
  }
  return false;
}

double allGatherPipelineCost(
    const cocclPreparedCall& prepared, cocclAlgorithmKind algorithm,
    const cocclHierarchicalComms& hierarchy, ncclComm_t gatherComm) {
  const cocclInfo& info = prepared.info;
  if (algorithm == cocclAlgorithmAllGatherTwoShot &&
      messageBytes(prepared) <
          4.0 * (double)kCocclAutotuneSliceStepBytes) {
    return std::numeric_limits<double>::infinity();
  }
  const cocclCompressionScope scope =
      algorithm == cocclAlgorithmAllGatherTwoShot
          ? cocclCompressionScope::Inter
          : info.comm->nNodes == 1
              ? cocclCompressionScope::Intra : cocclCompressionScope::Default;
  cocclPipelineStage stages[4];
  const cocclPipelineSpec spec = cocclBuildAllGatherSpec(
      info, algorithm, prepared.compressors.get(scope),
      algorithm == cocclAlgorithmAllGatherTwoShot
          ? hierarchy.interComm : gatherComm,
      hierarchy.intraComm, stages);
  return cocclAutotunePipelineLayout(&spec).predictedTimeUs;
}

}  // namespace

ncclResult_t cocclSelectAlgorithm(cocclPreparedCall* prepared) {
  prepared->algorithm = cocclAlgorithmNone;
  const cocclInfo& info = prepared->info;
  ncclComm_t comm = info.comm;
  const cocclAutotuneEligibility eligibility = {
      comm->nNodes > 1 && comm->localRanks > 1 && uniformNodeRanks(comm),
      info.count % (size_t)comm->nRanks == 0,
  };
  cocclAutotuneCandidateSet candidates =
      cocclAutotuneBuildCandidates(info.operation, eligibility);
  cocclAutotuneCandidateSet usable;
  for (size_t i = 0; i < candidates.count; ++i) {
    const cocclAlgorithmKind algorithm =
        candidates.candidates[i].spec->algorithm;
    const bool nativeAllGather =
        info.operation == cocclOperation::AllGather &&
        algorithm == cocclAlgorithmAllGatherOneShot &&
        !cocclPreparedAlgorithmHasCompression(prepared, algorithm);
    if (nativeAllGather ||
        cocclPreparedAlgorithmSupported(prepared, algorithm)) {
      usable.candidates[usable.count++] = candidates.candidates[i];
    }
  }
  candidates = usable;
  if (candidates.count == 0) return ncclInvalidArgument;

  const cocclAutotuneConfig& config = cocclGetConfig().autotune;
  const cocclAlgorithmKind requested = configuredAlgorithm(info.operation);
  const bool scoreCandidates =
      requested == cocclAlgorithmNone && config.enabled;

  if (scoreCandidates) {
    cocclHierarchicalComms hierarchy = {comm, comm, comm};
    if (comm->nNodes > 1 && comm->localRanks > 1) {
      NCCLCHECK(cocclCommGetHierarchicalComms(comm, &hierarchy));
    }
    ncclComm_t gatherComm = comm;
    if (comm->nNodes > 1 && !usesFramedCompressor(*prepared)) {
      NCCLCHECK(cocclCommGetZeroCtaComm(comm, &gatherComm));
    }
    if (info.operation == cocclOperation::AllGather) {
      for (size_t i = 0; i < candidates.count; ++i) {
        cocclAutotuneCandidate* candidate = &candidates.candidates[i];
        candidate->scoreUs = allGatherPipelineCost(
            *prepared, candidate->spec->algorithm, hierarchy, gatherComm);
      }
      if (comm->nNodes > 1 &&
          prepared->compressors.get(cocclCompressionScope::Default) ==
              nullptr) {
        cocclAutotuneCandidate* oneShot = cocclAutotuneFindCandidate(
            &candidates, cocclAlgorithmAllGatherOneShot);
        cocclAutotuneCandidate* twoShot = cocclAutotuneFindCandidate(
            &candidates, cocclAlgorithmAllGatherTwoShot);
        constexpr double kMinimumPredictedSpeedup = 1.05;
        if (oneShot != nullptr && twoShot != nullptr &&
            twoShot->scoreUs * kMinimumPredictedSpeedup >
                oneShot->scoreUs) {
          twoShot->scoreUs = oneShot->scoreUs;
        }
      }
    } else {
      cocclCodecModel defaultCodecModel;
      cocclCodecModel intraCodecModel;
      cocclCodecModel interCodecModel;
      const cocclSelectionPerformanceModel performance =
          cocclAutotuneSnapshotPerformanceModel(
              comm, hierarchy.intraComm, hierarchy.interComm, gatherComm);
      cocclAutotuneSnapshotCodecModels(
            prepared->compressors.get(cocclCompressionScope::Default),
            prepared->compressors.get(cocclCompressionScope::Intra),
            prepared->compressors.get(cocclCompressionScope::Inter),
            info.datatype,
            &defaultCodecModel, &intraCodecModel, &interCodecModel);
      const cocclAutotuneCodecSet codecs = {
          {prepared->compressors.get(cocclCompressionScope::Default) !=
               nullptr,
           &defaultCodecModel},
          {prepared->compressors.get(cocclCompressionScope::Intra) !=
               nullptr,
           &intraCodecModel},
          {prepared->compressors.get(cocclCompressionScope::Inter) !=
               nullptr,
           &interCodecModel},
          hasFusedDrc(
              prepared->compressors.get(cocclCompressionScope::Intra),
              prepared->compressors.get(cocclCompressionScope::Inter)),
          hasFusedDrc(
              prepared->compressors.get(cocclCompressionScope::Inter),
              prepared->compressors.get(cocclCompressionScope::Default)),
          hasFusedDr(
              prepared->compressors.get(cocclCompressionScope::Intra)),
      };
      const double bytes = messageBytes(*prepared);
      for (size_t i = 0; i < candidates.count; ++i) {
        cocclAutotuneCandidate* candidate = &candidates.candidates[i];
        candidate->scoreUs = cocclAutotuneEvaluateCost(
            candidate->spec->costKind, performance, codecs, bytes,
            comm->localRanks, comm->nNodes);
      }
    }
  }

  cocclAutotuneDecision decision = cocclAutotuneChooseCandidate(
      candidates, requested, config.enabled);
  if (requested == cocclAlgorithmNone && config.enabled &&
      !decision.usedModel && info.operation == cocclOperation::AllReduce &&
      usesFramedCompressor(*prepared)) {
    constexpr double kFramedOneShotMaxBytes = double(size_t{2} << 30);
    const cocclAlgorithmKind fallback =
        messageBytes(*prepared) <= kFramedOneShotMaxBytes
        ? cocclAlgorithmAllReduceOneShot
        : cocclAlgorithmAllReduceTwoShot;
    const cocclAutotuneCandidate* candidate =
        cocclAutotuneFindCandidate(candidates, fallback);
    if (candidate != nullptr) decision.candidate = candidate->spec;
  }
  if (decision.candidate == nullptr) return ncclInvalidArgument;
  if (decision.forcedFallback) warnForcedFallback(info.operation);

  prepared->algorithm = decision.candidate->algorithm;
  if (comm->rank == 0) {
    double scores[3] = {
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity()};
    for (size_t i = 0; i < candidates.count; ++i) {
      const cocclAutotuneCandidate& candidate = candidates.candidates[i];
      scores[static_cast<size_t>(candidate.spec->scoreSlot)] = candidate.scoreUs;
    }
    INFO(COCCL_TUNING,
         "COCCL select bytes=%g ranks=%d local=%d nodes=%d one=%g two=%g triple=%g model=%d -> %s",
         messageBytes(*prepared), comm->nRanks, comm->localRanks, comm->nNodes,
         scores[0], scores[1], scores[2], (int)decision.usedModel,
         cocclAutotuneAlgorithmName(prepared->algorithm));
  }
  return ncclSuccess;
}
