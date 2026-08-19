#include "coccl_autotune_internal.h"

#include "coccl_config.h"
#include "comm.h"
#include "compress.h"
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

void setCandidateScore(cocclPreparedCall* prepared,
                       cocclAutotuneCandidate* candidate, double scoreUs) {
  candidate->scoreUs = scoreUs;
  switch (candidate->spec->scoreSlot) {
    case cocclAutotuneScoreSlot::OneShot:
      prepared->oneShotUs = scoreUs;
      return;
    case cocclAutotuneScoreSlot::TwoShot:
      prepared->twoShotUs = scoreUs;
      return;
    case cocclAutotuneScoreSlot::TripleShot:
      prepared->tripleShotUs = scoreUs;
      return;
  }
}

void warnForcedFallback(cocclOperation operation) {
  if (operation == cocclOperation::ReduceScatter) {
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

ncclResult_t selectCandidate(cocclPreparedCall* prepared) {
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
    if (cocclPreparedAlgorithmSupported(
            prepared, candidates.candidates[i].spec->algorithm)) {
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
    cocclCodecModel defaultCodecModel;
    cocclCodecModel intraCodecModel;
    cocclCodecModel interCodecModel;
    const cocclSelectionPerformanceModel performance =
        cocclAutotuneSnapshotPerformanceModel(
            prepared->compressors.get(cocclCompressionScope::Default),
            prepared->compressors.get(cocclCompressionScope::Intra),
            prepared->compressors.get(cocclCompressionScope::Inter),
            &defaultCodecModel, &intraCodecModel, &interCodecModel);
    const cocclAutotuneCodecSet codecs = {
        {prepared->compressors.get(cocclCompressionScope::Default) !=
             nullptr,
         &defaultCodecModel},
        {prepared->compressors.get(cocclCompressionScope::Intra) != nullptr,
         &intraCodecModel},
        {prepared->compressors.get(cocclCompressionScope::Inter) != nullptr,
         &interCodecModel},
    };
    const double bytes = messageBytes(*prepared);
    for (size_t i = 0; i < candidates.count; ++i) {
      cocclAutotuneCandidate* candidate = &candidates.candidates[i];
      setCandidateScore(
          prepared, candidate,
          cocclAutotuneEvaluateCost(
              candidate->spec->costKind, performance,
              codecs, bytes,
              comm->localRanks, comm->nNodes));
    }
  }

  const cocclAutotuneDecision decision = cocclAutotuneChooseCandidate(
      candidates, requested, config.enabled);
  if (decision.candidate == nullptr) return ncclInvalidArgument;
  if (decision.forcedFallback) warnForcedFallback(info.operation);

  prepared->algorithm = decision.candidate->algorithm;
  prepared->usedModel = decision.usedModel;
  return ncclSuccess;
}

}  // namespace

ncclResult_t cocclSelectAlgorithm(cocclPreparedCall* prepared) {
  prepared->algorithm = cocclAlgorithmNone;
  prepared->oneShotUs = std::numeric_limits<double>::infinity();
  prepared->twoShotUs = std::numeric_limits<double>::infinity();
  prepared->tripleShotUs = std::numeric_limits<double>::infinity();
  prepared->usedModel = false;

  ncclResult_t result = selectCandidate(prepared);
  const cocclInfo& info = prepared->info;
  if (result == ncclSuccess && info.comm->rank == 0) {
    INFO(NCCL_TUNING,
         "COCCL select bytes=%g ranks=%d local=%d nodes=%d one=%g two=%g triple=%g model=%d -> %s",
         messageBytes(*prepared), info.comm->nRanks,
         info.comm->localRanks, info.comm->nNodes, prepared->oneShotUs,
         prepared->twoShotUs, prepared->tripleShotUs,
         (int)prepared->usedModel,
         cocclAutotuneAlgorithmName(prepared->algorithm));
  }
  return result;
}
