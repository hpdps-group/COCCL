#include "core/tuning/coccl_autotune_internal.h"

#include "comm.h"
#include "core/config/coccl_config.h"
#include "tuning.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

struct TuningCall {
  ncclComm_t comm;
  ncclFunc_t function;
  size_t bytes;
  uint64_t mask;
};

cocclConfig config;
std::vector<TuningCall> tuningCalls;

void initComm(ncclComm* comm, uint64_t id, int ranks, int nodes,
              float intraBandwidth, float interBandwidth,
              int p2pChannels, int channelsPerPeer) {
  comm->commHash = id;
  comm->nRanks = ranks;
  comm->nNodes = nodes;
  comm->p2pnChannels = p2pChannels;
  comm->p2pnChannelsPerPeer = channelsPerPeer;
  comm->p2pChunkSize = 128 << 10;
  comm->config.minCTAs = 1;
  comm->config.maxCTAs = 32;
  comm->graphs[NCCL_ALGO_RING].bwIntra = intraBandwidth;
  comm->graphs[NCCL_ALGO_RING].bwInter = interBandwidth;
  comm->tuningContext.generalLatencies
      [ncclFuncAllGather][NCCL_ALGO_RING][NCCL_PROTO_SIMPLE] = 12.0f;
}

double expectedP2p(ncclComm_t comm, size_t bytes, int peers) {
  const int scheduled = std::max(
      1, std::min(comm->p2pnChannels,
                  comm->p2pnChannelsPerPeer * peers));
  const size_t chunks = std::max<size_t>(
      1, (bytes + (size_t)comm->p2pChunkSize - 1) /
             (size_t)comm->p2pChunkSize);
  const int active = std::min<int>(scheduled, (int)chunks);
  const double bandwidth =
      (comm->nNodes > 1 ? comm->graphs[NCCL_ALGO_RING].bwInter
                        : comm->graphs[NCCL_ALGO_RING].bwIntra) * active;
  const double latency = 12.0 / std::max(1, comm->nRanks - 1);
  return latency + (double)bytes / (1000.0 * bandwidth);
}

bool close(double actual, double expected) {
  return std::abs(actual - expected) < 1e-5 * std::max(1.0, expected);
}

}  // namespace

const cocclConfig& cocclGetConfig() {
  return config;
}

ncclResult_t ncclTuningCompute(
    ncclTuningInput_t* input, ncclTuningResult_t* result) {
  tuningCalls.push_back(
      {input->comm, input->func, input->nBytes, input->tuningMask});
  result->valid = 1;
  result->timeUs =
      (float)input->comm->commHash + (float)input->nBytes * 1.0e-6f;
  return ncclSuccess;
}

int main() {
  config.autotune.profileMinBytes = 1 << 20;
  config.autotune.profileMaxBytes = 16 << 20;

  ncclComm owner = {};
  ncclComm intra = {};
  ncclComm inter = {};
  ncclComm gather = {};
  initComm(&owner, 10, 8, 2, 40.0f, 20.0f, 32, 4);
  initComm(&intra, 20, 4, 1, 50.0f, 0.0f, 16, 4);
  initComm(&inter, 30, 2, 2, 0.0f, 12.5f, 8, 2);
  initComm(&gather, 40, 8, 2, 40.0f, 20.0f, 32, 4);

  const cocclSelectionPerformanceModel model =
      cocclAutotuneSnapshotPerformanceModel(
          &owner, &intra, &inter, &gather);
  if (!model.intraP2p.valid || !model.interP2p.valid ||
      !model.allGather.valid || !model.allToAll.valid) {
    std::fprintf(stderr, "topology model is incomplete\n");
    return 1;
  }
  if (tuningCalls.size() != 3) {
    std::fprintf(stderr, "expected three AllGather tuning queries, got %zu\n",
                 tuningCalls.size());
    return 1;
  }
  for (const TuningCall& call : tuningCalls) {
    if (call.comm != &gather || call.function != ncclFuncAllGather ||
        (call.mask & NCCL_TUNING_MASK_SYM_KERNELS) == 0) {
      std::fprintf(stderr, "AllGather did not use its stage communicator\n");
      return 1;
    }
  }

  const size_t firstBytes = 1 << 20;
  if (!close(model.intraP2p.sampleTimeUs[0],
             expectedP2p(&intra, firstBytes, 1)) ||
      !close(model.interP2p.sampleTimeUs[0],
             expectedP2p(&inter, firstBytes, 1))) {
    std::fprintf(stderr, "P2P model did not use intra/inter topology\n");
    return 1;
  }
  const size_t allToAllWireBytes =
      firstBytes * (size_t)(owner.nRanks - 1) / (size_t)owner.nRanks;
  if (!close(model.allToAll.sampleTimeUs[0],
             expectedP2p(&owner, allToAllWireBytes, owner.nRanks - 1))) {
    std::fprintf(stderr, "AllToAll model did not use owner topology\n");
    return 1;
  }

  (void)cocclAutotuneSnapshotPerformanceModel(
      &owner, &intra, &inter, &gather);
  if (tuningCalls.size() != 3) {
    std::fprintf(stderr, "cached topology model was rebuilt\n");
    return 1;
  }

  ncclComm secondGather = {};
  initComm(&secondGather, 80, 8, 2, 40.0f, 20.0f, 32, 4);
  const cocclSelectionPerformanceModel changed =
      cocclAutotuneSnapshotPerformanceModel(
          &owner, &intra, &inter, &secondGather);
  if (tuningCalls.size() != 6 ||
      !(changed.allGather.sampleTimeUs[0] > model.allGather.sampleTimeUs[0])) {
    std::fprintf(stderr, "changed NCCL tuning state was not reflected\n");
    return 1;
  }

  cocclAutotuneTopologyCommDestroy(&owner);
  (void)cocclAutotuneSnapshotPerformanceModel(
      &owner, &intra, &inter, &gather);
  if (tuningCalls.size() != 9) {
    std::fprintf(stderr, "destroy did not clear owner topology cache\n");
    return 1;
  }

  std::printf("COCCL autotune topology tests: PASS\n");
  return 0;
}
