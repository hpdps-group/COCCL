#include "core/tuning/coccl_autotune_internal.h"

#include "comm.h"
#include "core/config/coccl_config.h"
#include "device.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

struct TuningCall {
  ncclComm_t comm;
  int function;
  int algorithm;
  int protocol;
  size_t bytes;
};

cocclConfig config;
std::vector<TuningCall> tuningCalls;

void initComm(ncclComm* comm, int ranks, int nodes,
              float intraBandwidth, float interBandwidth) {
  comm->nRanks = ranks;
  comm->nNodes = nodes;
  comm->localRanks = ranks / nodes;
  comm->maxLocalRanks = comm->localRanks;
  comm->nChannels = 8;
  comm->p2pnChannels = 8;
  comm->p2pnChannelsPerPeer = 2;
  comm->p2pChunkSize = 128 << 10;
  comm->graphs[NCCL_ALGO_RING].bwIntra = intraBandwidth;
  comm->graphs[NCCL_ALGO_RING].bwInter = interBandwidth;
  comm->latencies[ncclFuncAllGather][NCCL_ALGO_RING][NCCL_PROTO_SIMPLE] =
      12.0f;
  for (int function : {ncclFuncAllGather, ncclFuncReduceScatter}) {
    comm->bandwidths[function][NCCL_ALGO_RING][NCCL_PROTO_SIMPLE] = 40.0f;
    comm->bandwidths[function][NCCL_ALGO_PAT][NCCL_PROTO_SIMPLE] = 30.0f;
    comm->maxThreads[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE] = 256;
    comm->maxThreads[NCCL_ALGO_PAT][NCCL_PROTO_SIMPLE] = 256;
    comm->threadThresholds[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE] = 8;
    comm->threadThresholds[NCCL_ALGO_PAT][NCCL_PROTO_SIMPLE] = 8;
  }
}

void fail(const char* message) {
  std::fprintf(stderr, "%s\n", message);
  std::exit(1);
}

}  // namespace

const cocclConfig& cocclGetConfig() {
  return config;
}

ncclResult_t ncclTopoGetAlgoTime(
    ncclComm* comm, int function, int algorithm, int protocol,
    size_t bytes, int, float* time) {
  tuningCalls.push_back({comm, function, algorithm, protocol, bytes});
  const float startup = algorithm == NCCL_ALGO_PAT ? 4.0f : 9.0f;
  *time = startup + (float)bytes * 1.0e-6f;
  return ncclSuccess;
}

int main() {
  config.autotune.profileMinBytes = 1 << 20;
  config.autotune.profileMaxBytes = 4 << 20;

  ncclComm owner = {};
  ncclComm intra = {};
  ncclComm inter = {};
  ncclComm gather = {};
  initComm(&owner, 8, 2, 40.0f, 20.0f);
  initComm(&intra, 4, 1, 50.0f, 0.0f);
  initComm(&inter, 2, 2, 0.0f, 12.5f);
  initComm(&gather, 8, 2, 40.0f, 20.0f);

  const cocclNcclCostEstimate allGather =
      cocclAutotuneEstimateNcclStage(
          &gather, cocclAutotuneTopologyOperation::AllGather, 1 << 20);
  if (!std::isfinite(allGather.timeUs) ||
      allGather.algorithm != NCCL_ALGO_PAT ||
      allGather.protocol != NCCL_PROTO_SIMPLE || allGather.channels <= 0 ||
      allGather.executionClass != cocclNcclExecutionClass::General) {
    fail("2.27 collective estimate did not select the finite minimum");
  }
  if (tuningCalls.back().bytes != (size_t{8} << 20)) {
    fail("AllGather did not use NCCL 2.27 total-byte semantics");
  }

  const cocclNcclCostEstimate p2pIntra =
      cocclAutotuneEstimateNcclStage(
          &owner, cocclAutotuneTopologyOperation::P2pIntra, 1 << 20);
  const cocclNcclCostEstimate p2pInter =
      cocclAutotuneEstimateNcclStage(
          &owner, cocclAutotuneTopologyOperation::P2pInter, 1 << 20);
  if (!std::isfinite(p2pIntra.timeUs) ||
      !std::isfinite(p2pInter.timeUs) ||
      !(p2pInter.timeUs > p2pIntra.timeUs) ||
      p2pInter.channels >= p2pIntra.channels) {
    fail("P2P topology model did not distinguish intra/inter peers");
  }

  ncclComm unavailable = {};
  initComm(&unavailable, 4, 1, 40.0f, 0.0f);
  unavailable.bandwidths[ncclFuncAllGather][NCCL_ALGO_RING]
      [NCCL_PROTO_SIMPLE] = 0.0f;
  unavailable.bandwidths[ncclFuncAllGather][NCCL_ALGO_PAT]
      [NCCL_PROTO_SIMPLE] = 0.0f;
  if (std::isfinite(cocclAutotuneEstimateNcclStage(
          &unavailable, cocclAutotuneTopologyOperation::AllGather,
          1 << 20).timeUs)) {
    fail("unavailable NCCL candidates produced a finite estimate");
  }

  const size_t callsBeforeModel = tuningCalls.size();
  const cocclSelectionPerformanceModel model =
      cocclAutotuneSnapshotPerformanceModel(
          &owner, &intra, &inter, &gather);
  if (!model.intraP2p.valid || !model.interP2p.valid ||
      !model.allGather.valid || !model.allToAll.valid) {
    fail("topology model is incomplete");
  }
  for (size_t index = callsBeforeModel; index < tuningCalls.size(); ++index) {
    if (tuningCalls[index].comm != &gather ||
        tuningCalls[index].function != ncclFuncAllGather) {
      fail("model queried the wrong stage communicator");
    }
  }

  const size_t cachedCalls = tuningCalls.size();
  (void)cocclAutotuneSnapshotPerformanceModel(
      &owner, &intra, &inter, &gather);
  if (tuningCalls.size() != cachedCalls) {
    fail("cached topology model was rebuilt");
  }

  const cocclLinearModel reduceScatter =
      cocclAutotuneSnapshotTopologyStageModel(
          &inter, cocclAutotuneTopologyOperation::ReduceScatter);
  if (!reduceScatter.valid || tuningCalls.back().comm != &inter ||
      tuningCalls.back().function != ncclFuncReduceScatter) {
    fail("ReduceScatter stage did not use its executing communicator");
  }
  const cocclLinearModel allGatherStage =
      cocclAutotuneSnapshotTopologyStageModel(
          &gather, cocclAutotuneTopologyOperation::AllGather);
  if (!allGatherStage.valid ||
      std::fabs(allGatherStage.alphaUs - model.allGather.alphaUs - 35.0) >
          1.0e-6 || allGatherStage.sampleCount == 0 ||
      std::fabs(allGatherStage.sampleTimeUs[0] -
                    model.allGather.sampleTimeUs[0] - 35.0) > 1.0e-6) {
    fail("2.27 pipeline AllGather dispatch cost was not applied");
  }

  cocclAutotuneTopologyCommDestroy(&owner);
  (void)cocclAutotuneSnapshotPerformanceModel(
      &owner, &intra, &inter, &gather);
  if (tuningCalls.size() == cachedCalls) {
    fail("communicator destroy did not clear the topology cache");
  }

  std::printf("COCCL NCCL 2.27 topology adapter tests: PASS\n");
  return 0;
}
