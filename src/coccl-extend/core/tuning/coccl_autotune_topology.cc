#include "coccl_autotune_internal.h"

#include "comm.h"
#include "core/config/coccl_config.h"
#include "device.h"
#include "sym_kernels.h"
#include "tuning.h"

#include <algorithm>
#include <iterator>
#include <map>
#include <tuple>

namespace {

struct TopologyModelKey {
  ncclComm_t owner = nullptr;
  ncclComm_t intra = nullptr;
  ncclComm_t inter = nullptr;
  ncclComm_t gather = nullptr;

  bool operator<(const TopologyModelKey& other) const {
    return std::tie(owner, intra, inter, gather) <
        std::tie(other.owner, other.intra, other.inter, other.gather);
  }
};

thread_local std::map<TopologyModelKey, cocclSelectionPerformanceModel>
    topologyModels;
thread_local std::map<
    std::pair<ncclComm_t, cocclAutotuneTopologyOperation>, cocclLinearModel>
    topologyStageModels;

std::vector<size_t> topologySampleSizes() {
  const cocclAutotuneConfig& config = cocclGetConfig().autotune;
  std::vector<size_t> sizes;
  for (size_t bytes = config.profileMinBytes;
       bytes <= config.profileMaxBytes;) {
    sizes.push_back(bytes);
    if (bytes > config.profileMaxBytes / 4) break;
    bytes *= 4;
  }
  if (sizes.empty() || sizes.back() != config.profileMaxBytes) {
    sizes.push_back(config.profileMaxBytes);
  }
  return sizes;
}

double allGatherEstimate(ncclComm_t comm, size_t bytes, bool symmetric) {
  if (comm == nullptr || comm->nRanks <= 1) return 0.0;

  ncclTuningInput_t input = {};
  input.comm = comm;
  input.tuningMask = NCCL_TUNING_MASK_GENERAL_KERNELS |
      (symmetric ? NCCL_TUNING_MASK_SYM_KERNELS : 0);
  input.func = ncclFuncAllGather;
  input.redOp = ncclSum;
  input.devRedOp = ncclDevSum;
  input.datatype = ncclInt8;
  input.count = bytes;
  input.countMax = bytes;
  input.nBytes = bytes * (size_t)comm->nRanks;
  input.numPipeOps = 1;
  input.nWorks = 1;
  input.winRegType = symmetric
      ? ncclSymSendRegRecvReg : ncclSymSendNonregRecvNonreg;
  input.regBuff = 1;
  input.collNetSupport = comm->config.collnetEnable;
  input.nvlsSupport = comm->nvlsSupport;
  input.symAligned16B = true;
  input.minCTAs = comm->config.minCTAs;
  input.maxCTAs = comm->config.maxCTAs;
  input.CTAPolicy = comm->config.CTAPolicy;
  if (comm->cudaArch == 800 && comm->nNodes == 1 &&
      bytes >= (size_t{16} << 20)) {
    input.minCTAs = 9;
    input.maxCTAs = 9;
  }

  ncclTuningResult_t result = NCCL_TUNING_RESULT_INIT;
  if (ncclTuningCompute(&input, &result) != ncclSuccess ||
      !result.valid || !(result.timeUs > 0.0f)) {
    return 0.0;
  }
  return result.timeUs;
}

double reduceScatterEstimate(ncclComm_t comm, size_t bytes) {
  if (comm == nullptr || comm->nRanks <= 1) return 0.0;

  ncclTuningInput_t input = {};
  input.comm = comm;
  input.tuningMask = NCCL_TUNING_MASK_GENERAL_KERNELS |
      NCCL_TUNING_MASK_SYM_KERNELS;
  input.func = ncclFuncReduceScatter;
  input.redOp = ncclSum;
  input.devRedOp = ncclDevSum;
  input.datatype = ncclInt8;
  input.count = bytes / (size_t)comm->nRanks;
  input.countMax = input.count;
  input.nBytes = bytes;
  input.numPipeOps = 1;
  input.nWorks = 1;
  input.winRegType = ncclSymSendRegRecvReg;
  input.regBuff = 1;
  input.collNetSupport = comm->config.collnetEnable;
  input.nvlsSupport = comm->nvlsSupport;
  input.symAligned16B = true;
  input.minCTAs = comm->config.minCTAs;
  input.maxCTAs = comm->config.maxCTAs;
  input.CTAPolicy = comm->config.CTAPolicy;

  ncclTuningResult_t result = NCCL_TUNING_RESULT_INIT;
  if (ncclTuningCompute(&input, &result) != ncclSuccess ||
      !result.valid || !(result.timeUs > 0.0f)) {
    return 0.0;
  }
  return result.timeUs;
}

double p2pEstimate(ncclComm_t comm, size_t bytes, int peers) {
  // NCCL has no valid Send/Recv entry in ncclTuningCompute. Reuse its
  // initialized P2P channel plan and Ring link model instead.
  const ncclTopoGraph& ring = comm->graphs[NCCL_ALGO_RING];
  const int channelsPerPeer = std::max(1, comm->p2pnChannelsPerPeer);
  const int scheduledChannels = std::max(
      1, std::min(comm->p2pnChannels, channelsPerPeer * peers));
  const size_t chunks = std::max<size_t>(
      1, (bytes + (size_t)comm->p2pChunkSize - 1) /
             (size_t)comm->p2pChunkSize);
  const int activeChannels = std::min<int>(scheduledChannels, (int)chunks);
  const double channelBandwidth = comm->nNodes > 1
      ? ring.bwInter : ring.bwIntra;
  const double bandwidth = channelBandwidth * (double)activeChannels;
  const double collectiveLatency =
      comm->tuningContext.generalLatencies
          [ncclFuncAllGather][NCCL_ALGO_RING][NCCL_PROTO_SIMPLE];
  const double latency = collectiveLatency /
      (double)std::max(1, comm->nRanks - 1);
  return latency + (double)bytes / (1000.0 * bandwidth);
}

enum class TopologyOperation {
  P2p,
  AllGather,
  AllToAll,
  ReduceScatter,
};

cocclLinearModel fitTopologyModel(ncclComm_t comm,
                                  TopologyOperation operation) {
  std::vector<cocclAutotuneProfilePoint> points;
  if (comm == nullptr || comm->nRanks <= 1) return {};
  for (size_t bytes : topologySampleSizes()) {
    double timeUs = 0.0;
    if (operation == TopologyOperation::P2p) {
      timeUs = p2pEstimate(comm, bytes, 1);
    } else if (operation == TopologyOperation::AllGather) {
      timeUs = allGatherEstimate(comm, bytes, true);
    } else if (operation == TopologyOperation::AllToAll) {
      // Host AllToAll schedules every non-self peer over the shared P2P
      // channels. Model its per-rank wire volume and available concurrency.
      const int peers = std::max(1, comm->nRanks - 1);
      const size_t wireBytes = bytes * (size_t)peers /
          (size_t)comm->nRanks;
      timeUs = p2pEstimate(comm, wireBytes, peers);
    } else {
      timeUs = reduceScatterEstimate(comm, bytes);
    }
    if (!(timeUs > 0.0)) return {};
    points.push_back({(double)bytes, timeUs});
  }
  return cocclAutotuneFitLinearModel(points);
}

TopologyOperation topologyOperation(cocclAutotuneTopologyOperation operation) {
  switch (operation) {
    case cocclAutotuneTopologyOperation::AllGather:
      return TopologyOperation::AllGather;
    case cocclAutotuneTopologyOperation::AllToAll:
      return TopologyOperation::AllToAll;
    case cocclAutotuneTopologyOperation::ReduceScatter:
      return TopologyOperation::ReduceScatter;
  }
  __builtin_unreachable();
}

cocclSelectionPerformanceModel buildTopologyModel(
    const TopologyModelKey& key) {
  cocclSelectionPerformanceModel model;
  model.intraP2p = fitTopologyModel(key.intra, TopologyOperation::P2p);
  model.interP2p = fitTopologyModel(key.inter, TopologyOperation::P2p);
  model.allGather =
      fitTopologyModel(key.gather, TopologyOperation::AllGather);
  model.allToAll =
      fitTopologyModel(key.owner, TopologyOperation::AllToAll);
  return model;
}

}  // namespace

cocclSelectionPerformanceModel cocclAutotuneSnapshotPerformanceModel(
    ncclComm_t ownerComm, ncclComm_t intraComm, ncclComm_t interComm,
    ncclComm_t gatherComm) {
  const TopologyModelKey key = {ownerComm, intraComm, interComm, gatherComm};
  auto found = topologyModels.find(key);
  if (found == topologyModels.end()) {
    found = topologyModels.emplace(key, buildTopologyModel(key)).first;
  }
  return found->second;
}

cocclLinearModel cocclAutotuneSnapshotTopologyStageModel(
    ncclComm_t comm, cocclAutotuneTopologyOperation operation) {
  const auto key = std::make_pair(comm, operation);
  auto found = topologyStageModels.find(key);
  if (found == topologyStageModels.end()) {
    found = topologyStageModels.emplace(
        key, fitTopologyModel(comm, topologyOperation(operation))).first;
  }
  return found->second;
}

void cocclAutotuneTopologyCommDestroy(ncclComm_t comm) {
  for (auto item = topologyModels.begin(); item != topologyModels.end();) {
    const TopologyModelKey& key = item->first;
    item = key.owner == comm ? topologyModels.erase(item)
                             : std::next(item);
  }
  for (auto item = topologyStageModels.begin();
       item != topologyStageModels.end();) {
    item = item->first.first == comm ? topologyStageModels.erase(item)
                                     : std::next(item);
  }
}
