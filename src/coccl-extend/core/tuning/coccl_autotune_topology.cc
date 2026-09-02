#include "coccl_autotune_internal.h"

#include "comm.h"
#include "core/config/coccl_config.h"
#include "device.h"
#include "graph.h"

#include <algorithm>
#include <iterator>
#include <map>
#include <tuple>

namespace {

// ncclTopoGetAlgoTime models the collective kernel but not the per-slice
// Host/stream submission cost paid by the 2.27 pipeline AllGather path.
constexpr double kPipelineAllGatherDispatchUs = 35.0;

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

int collectiveFunction(cocclAutotuneTopologyOperation operation) {
  return operation == cocclAutotuneTopologyOperation::AllGather
      ? ncclFuncAllGather : ncclFuncReduceScatter;
}

bool collectiveCandidateAvailable(
    ncclComm_t comm, int function, int algorithm, int protocol) {
  if (comm->bandwidths[function][algorithm][protocol] <= 0.0f) return false;
  if (algorithm != NCCL_ALGO_RING && algorithm != NCCL_ALGO_PAT &&
      algorithm != NCCL_ALGO_NVLS &&
      algorithm != NCCL_ALGO_COLLNET_DIRECT) {
    return false;
  }
  if ((algorithm == NCCL_ALGO_PAT || algorithm == NCCL_ALGO_NVLS ||
       algorithm == NCCL_ALGO_COLLNET_DIRECT) &&
      protocol != NCCL_PROTO_SIMPLE) {
    return false;
  }
  if (algorithm == NCCL_ALGO_COLLNET_DIRECT) {
    if (comm->config.collnetEnable != 1 ||
        comm->maxLocalRanks > NCCL_MAX_DIRECT_ARITY + 1) {
      return false;
    }
    if (function == ncclFuncReduceScatter &&
        comm->collNetSupportMatrix[ncclSum][ncclInt8] == 0) {
      return false;
    }
  }
  if (algorithm == NCCL_ALGO_NVLS) {
    if (!comm->nvlsSupport || comm->localRanks > NCCL_MAX_NVLS_ARITY) {
      return false;
    }
    if (comm->nNodes > 1 && comm->config.collnetEnable != 1) return false;
  }
  return true;
}

int collectiveChannels(
    ncclComm_t comm, int algorithm, int protocol, size_t bytes) {
  if (algorithm == NCCL_ALGO_NVLS) return std::max(1, comm->nvlsChannels);
  int channels = std::max(1, comm->nChannels);
  const int threads = comm->maxThreads[algorithm][protocol];
  const int threshold = comm->threadThresholds[algorithm][protocol];
  while (channels > 1 &&
         bytes < (size_t)channels * (size_t)threads * (size_t)threshold) {
    --channels;
  }
  return channels;
}

cocclNcclCostEstimate collectiveEstimate(
    ncclComm_t comm, cocclAutotuneTopologyOperation operation,
    size_t bytes) {
  cocclNcclCostEstimate best;
  const int function = collectiveFunction(operation);
  size_t ncclBytes = bytes;
  if (operation == cocclAutotuneTopologyOperation::AllGather) {
    if (bytes > SIZE_MAX / (size_t)comm->nRanks) return best;
    ncclBytes *= (size_t)comm->nRanks;
  }

  for (int algorithm = 0; algorithm < NCCL_NUM_ALGORITHMS; ++algorithm) {
    for (int protocol = 0; protocol < NCCL_NUM_PROTOCOLS; ++protocol) {
      if (!collectiveCandidateAvailable(
              comm, function, algorithm, protocol)) {
        continue;
      }
      float timeUs = -1.0f;
      if (ncclTopoGetAlgoTime(
              comm, function, algorithm, protocol, ncclBytes, 1,
              &timeUs) != ncclSuccess ||
          !(timeUs >= 0.0f) || (double)timeUs >= best.timeUs) {
        continue;
      }
      best.timeUs = timeUs;
      best.algorithm = algorithm;
      best.protocol = protocol;
      best.channels = collectiveChannels(
          comm, algorithm, protocol, ncclBytes);
    }
  }
  return best;
}

cocclNcclCostEstimate p2pEstimate(
    ncclComm_t comm, size_t bytes, int peers, bool interNode) {
  cocclNcclCostEstimate estimate;
  const ncclTopoGraph& ring = comm->graphs[NCCL_ALGO_RING];
  const int channelsPerPeer = std::max(1, comm->p2pnChannelsPerPeer);
  const int scheduledChannels = std::max(
      1, std::min(comm->p2pnChannels, channelsPerPeer * peers));
  const size_t chunks = std::max<size_t>(
      1, (bytes + (size_t)comm->p2pChunkSize - 1) /
             (size_t)comm->p2pChunkSize);
  const int activeChannels = std::min<int>(scheduledChannels, (int)chunks);
  const double channelBandwidth = interNode ? ring.bwInter : ring.bwIntra;
  if (!(channelBandwidth > 0.0)) return estimate;
  const int effectiveChannels = interNode
      ? std::max(1, activeChannels / comm->localRanks)
      : activeChannels;

  const double latency =
      comm->latencies[ncclFuncAllGather][NCCL_ALGO_RING][NCCL_PROTO_SIMPLE] /
      (double)std::max(1, comm->nRanks - 1);
  estimate.timeUs = latency +
      (double)bytes / (1000.0 * channelBandwidth * effectiveChannels);
  estimate.algorithm = NCCL_ALGO_RING;
  estimate.protocol = NCCL_PROTO_SIMPLE;
  estimate.channels = effectiveChannels;
  return estimate;
}

cocclLinearModel fitTopologyModel(
    ncclComm_t comm, cocclAutotuneTopologyOperation operation) {
  std::vector<cocclAutotuneProfilePoint> points;
  if (comm == nullptr || comm->nRanks <= 1) return {};
  for (size_t bytes : topologySampleSizes()) {
    const cocclNcclCostEstimate estimate =
        cocclAutotuneEstimateNcclStage(comm, operation, bytes);
    if (!std::isfinite(estimate.timeUs)) return {};
    points.push_back({(double)bytes, estimate.timeUs});
  }
  return cocclAutotuneFitLinearModel(points);
}

cocclSelectionPerformanceModel buildTopologyModel(
    const TopologyModelKey& key) {
  cocclSelectionPerformanceModel model;
  model.intraP2p = fitTopologyModel(
      key.intra, cocclAutotuneTopologyOperation::P2pIntra);
  model.interP2p = fitTopologyModel(
      key.inter, cocclAutotuneTopologyOperation::P2pInter);
  model.allGather = cocclAutotuneSnapshotTopologyStageModel(
      key.gather, cocclAutotuneTopologyOperation::AllGather);
  model.allToAll = fitTopologyModel(
      key.owner, cocclAutotuneTopologyOperation::AllToAll);
  model.allGatherIntra = cocclAutotuneSnapshotTopologyStageModel(
      key.intra, cocclAutotuneTopologyOperation::AllGather);
  model.allGatherInter = cocclAutotuneSnapshotTopologyStageModel(
      key.inter, cocclAutotuneTopologyOperation::AllGather);
  return model;
}

}  // namespace

cocclNcclCostEstimate cocclAutotuneEstimateNcclStage(
    ncclComm_t comm, cocclAutotuneTopologyOperation operation,
    size_t bytes) {
  if (comm == nullptr || comm->nRanks <= 1) {
    cocclNcclCostEstimate estimate;
    estimate.timeUs = 0.0;
    estimate.channels = 1;
    return estimate;
  }
  if (operation == cocclAutotuneTopologyOperation::P2pIntra ||
      operation == cocclAutotuneTopologyOperation::P2pInter) {
    return p2pEstimate(
        comm, bytes, 1,
        operation == cocclAutotuneTopologyOperation::P2pInter);
  }
  if (operation == cocclAutotuneTopologyOperation::AllToAll) {
    const int peers = std::max(1, comm->nRanks - 1);
    const size_t wireBytes = bytes * (size_t)peers /
        (size_t)comm->nRanks;
    return p2pEstimate(comm, wireBytes, peers, comm->nNodes > 1);
  }
  return collectiveEstimate(comm, operation, bytes);
}

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
        key, fitTopologyModel(comm, operation)).first;
  }
  cocclLinearModel model = found->second;
  if (model.valid &&
      operation == cocclAutotuneTopologyOperation::AllGather) {
    model.alphaUs += kPipelineAllGatherDispatchUs;
    for (size_t i = 0; i < model.sampleCount; ++i) {
      model.sampleTimeUs[i] += kPipelineAllGatherDispatchUs;
    }
  }
  return model;
}

void cocclAutotuneTopologyCommDestroy(ncclComm_t comm) {
  for (auto item = topologyModels.begin(); item != topologyModels.end();) {
    const TopologyModelKey& key = item->first;
    item = (key.owner == comm || key.intra == comm || key.inter == comm ||
            key.gather == comm)
        ? topologyModels.erase(item) : std::next(item);
  }
  for (auto item = topologyStageModels.begin();
       item != topologyStageModels.end();) {
    item = item->first.first == comm ? topologyStageModels.erase(item)
                                     : std::next(item);
  }
}
