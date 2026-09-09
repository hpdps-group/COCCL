#include "coccl_autotune_internal.h"
#include "core/backend/coccl_backend_topology.h"

#include "comm.h"
#include "core/config/coccl_config.h"

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

cocclLinearModel fitTopologyModel(
    ncclComm_t comm, cocclAutotuneTopologyOperation operation) {
  std::vector<cocclAutotuneProfilePoint> points;
  if (comm == nullptr || comm->nRanks <= 1) return {};
  for (size_t bytes : topologySampleSizes()) {
    const double timeUs = cocclBackendEstimateStage(comm, operation, bytes);
    if (!std::isfinite(timeUs)) return {};
    points.push_back({(double)bytes, timeUs});
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
