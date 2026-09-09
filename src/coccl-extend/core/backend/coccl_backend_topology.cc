#include "core/backend/coccl_backend_topology.h"

#include "core/tuning/coccl_autotune_internal.h"
#include "comm.h"
#include "device.h"
#include "graph.h"

#include <algorithm>

namespace {

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

double cocclBackendEstimateStage(
    ncclComm_t comm, cocclAutotuneTopologyOperation operation, size_t bytes) {
  return cocclAutotuneEstimateNcclStage(comm, operation, bytes).timeUs;
}
