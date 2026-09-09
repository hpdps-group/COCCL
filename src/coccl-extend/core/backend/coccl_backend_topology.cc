#include "core/backend/coccl_backend_topology.h"

#include "core/tuning/coccl_autotune_internal.h"
#include "comm.h"
#include "device.h"
#include "core/backend/coccl_backend_collectives.h"
#include "sym_kernels.h"
#include "tuning.h"

#include <algorithm>

namespace {

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
  cocclBackendAllGatherCtaBounds(
      comm, bytes, &input.minCTAs, &input.maxCTAs);

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

double p2pEstimate(ncclComm_t comm, size_t bytes, int peers,
                   bool interNode) {
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
  const double channelBandwidth = interNode
      ? ring.bwInter : ring.bwIntra;
  const int effectiveChannels = interNode
      ? std::max(1, activeChannels / comm->localRanks)
      : activeChannels;
  const double bandwidth = channelBandwidth * (double)effectiveChannels;
  const double collectiveLatency =
      comm->tuningContext.generalLatencies
          [ncclFuncAllGather][NCCL_ALGO_RING][NCCL_PROTO_SIMPLE];
  const double latency = collectiveLatency /
      (double)std::max(1, comm->nRanks - 1);
  return latency + (double)bytes / (1000.0 * bandwidth);
}

}  // namespace

double cocclBackendEstimateStage(
    ncclComm_t comm, cocclAutotuneTopologyOperation operation, size_t bytes) {
  if (operation == cocclAutotuneTopologyOperation::P2pIntra ||
      operation == cocclAutotuneTopologyOperation::P2pInter) {
    return p2pEstimate(
        comm, bytes, 1, operation == cocclAutotuneTopologyOperation::P2pInter);
  }
  if (operation == cocclAutotuneTopologyOperation::AllGather) {
    return allGatherEstimate(comm, bytes, true);
  }
  if (operation == cocclAutotuneTopologyOperation::AllToAll) {
    const int peers = std::max(1, comm->nRanks - 1);
    const size_t wireBytes = bytes * (size_t)peers / (size_t)comm->nRanks;
    return p2pEstimate(comm, wireBytes, peers, comm->nNodes > 1);
  }
  return reduceScatterEstimate(comm, bytes);
}
