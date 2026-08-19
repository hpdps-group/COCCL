#include "coccl_autotune_internal.h"

#include "bootstrap.h"
#include "checks.h"
#include "coccl_config.h"
#include "coccl_runtime.h"
#include "comm.h"
#include "compress.h"
#include "debug.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <pthread.h>
#include <vector>

namespace {

enum cocclProfileNeed : uint32_t {
  cocclProfileNeedIntra = 1u << 0,
  cocclProfileNeedInter = 1u << 1,
  cocclProfileNeedCompressors = 1u << 2,
};

struct cocclProfileObservation {
  double timeUs = 0.0;
  double compressionRatio = 0.0;
  uint32_t active = 0;
  uint32_t valid = 0;
};

struct cocclProfiledCompressor {
  void* compressor = nullptr;
  cocclPolicyKey policy;
};

struct cocclProcessPerformanceModel {
  cocclLinearModel intraP2p;
  cocclLinearModel interP2p;
  std::vector<cocclProfiledCompressor> enabledCompressors;
  std::map<void*, cocclCodecModel> compressorModels;
  uint32_t attemptedProfiles = 0;
};

pthread_mutex_t cocclAutotuneLock = PTHREAD_MUTEX_INITIALIZER;
cocclProcessPerformanceModel cocclPerformanceModel;

double median(std::vector<double> values) {
  if (values.empty()) return 0.0;
  const size_t middle = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + middle, values.end());
  double result = values[middle];
  if ((values.size() & 1u) == 0) {
    std::nth_element(values.begin(), values.begin() + middle - 1,
                     values.end());
    result = (result + values[middle - 1]) * 0.5;
  }
  return result;
}

uint32_t localProfileNeeds(ncclComm_t comm) {
  uint32_t needs = 0;
  pthread_mutex_lock(&cocclAutotuneLock);
  const bool hasCompressors =
      !cocclPerformanceModel.enabledCompressors.empty();
  if (hasCompressors && comm->localRanks > 1 &&
      (cocclPerformanceModel.attemptedProfiles & cocclProfileNeedIntra) == 0) {
    needs |= cocclProfileNeedIntra;
  }
  if (hasCompressors && comm->nNodes > 1 &&
      (cocclPerformanceModel.attemptedProfiles & cocclProfileNeedInter) == 0) {
    needs |= cocclProfileNeedInter;
  }
  if (hasCompressors &&
      (cocclPerformanceModel.attemptedProfiles &
       cocclProfileNeedCompressors) == 0) {
    needs |= cocclProfileNeedCompressors;
  }
  pthread_mutex_unlock(&cocclAutotuneLock);
  return needs;
}

std::vector<cocclProfiledCompressor> snapshotEnabledCompressors() {
  pthread_mutex_lock(&cocclAutotuneLock);
  std::vector<cocclProfiledCompressor> result =
      cocclPerformanceModel.enabledCompressors;
  pthread_mutex_unlock(&cocclAutotuneLock);
  return result;
}

void markProfilesAttempted(uint32_t needs) {
  pthread_mutex_lock(&cocclAutotuneLock);
  cocclPerformanceModel.attemptedProfiles |= needs;
  pthread_mutex_unlock(&cocclAutotuneLock);
}

ncclResult_t collectiveProfileNeeds(ncclComm_t comm, uint32_t localNeeds,
                                    uint32_t* collectiveNeeds) {
  std::vector<uint32_t> allNeeds((size_t)comm->nRanks, 0);
  allNeeds[(size_t)comm->rank] = localNeeds;
  NCCLCHECK(bootstrapAllGather(
      comm->bootstrap, allNeeds.data(), sizeof(uint32_t)));
  uint32_t result = 0;
  for (uint32_t needs : allNeeds) result |= needs;
  *collectiveNeeds = result;
  return ncclSuccess;
}

ncclResult_t buildSampleSizes(ncclComm_t comm,
                              std::vector<size_t>* sampleSizes) {
  size_t freeBytes = 0;
  size_t totalBytes = 0;
  CUDACHECK(cudaMemGetInfo(&freeBytes, &totalBytes));

  const cocclAutotuneConfig& config = cocclGetConfig().autotune;
  const uint64_t configuredMin = (uint64_t)config.profileMinBytes;
  const uint64_t configuredMax = (uint64_t)config.profileMaxBytes;
  const uint64_t localMax =
      std::min<uint64_t>(configuredMax, (uint64_t)(freeBytes / 4));

  std::vector<uint64_t> allMax((size_t)comm->nRanks, 0);
  allMax[(size_t)comm->rank] = localMax;
  NCCLCHECK(bootstrapAllGather(
      comm->bootstrap, allMax.data(), sizeof(uint64_t)));
  uint64_t effectiveMax = configuredMax;
  for (uint64_t rankMax : allMax) {
    effectiveMax = std::min(effectiveMax, rankMax);
  }

  if (effectiveMax < configuredMin) return ncclSuccess;
  for (uint64_t bytes = configuredMin; bytes <= effectiveMax;) {
    sampleSizes->push_back((size_t)bytes);
    if (bytes > effectiveMax / 4) break;
    bytes *= 4;
  }
  if (sampleSizes->back() < effectiveMax) {
    sampleSizes->push_back((size_t)effectiveMax);
  }
  return ncclSuccess;
}

ncclResult_t aggregateObservation(ncclComm_t comm,
                                  const cocclProfileObservation& local,
                                  cocclProfileObservation* aggregate) {
  std::vector<cocclProfileObservation> observations((size_t)comm->nRanks);
  observations[(size_t)comm->rank] = local;
  NCCLCHECK(bootstrapAllGather(
      comm->bootstrap, observations.data(), sizeof(cocclProfileObservation)));

  std::vector<double> ratios;
  bool allActiveRanksValid = true;
  for (const cocclProfileObservation& observation : observations) {
    if (!observation.active) continue;
    aggregate->active = 1;
    if (!observation.valid || !std::isfinite(observation.timeUs) ||
        observation.timeUs <= 0.0) {
      allActiveRanksValid = false;
      continue;
    }
    aggregate->timeUs = std::max(aggregate->timeUs, observation.timeUs);
    if (std::isfinite(observation.compressionRatio) &&
        observation.compressionRatio > 0.0) {
      ratios.push_back(observation.compressionRatio);
    }
  }
  aggregate->valid = aggregate->active && allActiveRanksValid;
  if (aggregate->valid && !ratios.empty()) {
    aggregate->compressionRatio = median(std::move(ratios));
  }
  return ncclSuccess;
}

ncclResult_t enqueueP2pExchange(ncclComm_t comm, const void* sendBuffer,
                                void* recvBuffer, size_t bytes, int sendPeer,
                                int recvPeer, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;
  NCCLCHECKGOTO(ncclGroupStart(), ret, exit);
  {
    cocclInfo info;
    info.recvbuff = recvBuffer;
    info.count = bytes;
    info.datatype = ncclInt8;
    info.peer = recvPeer;
    info.func = ncclFuncRecv;
    info.operation = cocclOperation::SendRecv;
    info.comm = comm;
    info.stream = stream;
    NCCLCHECKGOTO(cocclReplayNativeCall(info), ret, group_exit);
  }
  {
    cocclInfo info;
    info.sendbuff = sendBuffer;
    info.count = bytes;
    info.datatype = ncclInt8;
    info.peer = sendPeer;
    info.func = ncclFuncSend;
    info.operation = cocclOperation::SendRecv;
    info.comm = comm;
    info.stream = stream;
    NCCLCHECKGOTO(cocclReplayNativeCall(info), ret, group_exit);
  }
group_exit:
  {
    const ncclResult_t groupResult = ncclGroupEnd();
    if (ret == ncclSuccess) ret = groupResult;
  }
exit:
  return ret;
}

bool topologyPeers(ncclComm_t comm, bool interNode, int* sendPeer,
                   int* recvPeer) {
  if (!interNode) {
    if (comm->localRanks <= 1) return false;
    *sendPeer =
        comm->localRankToRank[(comm->localRank + 1) % comm->localRanks];
    *recvPeer = comm->localRankToRank[
        (comm->localRank - 1 + comm->localRanks) % comm->localRanks];
    return true;
  }

  if (comm->nNodes <= 1) return false;
  int commonLocalRanks = comm->nodeRanks[0].localRanks;
  for (int node = 1; node < comm->nNodes; ++node) {
    commonLocalRanks =
        std::min(commonLocalRanks, comm->nodeRanks[node].localRanks);
  }
  if (comm->localRank >= commonLocalRanks) return false;
  const int sendNode = (comm->node + 1) % comm->nNodes;
  const int recvNode = (comm->node - 1 + comm->nNodes) % comm->nNodes;
  *sendPeer = comm->nodeRanks[sendNode].localRankToRank[comm->localRank];
  *recvPeer = comm->nodeRanks[recvNode].localRankToRank[comm->localRank];
  return true;
}

ncclResult_t profileP2p(ncclComm_t comm, bool interNode,
                        const std::vector<size_t>& sampleSizes,
                        cocclLinearModel* model) {
  ncclResult_t ret = ncclSuccess;
  void* sendBuffer = nullptr;
  void* recvBuffer = nullptr;
  cudaStream_t stream = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  int sendPeer = -1;
  int recvPeer = -1;
  const bool active = topologyPeers(
      comm, interNode, &sendPeer, &recvPeer);
  const cocclAutotuneConfig& config = cocclGetConfig().autotune;
  std::vector<cocclAutotuneProfilePoint> points;

  CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), ret, fail);
  CUDACHECKGOTO(cudaMalloc(&sendBuffer, sampleSizes.back()), ret, fail);
  CUDACHECKGOTO(cudaMalloc(&recvBuffer, sampleSizes.back()), ret, fail);
  CUDACHECKGOTO(cudaMemset(sendBuffer, 0x3f, sampleSizes.back()), ret, fail);
  CUDACHECKGOTO(
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), ret, fail);
  CUDACHECKGOTO(cudaEventCreate(&start), ret, fail);
  CUDACHECKGOTO(cudaEventCreate(&stop), ret, fail);

  for (size_t bytes : sampleSizes) {
    cocclProfileObservation local = {};
    local.active = active ? 1u : 0u;
    if (active) {
      for (int i = 0; i < config.warmup; ++i) {
        NCCLCHECKGOTO(
            enqueueP2pExchange(
                comm, sendBuffer, recvBuffer, bytes, sendPeer, recvPeer,
                stream),
            ret, fail);
      }
      CUDACHECKGOTO(cudaStreamSynchronize(stream), ret, fail);

      std::vector<double> times;
      for (int i = 0; i < config.iterations; ++i) {
        CUDACHECKGOTO(cudaEventRecord(start, stream), ret, fail);
        NCCLCHECKGOTO(
            enqueueP2pExchange(
                comm, sendBuffer, recvBuffer, bytes, sendPeer, recvPeer,
                stream),
            ret, fail);
        CUDACHECKGOTO(cudaEventRecord(stop, stream), ret, fail);
        CUDACHECKGOTO(cudaEventSynchronize(stop), ret, fail);
        float elapsedMs = 0.0f;
        CUDACHECKGOTO(
            cudaEventElapsedTime(&elapsedMs, start, stop), ret, fail);
        times.push_back((double)elapsedMs * 1000.0);
      }
      local.timeUs = median(std::move(times));
      local.valid = local.timeUs > 0.0 ? 1u : 0u;
    }

    cocclProfileObservation aggregate = {};
    NCCLCHECKGOTO(
        aggregateObservation(comm, local, &aggregate), ret, fail);
    if (aggregate.valid) {
      points.push_back({(double)bytes, aggregate.timeUs});
    }
  }
  *model = cocclAutotuneFitLinearModel(points);

exit:
  if (stop != nullptr) (void)cudaEventDestroy(stop);
  if (start != nullptr) (void)cudaEventDestroy(start);
  if (stream != nullptr) (void)cudaStreamDestroy(stream);
  if (recvBuffer != nullptr) (void)cudaFree(recvBuffer);
  if (sendBuffer != nullptr) (void)cudaFree(sendBuffer);
  return ret;
fail:
  goto exit;
}

bool runCompressorIteration(
    ncclComm_t comm, void* compressor, void* rawBuffer,
    void* compressedBuffer, size_t bytes, size_t compressedCapacity,
    cudaStream_t stream, cocclCompressorFrameMetadata* frameMetadata,
    size_t* compressedBytes) {
  const size_t elements = bytes / sizeof(float);
  const cocclCompressorView input = {
      rawBuffer, bytes, bytes, elements, 1, ncclFloat32, nullptr, 0};
  cocclCompressorView compressed = {
      compressedBuffer, compressedCapacity, 0, 0, 1, ncclInt8,
      frameMetadata, frameMetadata == nullptr ? 0 : bytes};
  if (ncclCompress(
          compressor, input, &compressed, comm->rank, stream) != ncclSuccess) {
    return false;
  }
  *compressedBytes = compressed.bytes;
  if (*compressedBytes == 0 || *compressedBytes > bytes) return false;

  const cocclCompressorView compressedInput = {
      compressed.data, compressed.bytes, compressed.bytes,
      compressed.elements, compressed.chunks, compressed.datatype,
      compressed.frameMetadata, compressed.frameStrideBytes};
  cocclCompressorView decompressed = {
      rawBuffer, bytes, 0, elements, 1, ncclFloat32, nullptr, 0};
  return ncclDecompress(
             compressor, compressedInput, &decompressed, stream) ==
      ncclSuccess;
}

ncclResult_t profileCompressor(
    ncclComm_t comm, void* compressor,
    const std::vector<size_t>& sampleSizes, cocclCodecModel* model) {
  ncclResult_t ret = ncclSuccess;
  void* rawBuffer = nullptr;
  void* compressedBuffer = nullptr;
  cocclCompressorFrameMetadata* deviceFrameMetadata = nullptr;
  cocclCompressorFrameMetadata* hostFrameMetadata = nullptr;
  cudaStream_t stream = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  std::vector<cocclAutotuneProfilePoint> points;
  std::vector<double> ratios;
  bool framed = false;
  const cocclAutotuneConfig& config = cocclGetConfig().autotune;

  framed = cocclCompressorSupports(
      compressor, cocclCompressorCapabilityFramed);

  CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), ret, fail);
  CUDACHECKGOTO(cudaMalloc(&rawBuffer, sampleSizes.back()), ret, fail);
  CUDACHECKGOTO(
      cudaMalloc(&compressedBuffer, sampleSizes.back()), ret, fail);
  if (framed) {
    CUDACHECKGOTO(
        cudaMalloc(&deviceFrameMetadata,
                   sizeof(cocclCompressorFrameMetadata)),
        ret, fail);
    CUDACHECKGOTO(
        cudaHostAlloc(&hostFrameMetadata,
                      sizeof(cocclCompressorFrameMetadata),
                      cudaHostAllocDefault),
        ret, fail);
  }
  CUDACHECKGOTO(
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), ret, fail);
  CUDACHECKGOTO(cudaEventCreate(&start), ret, fail);
  CUDACHECKGOTO(cudaEventCreate(&stop), ret, fail);

  for (size_t bytes : sampleSizes) {
    cocclProfileObservation local = {};
    local.active = 1;
    bool valid = true;
    size_t compressedBytes = 0;
    for (int i = 0; i < config.warmup && valid; ++i) {
      valid = cudaMemsetAsync(rawBuffer, 0x3f, bytes, stream) == cudaSuccess &&
          runCompressorIteration(
              comm, compressor, rawBuffer, compressedBuffer, bytes,
              sampleSizes.back(), stream, deviceFrameMetadata,
              &compressedBytes);
    }
    if (valid) valid = cudaStreamSynchronize(stream) == cudaSuccess;

    std::vector<double> times;
    for (int i = 0; i < config.iterations && valid; ++i) {
      valid = cudaMemsetAsync(rawBuffer, 0x3f, bytes, stream) == cudaSuccess &&
          cudaEventRecord(start, stream) == cudaSuccess &&
          runCompressorIteration(
              comm, compressor, rawBuffer, compressedBuffer, bytes,
              sampleSizes.back(), stream, deviceFrameMetadata,
              &compressedBytes) &&
          cudaEventRecord(stop, stream) == cudaSuccess &&
          cudaEventSynchronize(stop) == cudaSuccess;
      if (!valid) break;
      float elapsedMs = 0.0f;
      valid = cudaEventElapsedTime(&elapsedMs, start, stop) == cudaSuccess;
      if (valid) times.push_back((double)elapsedMs * 1000.0);
    }
    if (valid && framed) {
      valid = cudaMemcpy(
                  hostFrameMetadata, deviceFrameMetadata,
                  sizeof(cocclCompressorFrameMetadata),
                  cudaMemcpyDeviceToHost) == cudaSuccess &&
          hostFrameMetadata->payloadBytes > 0 &&
          hostFrameMetadata->payloadBytes <= bytes &&
          (hostFrameMetadata->encoding == cocclCompressorFrameEncoded ||
           (hostFrameMetadata->encoding == cocclCompressorFrameRaw &&
            hostFrameMetadata->payloadBytes == bytes));
      if (valid) {
        compressedBytes = (size_t)hostFrameMetadata->payloadBytes;
      }
    }
    if (valid && !times.empty() && compressedBytes > 0) {
      local.timeUs = median(std::move(times));
      local.compressionRatio = (double)bytes / (double)compressedBytes;
      local.valid = 1;
    }

    cocclProfileObservation aggregate = {};
    NCCLCHECKGOTO(
        aggregateObservation(comm, local, &aggregate), ret, fail);
    if (aggregate.valid && aggregate.compressionRatio > 0.0) {
      points.push_back({(double)bytes, aggregate.timeUs});
      ratios.push_back(aggregate.compressionRatio);
    }
  }

  model->time = cocclAutotuneFitLinearModel(points);
  model->compressionRatio = median(std::move(ratios));
  model->valid = model->time.valid && model->compressionRatio > 0.0 &&
                 std::isfinite(model->compressionRatio);

exit:
  if (stop != nullptr) (void)cudaEventDestroy(stop);
  if (start != nullptr) (void)cudaEventDestroy(start);
  if (stream != nullptr) (void)cudaStreamDestroy(stream);
  if (hostFrameMetadata != nullptr) (void)cudaFreeHost(hostFrameMetadata);
  if (deviceFrameMetadata != nullptr) (void)cudaFree(deviceFrameMetadata);
  if (compressedBuffer != nullptr) (void)cudaFree(compressedBuffer);
  if (rawBuffer != nullptr) (void)cudaFree(rawBuffer);
  return ret;
fail:
  goto exit;
}

const char* compressorName(const cocclCompressorPlugin* compressor) {
  return compressor->name;
}

const char* operationName(cocclOperation operation) {
  return operation == cocclOperation::ReduceScatter
      ? "reducescatter" : "allreduce";
}

const char* scopeName(cocclCompressionScope scope) {
  switch (scope) {
    case cocclCompressionScope::Default: return "default";
    case cocclCompressionScope::Intra: return "intra";
    case cocclCompressionScope::Inter: return "inter";
    case cocclCompressionScope::Count: return "unknown";
  }
  return "unknown";
}

void publishP2pModel(bool interNode, const cocclLinearModel& model, int rank) {
  if (!model.valid) {
    if (rank == 0) {
      WARN("COCCL failed to fit %s-node P2P profile",
           interNode ? "inter" : "intra");
    }
    return;
  }
  pthread_mutex_lock(&cocclAutotuneLock);
  if (interNode) {
    cocclPerformanceModel.interP2p = model;
  } else {
    cocclPerformanceModel.intraP2p = model;
  }
  pthread_mutex_unlock(&cocclAutotuneLock);
  if (rank == 0) {
    INFO(NCCL_TUNING,
         "COCCL profile %s P2P: time_us=%g+%g*bytes",
         interNode ? "inter" : "intra", model.alphaUs,
         model.betaUsPerByte);
  }
}

void publishCompressorModel(const cocclProfiledCompressor& profiled,
                            const cocclCodecModel& model, int rank) {
  const cocclCompressorPlugin* compressor =
      cocclCompressorDescriptor(profiled.compressor);
  if (!model.valid) {
    if (rank == 0) {
      WARN("COCCL failed to fit compressor model for %s",
           compressorName(compressor));
    }
    return;
  }
  pthread_mutex_lock(&cocclAutotuneLock);
  cocclPerformanceModel.compressorModels[profiled.compressor] = model;
  pthread_mutex_unlock(&cocclAutotuneLock);
  if (rank == 0) {
    INFO(NCCL_TUNING,
         "COCCL compressor %s policy=%s-%s model: time_us=%g+%g*bytes ratio=%g",
         compressorName(compressor), operationName(profiled.policy.operation),
         scopeName(profiled.policy.scope), model.time.alphaUs,
         model.time.betaUsPerByte, model.compressionRatio);
  }
}

void copyCompressorModelLocked(void* compressor, cocclCodecModel* model) {
  *model = {};
  const auto found = cocclPerformanceModel.compressorModels.find(compressor);
  if (found != cocclPerformanceModel.compressorModels.end()) {
    *model = found->second;
  }
}

}  // namespace

cocclSelectionPerformanceModel cocclAutotuneSnapshotPerformanceModel(
    void* defaultCompressor, void* intraCompressor, void* interCompressor,
    cocclCodecModel* defaultModel, cocclCodecModel* intraModel,
    cocclCodecModel* interModel) {
  pthread_mutex_lock(&cocclAutotuneLock);
  const cocclSelectionPerformanceModel snapshot = {
      cocclPerformanceModel.intraP2p,
      cocclPerformanceModel.interP2p,
  };
  copyCompressorModelLocked(defaultCompressor, defaultModel);
  copyCompressorModelLocked(intraCompressor, intraModel);
  copyCompressorModelLocked(interCompressor, interModel);
  pthread_mutex_unlock(&cocclAutotuneLock);
  return snapshot;
}

ncclResult_t cocclAutotuneRegisterEnabledCompressor(
    void* compressor, cocclPolicyKey policy) {
  pthread_mutex_lock(&cocclAutotuneLock);
  auto& compressors = cocclPerformanceModel.enabledCompressors;
  const auto existing = std::find_if(
      compressors.begin(), compressors.end(),
      [compressor](const cocclProfiledCompressor& item) {
        return item.compressor == compressor;
      });
  if (existing == compressors.end()) {
    compressors.push_back({compressor, policy});
  }
  pthread_mutex_unlock(&cocclAutotuneLock);
  return ncclSuccess;
}

ncclResult_t cocclAutotuneEnsureGlobalModels(ncclComm_t measurementComm) {
  if (!cocclGetConfig().autotune.enabled) return ncclSuccess;

  const uint32_t localNeeds = localProfileNeeds(measurementComm);
  if (localNeeds == 0) return ncclSuccess;

  uint32_t needs = 0;
  NCCLCHECK(collectiveProfileNeeds(measurementComm, localNeeds, &needs));
  if (needs == 0) return ncclSuccess;
  markProfilesAttempted(needs);

  std::vector<size_t> sampleSizes;
  NCCLCHECK(buildSampleSizes(measurementComm, &sampleSizes));
  if (sampleSizes.size() < 2) {
    if (measurementComm->rank == 0) {
      WARN("COCCL autotune has fewer than two profile sizes; using heuristics");
    }
    return ncclSuccess;
  }
  if (measurementComm->rank == 0) {
    INFO(NCCL_TUNING,
         "COCCL fitting global models at %zu sizes from %zu to %zu bytes",
         sampleSizes.size(), sampleSizes.front(), sampleSizes.back());
  }

  if ((needs & cocclProfileNeedIntra) != 0) {
    cocclLinearModel model;
    const ncclResult_t result =
        profileP2p(measurementComm, false, sampleSizes, &model);
    if (result == ncclSuccess) {
      publishP2pModel(false, model, measurementComm->rank);
    }
  }
  if ((needs & cocclProfileNeedInter) != 0) {
    cocclLinearModel model;
    const ncclResult_t result =
        profileP2p(measurementComm, true, sampleSizes, &model);
    if (result == ncclSuccess) {
      publishP2pModel(true, model, measurementComm->rank);
    }
  }
  if ((needs & cocclProfileNeedCompressors) != 0) {
    for (const cocclProfiledCompressor& compressor :
         snapshotEnabledCompressors()) {
      cocclCodecModel model;
      const ncclResult_t result = profileCompressor(
          measurementComm, compressor.compressor, sampleSizes, &model);
      if (result == ncclSuccess) {
        publishCompressorModel(compressor, model, measurementComm->rank);
      }
    }
  }
  return ncclSuccess;
}
