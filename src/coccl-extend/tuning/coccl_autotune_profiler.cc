#include "coccl_autotune_internal.h"

#include "bootstrap.h"
#include "checks.h"
#include "compression/compress.h"
#include "runtime/coccl_compressor_runtime.h"
#include "config/coccl_config.h"
#include "runtime/coccl_comm.h"
#include "comm.h"
#include "debug.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <utility>
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

struct cocclProcessPerformanceModel {
  cocclLinearModel intraP2p;
  cocclLinearModel interP2p;
  // Handles remain loaded for the process lifetime, so pointer identity is a
  // stable key shared by the catalog, configured policies, and fitted models.
  std::vector<const cocclCompressorPlugin*> enabledCompressors;
  std::map<const cocclCompressorPlugin*, cocclCodecModel> compressorModels;
  uint32_t attemptedProfiles = 0;
};

// Models are process-wide because they describe the device/interconnect and
// plugin implementation, not one communicator. The mutex protects publication
// and snapshots while communicators initialize concurrently.
pthread_mutex_t cocclAutotuneLock = PTHREAD_MUTEX_INITIALIZER;
cocclProcessPerformanceModel cocclPerformanceModel;

double median(std::vector<double> values) {
  if (values.empty()) return 0.0;
  size_t middle = values.size() / 2;
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

std::vector<const cocclCompressorPlugin*> snapshotEnabledCompressors() {
  pthread_mutex_lock(&cocclAutotuneLock);
  std::vector<const cocclCompressorPlugin*> compressors =
      cocclPerformanceModel.enabledCompressors;
  pthread_mutex_unlock(&cocclAutotuneLock);
  return compressors;
}

void markProfilesAttempted(uint32_t needs) {
  pthread_mutex_lock(&cocclAutotuneLock);
  cocclPerformanceModel.attemptedProfiles |= needs;
  pthread_mutex_unlock(&cocclAutotuneLock);
}

ncclResult_t collectiveProfileNeeds(ncclComm_t comm, uint32_t localNeeds,
                                    uint32_t* collectiveNeeds) {
  if (collectiveNeeds == nullptr) return ncclInvalidArgument;
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
  if (sampleSizes == nullptr) return ncclInvalidArgument;
  sampleSizes->clear();

  size_t freeBytes = 0;
  size_t totalBytes = 0;
  CUDACHECK(cudaMemGetInfo(&freeBytes, &totalBytes));

  const cocclAutotuneConfig& config = cocclGetConfig().autotune;
  uint64_t configuredMin = (uint64_t)config.profileMinBytes;
  uint64_t configuredMax = (uint64_t)config.profileMaxBytes;
  uint64_t localMax =
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
  if (!sampleSizes->empty() && sampleSizes->back() < effectiveMax &&
      effectiveMax == configuredMax) {
    sampleSizes->push_back((size_t)effectiveMax);
  }
  return ncclSuccess;
}

ncclResult_t aggregateObservation(ncclComm_t comm,
                                  const cocclProfileObservation& local,
                                  cocclProfileObservation* aggregate) {
  if (aggregate == nullptr) return ncclInvalidArgument;
  std::vector<cocclProfileObservation> observations((size_t)comm->nRanks);
  observations[(size_t)comm->rank] = local;
  NCCLCHECK(bootstrapAllGather(
      comm->bootstrap, observations.data(), sizeof(cocclProfileObservation)));

  aggregate->timeUs = 0.0;
  aggregate->compressionRatio = 0.0;
  aggregate->active = 0;
  aggregate->valid = 0;
  std::vector<double> ratios;
  bool allActiveRanksValid = true;
  // Collective latency is bounded by the slowest participating rank, while a
  // median ratio avoids one rank's encoded-size outlier skewing the model.
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
  NCCLCHECKGOTO(
      ncclRecvNaive(
          recvBuffer, bytes, ncclInt8, recvPeer, comm, stream),
      ret, group_exit);
  NCCLCHECKGOTO(
      ncclSendNaive(
          sendBuffer, bytes, ncclInt8, sendPeer, comm, stream),
      ret, group_exit);
group_exit:
  {
    ncclResult_t groupRet = ncclGroupEnd();
    if (ret == ncclSuccess) ret = groupRet;
  }
exit:
  return ret;
}

bool topologyPeers(ncclComm_t comm, bool interNode, int* sendPeer,
                   int* recvPeer) {
  if (sendPeer == nullptr || recvPeer == nullptr) return false;
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
  int sendNode = (comm->node + 1) % comm->nNodes;
  int recvNode = (comm->node - 1 + comm->nNodes) % comm->nNodes;
  *sendPeer = comm->nodeRanks[sendNode].localRankToRank[comm->localRank];
  *recvPeer = comm->nodeRanks[recvNode].localRankToRank[comm->localRank];
  return true;
}

ncclResult_t profileP2p(ncclComm_t comm, bool interNode,
                        const std::vector<size_t>& sampleSizes,
                        cocclLinearModel* model) {
  if (model == nullptr || sampleSizes.size() < 2) {
    return ncclInvalidArgument;
  }
  ncclResult_t ret = ncclSuccess;
  void* sendBuffer = nullptr;
  void* recvBuffer = nullptr;
  cudaStream_t stream = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  int sendPeer = -1;
  int recvPeer = -1;
  bool active = topologyPeers(
      comm, interNode, &sendPeer, &recvPeer);
  const cocclAutotuneConfig& config = cocclGetConfig().autotune;
  int warmup = config.warmup;
  int iterations = config.iterations;
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
      for (int i = 0; i < warmup; ++i) {
        NCCLCHECKGOTO(
            enqueueP2pExchange(
                comm, sendBuffer, recvBuffer, bytes, sendPeer, recvPeer,
                stream),
            ret, fail);
      }
      CUDACHECKGOTO(cudaStreamSynchronize(stream), ret, fail);

      std::vector<double> times;
      times.reserve((size_t)iterations);
      for (int i = 0; i < iterations; ++i) {
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
    ncclComm_t measurementComm, const cocclCompressorHandle& compressor,
    void* rawBuffer, void* compressedStorage, size_t bytes,
    size_t compressedCapacity, cudaStream_t stream,
    size_t* compressedBytes) {
  if (!compressor || compressedBytes == nullptr ||
      bytes % sizeof(float) != 0) {
    return false;
  }
  size_t elementCount = bytes / sizeof(float);
  const cocclCompressorDataView input = {
      rawBuffer, bytes, elementCount, 1, ncclFloat32};
  cocclCompressorOutputView compressed = {
      compressedStorage, compressedCapacity, 0, 0, 1, ncclInt8};
  if (ncclCompress(compressor, input, &compressed, measurementComm->rank,
                   stream) != ncclSuccess) {
    return false;
  }
  *compressedBytes = compressed.bytes;
  if (*compressedBytes == 0 || *compressedBytes > bytes) return false;

  // Compression and decompression use the same stream, so decompression may
  // safely overwrite the raw scratch after compression has consumed it.
  const cocclCompressorDataView compressedInput = {
      compressed.data, compressed.bytes, compressed.elements,
      compressed.chunks, compressed.datatype};
  cocclCompressorOutputView decompressed = {
      rawBuffer, bytes, 0, elementCount, 1, ncclFloat32};
  return ncclDecompress(compressor, compressedInput, &decompressed,
                        stream) == ncclSuccess;
}

ncclResult_t profileCompressor(
    ncclComm_t measurementComm, const cocclCompressorPlugin* compressor,
    const std::vector<size_t>& sampleSizes, cocclCodecModel* model) {
  if (measurementComm == nullptr || compressor == nullptr ||
      compressor->parseConfig == nullptr || model == nullptr ||
      sampleSizes.size() < 2) {
    return ncclInvalidArgument;
  }
  ncclResult_t ret = ncclSuccess;
  void* rawBuffer = nullptr;
  void* compressedBuffer = nullptr;
  void* config = nullptr;
  cocclCompressorHandle compressorHandle;
  cudaStream_t stream = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  const cocclAutotuneConfig& autotune = cocclGetConfig().autotune;
  int warmup = autotune.warmup;
  int iterations = autotune.iterations;
  std::vector<cocclAutotuneProfilePoint> points;
  std::vector<double> ratios;
  size_t compressedCapacity = 0;

  // Profiling intentionally uses the plugin's intrinsic defaults rather than
  // any operation-specific policy stored in the TOML configuration.
  {
    const cocclConfigView emptyView = {nullptr, 0};
    const cocclCompressorConfigContext context = {
        cocclCompressorConfigDefault, measurementComm->nNodes,
        measurementComm->localRanks};
    NCCLCHECKGOTO(
        compressor->parseConfig(
            &emptyView, &context, &config, nullptr, 0),
        ret, fail);
  }
  NCCLCHECKGOTO(
      cocclCreateCompressorHandle(
          measurementComm, compressor, config, &compressorHandle),
      ret, fail);
  config = nullptr;

  CUDACHECKGOTO(cudaSetDevice(measurementComm->cudaDev), ret, fail);
  CUDACHECKGOTO(cudaMalloc(&rawBuffer, sampleSizes.back()), ret, fail);
  if (sampleSizes.back() > SIZE_MAX / 2) {
    ret = ncclInvalidArgument;
    goto fail;
  }
  compressedCapacity = sampleSizes.back() * 2;
  CUDACHECKGOTO(
      cudaMalloc(&compressedBuffer, compressedCapacity), ret, fail);
  CUDACHECKGOTO(
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), ret, fail);
  CUDACHECKGOTO(cudaEventCreate(&start), ret, fail);
  CUDACHECKGOTO(cudaEventCreate(&stop), ret, fail);

  for (size_t bytes : sampleSizes) {
    cocclProfileObservation local = {};
    local.active = 1;
    bool valid = true;
    size_t compressedBytes = 0;
    for (int i = 0; i < warmup && valid; ++i) {
      if (cudaMemsetAsync(rawBuffer, 0x3f, bytes, stream) != cudaSuccess) {
        valid = false;
        break;
      }
      valid = runCompressorIteration(
          measurementComm, compressorHandle, rawBuffer, compressedBuffer,
          bytes, compressedCapacity, stream, &compressedBytes);
    }
    if (valid && cudaStreamSynchronize(stream) != cudaSuccess) valid = false;

    std::vector<double> times;
    times.reserve((size_t)iterations);
    for (int i = 0; i < iterations && valid; ++i) {
      // Recreate the same input outside the timed interval because a lossy
      // decompressor overwrites rawBuffer at the end of the previous iteration.
      if (cudaMemsetAsync(rawBuffer, 0x3f, bytes, stream) != cudaSuccess ||
          cudaEventRecord(start, stream) != cudaSuccess) {
        valid = false;
        break;
      }
      valid = runCompressorIteration(
          measurementComm, compressorHandle, rawBuffer, compressedBuffer,
          bytes, compressedCapacity, stream, &compressedBytes);
      if (!valid || cudaEventRecord(stop, stream) != cudaSuccess ||
          cudaEventSynchronize(stop) != cudaSuccess) {
        valid = false;
        break;
      }
      float elapsedMs = 0.0f;
      if (cudaEventElapsedTime(&elapsedMs, start, stop) != cudaSuccess) {
        valid = false;
        break;
      }
      times.push_back((double)elapsedMs * 1000.0);
    }
    if (valid && !times.empty() && compressedBytes > 0) {
      local.timeUs = median(std::move(times));
      local.compressionRatio = (double)bytes / (double)compressedBytes;
      local.valid = 1;
    }

    cocclProfileObservation aggregate = {};
    NCCLCHECKGOTO(
        aggregateObservation(measurementComm, local, &aggregate), ret, fail);
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
  if (compressedBuffer != nullptr) (void)cudaFree(compressedBuffer);
  if (rawBuffer != nullptr) (void)cudaFree(rawBuffer);
  if (config != nullptr) compressor->destroyConfig(config);
  return ret;
fail:
  goto exit;
}

const char* compressorName(const cocclCompressorPlugin* compressor) {
  return compressor != nullptr && compressor->name != nullptr
             ? compressor->name
             : "unknown";
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

void publishCompressorModel(const cocclCompressorPlugin* compressor,
                            const cocclCodecModel& model, int rank) {
  if (!model.valid) {
    if (rank == 0) {
      WARN("COCCL failed to fit compressor model for %s",
           compressorName(compressor));
    }
    return;
  }
  pthread_mutex_lock(&cocclAutotuneLock);
  cocclPerformanceModel.compressorModels.insert_or_assign(compressor, model);
  pthread_mutex_unlock(&cocclAutotuneLock);
  if (rank == 0) {
    INFO(NCCL_TUNING,
         "COCCL compressor %s model: time_us=%g+%g*bytes ratio=%g",
         compressorName(compressor), model.time.alphaUs,
         model.time.betaUsPerByte, model.compressionRatio);
  }
}

void copyCompressorModelLocked(const cocclCompressorHandle& handle,
                               cocclCodecModel* model) {
  if (model == nullptr) return;
  *model = {};
  const cocclCompressorPlugin* compressor =
      cocclCompressorDescriptor(handle);
  auto found = cocclPerformanceModel.compressorModels.find(compressor);
  if (found != cocclPerformanceModel.compressorModels.end() &&
      found->second.valid) {
    *model = found->second;
  }
}

}  // namespace

cocclSelectionPerformanceModel cocclAutotuneSnapshotPerformanceModel(
    const cocclCompressorHandle& primary,
    const cocclCompressorHandle* secondary,
    cocclCodecModel* primaryModel, cocclCodecModel* secondaryModel) {
  // Copy P2P and codec models in one critical section so one selection never
  // combines values from different publication generations.
  pthread_mutex_lock(&cocclAutotuneLock);
  cocclSelectionPerformanceModel snapshot = {
      cocclPerformanceModel.intraP2p,
      cocclPerformanceModel.interP2p,
  };
  copyCompressorModelLocked(primary, primaryModel);
  if (secondary != nullptr) {
    copyCompressorModelLocked(*secondary, secondaryModel);
  } else if (secondaryModel != nullptr) {
    *secondaryModel = {};
  }
  pthread_mutex_unlock(&cocclAutotuneLock);
  return snapshot;
}

ncclResult_t cocclAutotuneRegisterEnabledCompressor(
    const cocclCompressorPlugin* compressor) {
  if (compressor == nullptr) return ncclInvalidArgument;

  pthread_mutex_lock(&cocclAutotuneLock);
  auto& compressors = cocclPerformanceModel.enabledCompressors;
  if (std::find(compressors.begin(), compressors.end(), compressor) ==
      compressors.end()) {
    compressors.push_back(compressor);
  }
  pthread_mutex_unlock(&cocclAutotuneLock);
  return ncclSuccess;
}

ncclResult_t cocclAutotuneEnsureGlobalModels(ncclComm_t measurementComm) {
  if (measurementComm == nullptr || measurementComm->nRanks <= 0 ||
      measurementComm->rank < 0 ||
      measurementComm->rank >= measurementComm->nRanks) {
    return ncclInvalidArgument;
  }
  if (!cocclGetConfig().autotune.enabled) return ncclSuccess;

  // This local check is the steady-state path. Once global compressor/intra
  // models are attempted, only the first eligible multi-node communicator can
  // reach the collective coordination below to add an inter-node model.
  uint32_t localNeeds = localProfileNeeds(measurementComm);
  if (localNeeds == 0) return ncclSuccess;

  uint32_t needs = 0;
  NCCLCHECK(collectiveProfileNeeds(measurementComm, localNeeds, &needs));
  if (needs == 0) return ncclSuccess;

  // Every rank has entered the collective needs exchange before the categories
  // are claimed, so all local init workers execute the same profile sequence.
  markProfilesAttempted(needs);

  std::vector<size_t> sampleSizes;
  ncclResult_t sizeResult =
      buildSampleSizes(measurementComm, &sampleSizes);
  if (sizeResult != ncclSuccess || sampleSizes.size() < 2) {
    if (measurementComm->rank == 0) {
      WARN("COCCL autotune has fewer than two safe profile sizes; using heuristic selection");
    }
    return ncclSuccess;
  }
  if (measurementComm->rank == 0) {
    INFO(NCCL_TUNING,
         "COCCL fitting global models at %zu sizes from %zu to %zu bytes using comm %p",
         sampleSizes.size(), sampleSizes.front(), sampleSizes.back(),
         measurementComm);
  }

  if (needs & cocclProfileNeedIntra) {
    cocclLinearModel model = {};
    ncclResult_t result =
        profileP2p(measurementComm, false, sampleSizes, &model);
    if (result == ncclSuccess) {
      publishP2pModel(false, model, measurementComm->rank);
    } else if (measurementComm->rank == 0) {
      WARN("COCCL intra-node P2P profiling failed: %d", result);
    }
  }
  if (needs & cocclProfileNeedInter) {
    cocclLinearModel model = {};
    ncclResult_t result =
        profileP2p(measurementComm, true, sampleSizes, &model);
    if (result == ncclSuccess) {
      publishP2pModel(true, model, measurementComm->rank);
    } else if (measurementComm->rank == 0) {
      WARN("COCCL inter-node P2P profiling failed: %d", result);
    }
  }

  if (needs & cocclProfileNeedCompressors) {
    std::vector<const cocclCompressorPlugin*> compressors =
        snapshotEnabledCompressors();
    for (const cocclCompressorPlugin* compressor : compressors) {
      cocclCodecModel model = {};
      ncclResult_t result = profileCompressor(
          measurementComm, compressor, sampleSizes, &model);
      if (result == ncclSuccess) {
        publishCompressorModel(compressor, model, measurementComm->rank);
      } else if (measurementComm->rank == 0) {
        WARN("COCCL compressor modeling failed for %s: %d",
             compressorName(compressor), result);
      }
    }
  }
  return ncclSuccess;
}
