#include "coccl_autotune.h"

#include "bootstrap.h"
#include "checks.h"
#include "coccl_runtime.h"
#include "comm.h"
#include "compress.h"
#include "debug.h"
#include "param.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <pthread.h>
#include <string.h>
#include <strings.h>
#include <utility>
#include <vector>

NCCL_PARAM(CocclAutotune, "COCCL_AUTOTUNE", 1);
NCCL_PARAM(CocclProfileMinBytes, "COCCL_PROFILE_MIN_BYTES", 256ll * 1024);
NCCL_PARAM(CocclProfileMaxBytes, "COCCL_PROFILE_MAX_BYTES", 8ll * 1024 * 1024 * 1024);
NCCL_PARAM(CocclProfileWarmup, "COCCL_PROFILE_WARMUP", 3);
NCCL_PARAM(CocclProfileIters, "COCCL_PROFILE_ITERS", 10);

namespace {

enum cocclProfileNeed : uint32_t {
  cocclProfileNeedIntra = 1u << 0,
  cocclProfileNeedInter = 1u << 1,
  cocclProfileNeedReduceScatter = 1u << 2,
  cocclProfileNeedReduceScatterInter = 1u << 3,
  cocclProfileNeedAllReduce = 1u << 4,
  cocclProfileNeedAllReduceInter = 1u << 5,
};

struct cocclProfilePoint {
  double bytes = 0.0;
  double timeUs = 0.0;
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
  std::map<ncclCommOp_t, cocclCodecModel> codecModels;
  uint32_t attemptedProfiles = 0;
};

pthread_mutex_t cocclAutotuneLock = PTHREAD_MUTEX_INITIALIZER;
cocclProcessPerformanceModel cocclPerformanceModel;

static double median(std::vector<double> values) {
  if (values.empty()) return 0.0;
  size_t middle = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + middle, values.end());
  double result = values[middle];
  if ((values.size() & 1u) == 0) {
    std::nth_element(values.begin(), values.begin() + middle - 1, values.end());
    result = (result + values[middle - 1]) * 0.5;
  }
  return result;
}

static cocclLinearModel fitLinearModel(const std::vector<cocclProfilePoint>& points) {
  cocclLinearModel model = {};
  if (points.size() < 2) return model;

  double meanX = 0.0;
  double meanY = 0.0;
  for (const cocclProfilePoint& point : points) {
    if (!std::isfinite(point.bytes) || !std::isfinite(point.timeUs) ||
        point.bytes <= 0.0 || point.timeUs <= 0.0) {
      return model;
    }
    meanX += point.bytes;
    meanY += point.timeUs;
  }
  meanX /= (double)points.size();
  meanY /= (double)points.size();

  double covariance = 0.0;
  double variance = 0.0;
  for (const cocclProfilePoint& point : points) {
    double deltaX = point.bytes - meanX;
    covariance += deltaX * (point.timeUs - meanY);
    variance += deltaX * deltaX;
  }
  if (!(variance > 0.0) || !std::isfinite(variance)) return model;

  model.betaUsPerByte = std::max(0.0, covariance / variance);
  model.alphaUs = std::max(0.0, meanY - model.betaUsPerByte * meanX);
  model.valid = std::isfinite(model.alphaUs) &&
                std::isfinite(model.betaUsPerByte);
  return model;
}

static double predict(const cocclLinearModel& model, double bytes) {
  if (!model.valid || !std::isfinite(bytes) || bytes < 0.0) {
    return std::numeric_limits<double>::infinity();
  }
  return std::max(0.0, model.alphaUs + model.betaUsPerByte * bytes);
}

static uint32_t localProfileNeeds(ncclComm_t comm,
                                  const cocclAutotuneProfileOptions* options) {
  uint32_t needs = 0;
  pthread_mutex_lock(&cocclAutotuneLock);
  if (comm->localRanks > 1 &&
      (cocclPerformanceModel.attemptedProfiles & cocclProfileNeedIntra) == 0) {
    needs |= cocclProfileNeedIntra;
  }
  if (comm->nNodes > 1 &&
      (cocclPerformanceModel.attemptedProfiles & cocclProfileNeedInter) == 0) {
    needs |= cocclProfileNeedInter;
  }
  if (options->profileReduceScatter) {
    if ((cocclPerformanceModel.attemptedProfiles &
         cocclProfileNeedReduceScatter) == 0) {
      needs |= cocclProfileNeedReduceScatter;
    }
    if (comm->nNodes > 1 &&
        (cocclPerformanceModel.attemptedProfiles &
         cocclProfileNeedReduceScatterInter) == 0) {
      needs |= cocclProfileNeedReduceScatterInter;
    }
  }
  if (options->profileAllReduce) {
    if ((cocclPerformanceModel.attemptedProfiles &
         cocclProfileNeedAllReduce) == 0) {
      needs |= cocclProfileNeedAllReduce;
    }
    if (comm->nNodes > 1 &&
        (cocclPerformanceModel.attemptedProfiles &
         cocclProfileNeedAllReduceInter) == 0) {
      needs |= cocclProfileNeedAllReduceInter;
    }
  }
  pthread_mutex_unlock(&cocclAutotuneLock);
  return needs;
}

static void markProfilesAttempted(uint32_t needs) {
  pthread_mutex_lock(&cocclAutotuneLock);
  cocclPerformanceModel.attemptedProfiles |= needs;
  pthread_mutex_unlock(&cocclAutotuneLock);
}

static ncclResult_t collectiveProfileNeeds(ncclComm_t comm, uint32_t localNeeds,
                                           uint32_t* collectiveNeeds) {
  if (collectiveNeeds == nullptr) return ncclInvalidArgument;
  std::vector<uint32_t> allNeeds((size_t)comm->nRanks, 0);
  allNeeds[(size_t)comm->rank] = localNeeds;
  NCCLCHECK(bootstrapAllGather(comm->bootstrap, allNeeds.data(), sizeof(uint32_t)));
  uint32_t result = 0;
  for (uint32_t needs : allNeeds) result |= needs;
  *collectiveNeeds = result;
  return ncclSuccess;
}

static ncclResult_t buildSampleSizes(ncclComm_t comm,
                                     std::vector<size_t>* sampleSizes) {
  if (sampleSizes == nullptr) return ncclInvalidArgument;
  sampleSizes->clear();

  size_t freeBytes = 0;
  size_t totalBytes = 0;
  CUDACHECK(cudaMemGetInfo(&freeBytes, &totalBytes));

  uint64_t configuredMin = (uint64_t)std::max<int64_t>(1, ncclParamCocclProfileMinBytes());
  uint64_t configuredMax = (uint64_t)std::max<int64_t>(1, ncclParamCocclProfileMaxBytes());
  uint64_t localMax = std::min<uint64_t>(configuredMax, (uint64_t)(freeBytes / 4));

  std::vector<uint64_t> allMax((size_t)comm->nRanks, 0);
  allMax[(size_t)comm->rank] = localMax;
  NCCLCHECK(bootstrapAllGather(comm->bootstrap, allMax.data(), sizeof(uint64_t)));
  uint64_t effectiveMax = configuredMax;
  for (uint64_t rankMax : allMax) effectiveMax = std::min(effectiveMax, rankMax);

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

static ncclResult_t aggregateObservation(ncclComm_t comm,
                                         const cocclProfileObservation& local,
                                         cocclProfileObservation* aggregate) {
  if (aggregate == nullptr) return ncclInvalidArgument;
  std::vector<cocclProfileObservation> observations((size_t)comm->nRanks);
  observations[(size_t)comm->rank] = local;
  NCCLCHECK(bootstrapAllGather(comm->bootstrap, observations.data(),
                               sizeof(cocclProfileObservation)));

  aggregate->timeUs = 0.0;
  aggregate->compressionRatio = 0.0;
  aggregate->active = 0;
  aggregate->valid = 0;
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

static ncclResult_t enqueueP2pExchange(ncclComm_t comm, const void* sendBuffer,
                                       void* recvBuffer, size_t bytes, int sendPeer,
                                       int recvPeer, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;
  NCCLCHECKGOTO(ncclGroupStart(), ret, exit);
  NCCLCHECKGOTO(ncclRecvNaive(recvBuffer, bytes, ncclInt8, recvPeer, comm, stream),
                ret, group_exit);
  NCCLCHECKGOTO(ncclSendNaive(sendBuffer, bytes, ncclInt8, sendPeer, comm, stream),
                ret, group_exit);
group_exit:
  {
    ncclResult_t groupRet = ncclGroupEnd();
    if (ret == ncclSuccess) ret = groupRet;
  }
exit:
  return ret;
}

static bool topologyPeers(ncclComm_t comm, bool interNode, int* sendPeer,
                          int* recvPeer) {
  if (sendPeer == nullptr || recvPeer == nullptr) return false;
  if (!interNode) {
    if (comm->localRanks <= 1) return false;
    *sendPeer = comm->localRankToRank[(comm->localRank + 1) % comm->localRanks];
    *recvPeer = comm->localRankToRank[(comm->localRank - 1 + comm->localRanks) %
                                      comm->localRanks];
    return true;
  }

  if (comm->nNodes <= 1) return false;
  int commonLocalRanks = comm->nodeRanks[0].localRanks;
  for (int node = 1; node < comm->nNodes; ++node) {
    commonLocalRanks = std::min(commonLocalRanks,
                                comm->nodeRanks[node].localRanks);
  }
  if (comm->localRank >= commonLocalRanks) return false;
  int sendNode = (comm->node + 1) % comm->nNodes;
  int recvNode = (comm->node - 1 + comm->nNodes) % comm->nNodes;
  *sendPeer = comm->nodeRanks[sendNode].localRankToRank[comm->localRank];
  *recvPeer = comm->nodeRanks[recvNode].localRankToRank[comm->localRank];
  return true;
}

static ncclResult_t profileP2p(ncclComm_t comm, bool interNode,
                               const std::vector<size_t>& sampleSizes,
                               cocclLinearModel* model) {
  if (model == nullptr || sampleSizes.size() < 2) return ncclInvalidArgument;
  ncclResult_t ret = ncclSuccess;
  void* sendBuffer = nullptr;
  void* recvBuffer = nullptr;
  cudaStream_t stream = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  int sendPeer = -1;
  int recvPeer = -1;
  bool active = topologyPeers(comm, interNode, &sendPeer, &recvPeer);
  int warmup = (int)std::max<int64_t>(0, ncclParamCocclProfileWarmup());
  int iterations = (int)std::max<int64_t>(1, ncclParamCocclProfileIters());
  std::vector<cocclProfilePoint> points;

  CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), ret, fail);
  CUDACHECKGOTO(cudaMalloc(&sendBuffer, sampleSizes.back()), ret, fail);
  CUDACHECKGOTO(cudaMalloc(&recvBuffer, sampleSizes.back()), ret, fail);
  CUDACHECKGOTO(cudaMemset(sendBuffer, 0x3f, sampleSizes.back()), ret, fail);
  CUDACHECKGOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), ret, fail);
  CUDACHECKGOTO(cudaEventCreate(&start), ret, fail);
  CUDACHECKGOTO(cudaEventCreate(&stop), ret, fail);

  for (size_t bytes : sampleSizes) {
    cocclProfileObservation local = {};
    local.active = active ? 1u : 0u;
    if (active) {
      for (int i = 0; i < warmup; ++i) {
        NCCLCHECKGOTO(enqueueP2pExchange(comm, sendBuffer, recvBuffer, bytes,
                                         sendPeer, recvPeer, stream),
                      ret, fail);
      }
      CUDACHECKGOTO(cudaStreamSynchronize(stream), ret, fail);

      std::vector<double> times;
      times.reserve((size_t)iterations);
      for (int i = 0; i < iterations; ++i) {
        CUDACHECKGOTO(cudaEventRecord(start, stream), ret, fail);
        NCCLCHECKGOTO(enqueueP2pExchange(comm, sendBuffer, recvBuffer, bytes,
                                         sendPeer, recvPeer, stream),
                      ret, fail);
        CUDACHECKGOTO(cudaEventRecord(stop, stream), ret, fail);
        CUDACHECKGOTO(cudaEventSynchronize(stop), ret, fail);
        float elapsedMs = 0.0f;
        CUDACHECKGOTO(cudaEventElapsedTime(&elapsedMs, start, stop), ret, fail);
        times.push_back((double)elapsedMs * 1000.0);
      }
      local.timeUs = median(std::move(times));
      local.valid = local.timeUs > 0.0 ? 1u : 0u;
    }

    cocclProfileObservation aggregate = {};
    NCCLCHECKGOTO(aggregateObservation(comm, local, &aggregate), ret, fail);
    if (aggregate.valid) {
      points.push_back({(double)bytes, aggregate.timeUs});
    }
  }

  *model = fitLinearModel(points);

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

static bool runCodecIteration(ncclComm_t comm, ncclCommOp_t op, void* rawBuffer,
                              void* compressedStorage, size_t bytes,
                              cudaStream_t stream, size_t* compressedBytes) {
  void* compressedBuffer = compressedStorage;
  size_t compressedCount = 0;
  ncclDataType_t compressedDatatype = ncclInt8;
  size_t elementCount = bytes / sizeof(float);
  if (ncclCompress(rawBuffer, &compressedBuffer, elementCount, ncclFloat32,
                   &compressedCount, &compressedDatatype, 1, comm->rank,
                   comm, op, stream) != ncclSuccess) {
    return false;
  }
  size_t datatypeBytes = ncclTypeSize(compressedDatatype);
  if (compressedCount == 0 || datatypeBytes == 0 ||
      compressedCount > std::numeric_limits<size_t>::max() / datatypeBytes) {
    return false;
  }
  *compressedBytes = compressedCount * datatypeBytes;
  if (*compressedBytes > bytes) return false;
  // Compression and decompression use the same stream, so decompression may
  // safely overwrite the raw scratch after compression has consumed it.
  return ncclDecompress(rawBuffer, compressedBuffer, elementCount,
                        ncclFloat32, compressedCount, compressedDatatype, 1,
                        comm, op, stream) == ncclSuccess;
}

static ncclResult_t profileCodec(ncclComm_t comm, ncclCommOp_t op,
                                 const std::vector<size_t>& sampleSizes,
                                 cocclCodecModel* model) {
  if (model == nullptr || sampleSizes.size() < 2) return ncclInvalidArgument;
  ncclResult_t ret = ncclSuccess;
  void* rawBuffer = nullptr;
  void* compressedBuffer = nullptr;
  cudaStream_t stream = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  int warmup = (int)std::max<int64_t>(0, ncclParamCocclProfileWarmup());
  int iterations = (int)std::max<int64_t>(1, ncclParamCocclProfileIters());
  std::vector<cocclProfilePoint> points;
  std::vector<double> ratios;

  CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), ret, fail);
  CUDACHECKGOTO(cudaMalloc(&rawBuffer, sampleSizes.back()), ret, fail);
  CUDACHECKGOTO(cudaMalloc(&compressedBuffer, sampleSizes.back()), ret, fail);
  CUDACHECKGOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), ret, fail);
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
      valid = runCodecIteration(comm, op, rawBuffer, compressedBuffer, bytes,
                                stream, &compressedBytes);
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
      valid = runCodecIteration(comm, op, rawBuffer, compressedBuffer, bytes,
                                stream, &compressedBytes);
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
    NCCLCHECKGOTO(aggregateObservation(comm, local, &aggregate), ret, fail);
    if (aggregate.valid && aggregate.compressionRatio > 0.0) {
      points.push_back({(double)bytes, aggregate.timeUs});
      ratios.push_back(aggregate.compressionRatio);
    }
  }

  model->time = fitLinearModel(points);
  model->compressionRatio = median(std::move(ratios));
  model->valid = model->time.valid && model->compressionRatio > 0.0 &&
                 std::isfinite(model->compressionRatio);

exit:
  if (stop != nullptr) (void)cudaEventDestroy(stop);
  if (start != nullptr) (void)cudaEventDestroy(start);
  if (stream != nullptr) (void)cudaStreamDestroy(stream);
  if (compressedBuffer != nullptr) (void)cudaFree(compressedBuffer);
  if (rawBuffer != nullptr) (void)cudaFree(rawBuffer);
  return ret;
fail:
  goto exit;
}

static const char* opName(ncclCommOp_t op) {
  switch (op) {
    case ReduceScatter: return "ReduceScatter";
    case ReduceScatter_Inter: return "ReduceScatter_Inter";
    case AllReduce: return "AllReduce";
    case AllReduce_Inter: return "AllReduce_Inter";
    default: return "Unknown";
  }
}

static void publishP2pModel(bool interNode, const cocclLinearModel& model,
                            int rank) {
  if (!model.valid) {
    if (rank == 0) WARN("COCCL failed to fit %s-node P2P profile",
                        interNode ? "inter" : "intra");
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

static void publishCodecModel(ncclCommOp_t op, const cocclCodecModel& model,
                              int rank) {
  if (!model.valid) {
    if (rank == 0) WARN("COCCL failed to fit compressor profile for %s",
                        opName(op));
    return;
  }
  pthread_mutex_lock(&cocclAutotuneLock);
  cocclPerformanceModel.codecModels.insert_or_assign(op, model);
  pthread_mutex_unlock(&cocclAutotuneLock);
  if (rank == 0) {
    INFO(NCCL_TUNING,
         "COCCL profile %s codec: time_us=%g+%g*bytes ratio=%g",
         opName(op), model.time.alphaUs, model.time.betaUsPerByte,
         model.compressionRatio);
  }
}

static bool uniformNodeRanks(ncclComm_t comm) {
  if (comm == nullptr || comm->nNodes <= 0 || comm->localRanks <= 0) return false;
  for (int node = 0; node < comm->nNodes; ++node) {
    if (comm->nodeRanks[node].localRanks != comm->localRanks) return false;
  }
  return comm->nRanks == comm->nNodes * comm->localRanks;
}

static cocclProcessPerformanceModel snapshotPerformanceModel() {
  pthread_mutex_lock(&cocclAutotuneLock);
  cocclProcessPerformanceModel snapshot = cocclPerformanceModel;
  pthread_mutex_unlock(&cocclAutotuneLock);
  return snapshot;
}

static const cocclCodecModel* findCodecModel(
    const cocclProcessPerformanceModel& model, ncclCommOp_t op) {
  auto it = model.codecModels.find(op);
  return it == model.codecModels.end() ? nullptr : &it->second;
}

static double pccaCost(const cocclLinearModel& p2p,
                       const cocclCodecModel* codec, double messageBytes,
                       int ranks) {
  if (ranks <= 1) return 0.0;
  if (codec == nullptr || !codec->valid || codec->compressionRatio <= 0.0 ||
      !p2p.valid || messageBytes < 0.0) {
    return std::numeric_limits<double>::infinity();
  }
  double wireBytes = messageBytes * (double)(ranks - 1) /
                     (codec->compressionRatio * (double)ranks);
  return predict(p2p, wireBytes) + predict(codec->time, messageBytes);
}

static double globalPccaCost(const cocclProcessPerformanceModel& model,
                             const cocclCodecModel* codec, double messageBytes,
                             int localRanks, int nodes) {
  double cost = 0.0;
  bool hasBranch = false;
  if (localRanks > 1) {
    cost = pccaCost(model.intraP2p, codec, messageBytes / (double)nodes,
                    localRanks);
    hasBranch = true;
  }
  if (nodes > 1) {
    double interBytes = messageBytes * (double)(nodes - 1) / (double)nodes +
                        messageBytes / ((double)localRanks * (double)nodes);
    int interRanks = localRanks * (nodes - 1) + 1;
    double interCost = pccaCost(model.interP2p, codec, interBytes, interRanks);
    cost = hasBranch ? std::max(cost, interCost) : interCost;
    hasBranch = true;
  }
  return hasBranch ? cost : 0.0;
}

static double reduceScatterTwoShotCost(
    const cocclProcessPerformanceModel& model,
    const cocclCodecModel* codec, double messageBytes, int localRanks,
    int nodes) {
  return pccaCost(model.intraP2p, codec, messageBytes, localRanks) +
         pccaCost(model.interP2p, codec,
                  messageBytes / (double)localRanks, nodes);
}

static double allReduceOneShotCost(
    const cocclProcessPerformanceModel& model,
    const cocclCodecModel* codec, double messageBytes, int localRanks,
    int nodes) {
  double cost = 0.0;
  bool hasBranch = false;
  if (localRanks > 1) {
    cost = pccaCost(model.intraP2p, codec,
                    messageBytes * (double)localRanks, localRanks);
    hasBranch = true;
  }
  if (nodes > 1) {
    double interBytes = messageBytes * (double)localRanks *
                        (double)(nodes - 1);
    int interRanks = localRanks * (nodes - 1) + 1;
    double interCost = pccaCost(model.interP2p, codec, interBytes, interRanks);
    cost = hasBranch ? std::max(cost, interCost) : interCost;
    hasBranch = true;
  }
  return hasBranch ? cost : 0.0;
}

static const char* algorithmName(cocclAlgorithmKind algorithm) {
  switch (algorithm) {
    case cocclAlgorithmReduceScatterOneShot: return "reducescatter-oneshot";
    case cocclAlgorithmReduceScatterTwoShot: return "reducescatter-twoshot";
    case cocclAlgorithmAllReduceOneShot: return "allreduce-oneshot";
    case cocclAlgorithmAllReduceTwoShot: return "allreduce-twoshot";
    case cocclAlgorithmAllReduceTripleShot: return "allreduce-tripleshot";
    default: return "none";
  }
}

static ncclResult_t selectReduceScatter(const cocclRuntimeArgs* args,
                                        cocclAlgorithmDecision* decision) {
  ncclComm_t comm = args->comm;
  bool hierarchical = comm->nNodes > 1 && comm->localRanks > 1 &&
                      uniformNodeRanks(comm);
  const char* policy = getenv("NCCL_COCCL_RS_ALGORITHM");
  if (policy != nullptr && strcasecmp(policy, "oneshot") == 0) {
    decision->algorithm = cocclAlgorithmReduceScatterOneShot;
    return ncclSuccess;
  }
  if (policy != nullptr && strcasecmp(policy, "twoshot") == 0) {
    if (hierarchical) {
      decision->algorithm = cocclAlgorithmReduceScatterTwoShot;
      return ncclSuccess;
    }
    WARN("COCCL forced ReduceScatter twoshot is unavailable for this topology; using oneshot");
    decision->algorithm = cocclAlgorithmReduceScatterOneShot;
    return ncclSuccess;
  }
  if (ncclParamCocclAutotune() == 0) {
    decision->algorithm = hierarchical ? cocclAlgorithmReduceScatterTwoShot
                                       : cocclAlgorithmReduceScatterOneShot;
    return ncclSuccess;
  }

  cocclProcessPerformanceModel model = snapshotPerformanceModel();
  const cocclCodecModel* oneCodec = findCodecModel(model, ReduceScatter);
  double messageBytes = (double)args->count * (double)comm->nRanks *
                        (double)ncclTypeSize(args->datatype);
  decision->oneShotUs = globalPccaCost(model, oneCodec, messageBytes,
                                       comm->localRanks, comm->nNodes);
  if (!hierarchical) {
    decision->algorithm = cocclAlgorithmReduceScatterOneShot;
    decision->usedModel = std::isfinite(decision->oneShotUs);
    return ncclSuccess;
  }

  const cocclCodecModel* twoCodec = findCodecModel(model, ReduceScatter_Inter);
  decision->twoShotUs = reduceScatterTwoShotCost(
      model, twoCodec, messageBytes, comm->localRanks, comm->nNodes);
  if (std::isfinite(decision->oneShotUs) &&
      std::isfinite(decision->twoShotUs)) {
    decision->algorithm = decision->oneShotUs <= decision->twoShotUs
                              ? cocclAlgorithmReduceScatterOneShot
                              : cocclAlgorithmReduceScatterTwoShot;
    decision->usedModel = true;
  } else {
    decision->algorithm = cocclAlgorithmReduceScatterTwoShot;
  }
  return ncclSuccess;
}

static ncclResult_t selectAllReduce(const cocclRuntimeArgs* args,
                                    cocclAlgorithmDecision* decision) {
  ncclComm_t comm = args->comm;
  bool divisible = comm->nRanks > 0 &&
                   args->count % (size_t)comm->nRanks == 0;
  bool hierarchical = divisible && comm->nNodes > 1 &&
                      comm->localRanks > 1 && uniformNodeRanks(comm);
  const char* policy = getenv("NCCL_COCCL_AR_ALGORITHM");
  if (policy != nullptr && strcasecmp(policy, "oneshot") == 0) {
    decision->algorithm = cocclAlgorithmAllReduceOneShot;
    return ncclSuccess;
  }
  if (policy != nullptr && strcasecmp(policy, "twoshot") == 0) {
    if (divisible) {
      decision->algorithm = cocclAlgorithmAllReduceTwoShot;
      return ncclSuccess;
    }
    WARN("COCCL forced AllReduce twoshot requires count divisible by nRanks; using oneshot");
    decision->algorithm = cocclAlgorithmAllReduceOneShot;
    return ncclSuccess;
  }
  if (policy != nullptr && strcasecmp(policy, "tripleshot") == 0) {
    if (hierarchical) {
      decision->algorithm = cocclAlgorithmAllReduceTripleShot;
      return ncclSuccess;
    }
    WARN("COCCL forced AllReduce tripleshot is unavailable; using %s",
         divisible ? "twoshot" : "oneshot");
    decision->algorithm = divisible ? cocclAlgorithmAllReduceTwoShot
                                    : cocclAlgorithmAllReduceOneShot;
    return ncclSuccess;
  }
  if (ncclParamCocclAutotune() == 0) {
    decision->algorithm = divisible ? cocclAlgorithmAllReduceTwoShot
                                    : cocclAlgorithmAllReduceOneShot;
    return ncclSuccess;
  }

  cocclProcessPerformanceModel model = snapshotPerformanceModel();
  const cocclCodecModel* baseCodec = findCodecModel(model, AllReduce);
  double messageBytes = (double)args->count *
                        (double)ncclTypeSize(args->datatype);
  decision->oneShotUs = allReduceOneShotCost(
      model, baseCodec, messageBytes, comm->localRanks, comm->nNodes);
  if (!divisible) {
    decision->algorithm = cocclAlgorithmAllReduceOneShot;
    decision->usedModel = std::isfinite(decision->oneShotUs);
    return ncclSuccess;
  }

  const cocclCodecModel* interCodec = findCodecModel(model, AllReduce_Inter);
  double oneShotRs = globalPccaCost(model, baseCodec, messageBytes,
                                    comm->localRanks, comm->nNodes);
  double allGather = globalPccaCost(model, baseCodec, messageBytes,
                                    comm->localRanks, comm->nNodes);
  decision->twoShotUs = oneShotRs + allGather;
  decision->tripleShotUs = std::numeric_limits<double>::infinity();
  if (hierarchical) {
    double twoShotRs = reduceScatterTwoShotCost(
        model, interCodec, messageBytes, comm->localRanks, comm->nNodes);
    double tripleAllGather = globalPccaCost(
        model, interCodec, messageBytes, comm->localRanks, comm->nNodes);
    decision->tripleShotUs = twoShotRs + tripleAllGather;
  }

  if (std::isfinite(decision->oneShotUs) &&
      std::isfinite(decision->twoShotUs) &&
      (!hierarchical || std::isfinite(decision->tripleShotUs))) {
    decision->algorithm = cocclAlgorithmAllReduceOneShot;
    double best = decision->oneShotUs;
    if (decision->twoShotUs < best) {
      best = decision->twoShotUs;
      decision->algorithm = cocclAlgorithmAllReduceTwoShot;
    }
    if (hierarchical && decision->tripleShotUs < best) {
      decision->algorithm = cocclAlgorithmAllReduceTripleShot;
    }
    decision->usedModel = true;
  } else {
    decision->algorithm = cocclAlgorithmAllReduceTwoShot;
  }
  return ncclSuccess;
}

}  // namespace

ncclResult_t cocclAutotuneProfile(
    ncclComm_t comm, const cocclAutotuneProfileOptions* options) {
  if (comm == nullptr || options == nullptr) return ncclInvalidArgument;
  if (comm->nRanks <= 1 || ncclParamCocclAutotune() == 0 ||
      (!options->profileReduceScatter && !options->profileAllReduce)) {
    return ncclSuccess;
  }

  uint32_t localNeeds = localProfileNeeds(comm, options);
  uint32_t needs = 0;
  NCCLCHECK(collectiveProfileNeeds(comm, localNeeds, &needs));
  if (needs == 0) return ncclSuccess;

  // Every rank has entered the collective needs exchange before the categories
  // are claimed, so all local init workers execute the same profile sequence.
  markProfilesAttempted(needs);

  std::vector<size_t> sampleSizes;
  ncclResult_t sizeResult = buildSampleSizes(comm, &sampleSizes);
  if (sizeResult != ncclSuccess || sampleSizes.size() < 2) {
    if (comm->rank == 0) {
      WARN("COCCL autotune has fewer than two safe profile sizes; using heuristic selection");
    }
    return ncclSuccess;
  }
  if (comm->rank == 0) {
    INFO(NCCL_TUNING,
         "COCCL profiling %zu sizes from %zu to %zu bytes on comm %p",
         sampleSizes.size(), sampleSizes.front(), sampleSizes.back(), comm);
  }

  if (needs & cocclProfileNeedIntra) {
    cocclLinearModel model = {};
    ncclResult_t result = profileP2p(comm, false, sampleSizes, &model);
    if (result == ncclSuccess) publishP2pModel(false, model, comm->rank);
    else if (comm->rank == 0) WARN("COCCL intra-node P2P profiling failed: %d", result);
  }
  if (needs & cocclProfileNeedInter) {
    cocclLinearModel model = {};
    ncclResult_t result = profileP2p(comm, true, sampleSizes, &model);
    if (result == ncclSuccess) publishP2pModel(true, model, comm->rank);
    else if (comm->rank == 0) WARN("COCCL inter-node P2P profiling failed: %d", result);
  }

  const struct {
    uint32_t bit;
    ncclCommOp_t op;
  } codecProfiles[] = {
      {cocclProfileNeedReduceScatter, ReduceScatter},
      {cocclProfileNeedReduceScatterInter, ReduceScatter_Inter},
      {cocclProfileNeedAllReduce, AllReduce},
      {cocclProfileNeedAllReduceInter, AllReduce_Inter},
  };
  for (const auto& profile : codecProfiles) {
    if ((needs & profile.bit) == 0) continue;
    cocclCodecModel model = {};
    ncclResult_t result = profileCodec(comm, profile.op, sampleSizes, &model);
    if (result == ncclSuccess) publishCodecModel(profile.op, model, comm->rank);
    else if (comm->rank == 0) {
      WARN("COCCL compressor profiling failed for %s: %d", opName(profile.op), result);
    }
  }
  return ncclSuccess;
}

ncclResult_t cocclSelectAlgorithm(const cocclRuntimeArgs* args,
                                  cocclAlgorithmDecision* decision) {
  if (args == nullptr || args->comm == nullptr || decision == nullptr) {
    return ncclInvalidArgument;
  }
  *decision = cocclAlgorithmDecision{};
  decision->oneShotUs = std::numeric_limits<double>::infinity();
  decision->twoShotUs = std::numeric_limits<double>::infinity();
  decision->tripleShotUs = std::numeric_limits<double>::infinity();
  ncclResult_t result = ncclInvalidArgument;
  if (args->func == ncclFuncReduceScatter) {
    result = selectReduceScatter(args, decision);
  } else if (args->func == ncclFuncAllReduce) {
    result = selectAllReduce(args, decision);
  }
  if (result == ncclSuccess && args->comm->rank == 0) {
    double messageBytes = (double)args->count *
                          (double)ncclTypeSize(args->datatype);
    if (args->func == ncclFuncReduceScatter) {
      messageBytes *= (double)args->comm->nRanks;
    }
    INFO(NCCL_TUNING,
         "COCCL select bytes=%g ranks=%d local=%d nodes=%d one=%g two=%g triple=%g model=%d -> %s",
         messageBytes, args->comm->nRanks,
         args->comm->localRanks, args->comm->nNodes, decision->oneShotUs,
         decision->twoShotUs, decision->tripleShotUs,
         (int)decision->usedModel, algorithmName(decision->algorithm));
  }
  return result;
}
