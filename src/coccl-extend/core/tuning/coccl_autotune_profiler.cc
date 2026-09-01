#include "coccl_autotune_internal.h"

#include "bootstrap.h"
#include "checks.h"
#include "core/config/coccl_config.h"
#include "comm.h"
#include "core/compression/compress.h"
#include "core/compression/reduce_extend.h"
#include "debug.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <pthread.h>
#include <vector>

namespace {

enum cocclProfileNeed : uint32_t {
  cocclProfileNeedCompressors = 1u << 0,
};

struct cocclProfileObservation {
  double timeUs = 0.0;
  double compressionRatio = 0.0;
  double encodeTimeUs = 0.0;
  double decodeTimeUs = 0.0;
  uint32_t active = 0;
  uint32_t valid = 0;
};

struct cocclProfiledCompressor {
  void* compressor = nullptr;
  cocclPolicyKey policy;
};

struct cocclProcessPerformanceModel {
  std::vector<cocclProfiledCompressor> enabledCompressors;
  std::map<void*, std::map<ncclDataType_t, cocclCodecModel>>
      compressorModels;
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

uint32_t localProfileNeeds() {
  uint32_t needs = 0;
  pthread_mutex_lock(&cocclAutotuneLock);
  const bool hasCompressors =
      !cocclPerformanceModel.enabledCompressors.empty();
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
    aggregate->encodeTimeUs =
        std::max(aggregate->encodeTimeUs, observation.encodeTimeUs);
    aggregate->decodeTimeUs =
        std::max(aggregate->decodeTimeUs, observation.decodeTimeUs);
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

bool runCompressorIteration(
    ncclComm_t comm, void* compressor, void* rawBuffer,
    void* compressedBuffer, size_t bytes, size_t compressedCapacity,
    ncclDataType_t datatype,
    cudaStream_t stream, cudaEvent_t encoded,
    cocclCompressorFrameMetadata* frameMetadata,
    size_t* compressedBytes) {
  const size_t elements = bytes / ncclTypeSize(datatype);
  const cocclCompressorView input = {
      rawBuffer, bytes, bytes, elements, 1, datatype, nullptr, 0};
  cocclCompressorView compressed = {
      compressedBuffer, compressedCapacity, 0, 0, 1, ncclInt8,
      frameMetadata, frameMetadata == nullptr ? 0 : bytes};
  if (ncclCompress(
          compressor, input, &compressed, comm->rank, stream) != ncclSuccess) {
    return false;
  }
  if (encoded != nullptr && cudaEventRecord(encoded, stream) != cudaSuccess) {
    return false;
  }
  *compressedBytes = compressed.bytes;
  if (*compressedBytes == 0 || *compressedBytes > bytes) return false;

  const cocclCompressorView compressedInput = {
      compressed.data, compressed.bytes, compressed.bytes,
      compressed.elements, compressed.chunks, compressed.datatype,
      compressed.frameMetadata, compressed.frameStrideBytes};
  cocclCompressorView decompressed = {
      rawBuffer, bytes, 0, elements, 1, datatype, nullptr, 0};
  return ncclDecompress(
             compressor, compressedInput, &decompressed, stream) ==
      ncclSuccess;
}

bool runDrcIteration(
    void* compressor, const cocclCompressorView& input, void* outputBuffer,
    size_t outputCapacity, size_t reduceChunks,
    ncclDataType_t originalDatatype, size_t originalElements,
    cudaStream_t stream) {
  cocclCompressorView output = {
      outputBuffer, outputCapacity, 0, 0, input.chunks / reduceChunks,
      ncclInt8, nullptr, 0};
  return cocclExecuteCompressor(
             compressor, compressor,
             cocclCompressorOperationDecompressReduceCompress,
             input, &output, -1, reduceChunks, originalDatatype,
             originalElements, stream) == ncclSuccess;
}

bool runDrIteration(
    ncclComm_t comm, void* compressor, const cocclCompressorView& input,
    void* decompressedBuffer, void* outputBuffer, size_t outputCapacity,
    size_t reduceChunks, ncclDataType_t datatype, size_t outputElements,
    cudaStream_t stream) {
  cocclCompressorView output = {
      outputBuffer, outputCapacity, 0, outputElements,
      input.chunks / reduceChunks, datatype, nullptr, 0};
  if (cocclCompressorSupports(
          compressor, cocclCompressorCapabilityDecompressReduce)) {
    return ncclDecompressReduce(
               compressor, comm, input, &output, reduceChunks, stream) ==
        ncclSuccess;
  }

  cocclCompressorView decompressed = {
      decompressedBuffer, outputCapacity * reduceChunks,
      outputCapacity * reduceChunks, outputElements * reduceChunks,
      input.chunks, datatype, nullptr, 0};
  return ncclDecompress(compressor, input, &decompressed, stream) ==
             ncclSuccess &&
      ncclReduceChunk(
          decompressedBuffer, outputElements, outputBuffer, datatype,
          (int)reduceChunks, stream) == ncclSuccess;
}

ncclResult_t profileCompressor(
    ncclComm_t comm, void* compressor,
    const std::vector<size_t>& sampleSizes, ncclDataType_t datatype,
    bool profileReductions, cocclCodecModel* model) {
  ncclResult_t ret = ncclSuccess;
  void* rawBuffer = nullptr;
  void* compressedBuffer = nullptr;
  void* reductionBuffer = nullptr;
  cocclCompressorFrameMetadata* deviceFrameMetadata = nullptr;
  cocclCompressorFrameMetadata* hostFrameMetadata = nullptr;
  cudaStream_t stream = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t encoded = nullptr;
  cudaEvent_t stop = nullptr;
  std::vector<cocclAutotuneProfilePoint> points;
  std::vector<cocclAutotuneProfilePoint> encodePoints;
  std::vector<cocclAutotuneProfilePoint> decodePoints;
  std::vector<cocclAutotuneProfilePoint> drcPoints;
  std::vector<cocclAutotuneProfilePoint> drPoints;
  std::vector<double> ratios;
  bool framed = false;
  const cocclAutotuneConfig& config = cocclGetConfig().autotune;
  // Codec work is local. Rank 0 measures it and aggregateObservation makes
  // the same samples available to every rank before fitting the model.
  const bool active = comm->rank == 0;

  framed = cocclCompressorSupports(
      compressor, cocclCompressorCapabilityFramed);
  const bool fusedDrc =
      profileReductions && !framed && cocclCompressorSupports(
          compressor, cocclCompressorCapabilityDecompressReduceCompress);

  if (active) {
    CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), ret, fail);
    CUDACHECKGOTO(cudaMalloc(&rawBuffer, sampleSizes.back()), ret, fail);
    CUDACHECKGOTO(
        cudaMalloc(&compressedBuffer, sampleSizes.back()), ret, fail);
    if (profileReductions && !framed) {
      CUDACHECKGOTO(
          cudaMalloc(&reductionBuffer,
                     sampleSizes.back() / (size_t)comm->nRanks),
          ret, fail);
    }
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
    CUDACHECKGOTO(cudaEventCreate(&encoded), ret, fail);
    CUDACHECKGOTO(cudaEventCreate(&stop), ret, fail);
  }

  for (size_t bytes : sampleSizes) {
    cocclProfileObservation local = {};
    local.active = active ? 1u : 0u;
    bool valid = active;
    size_t compressedBytes = 0;
    for (int i = 0; i < config.warmup && valid; ++i) {
      valid = cudaMemsetAsync(rawBuffer, 0x3f, bytes, stream) == cudaSuccess &&
          runCompressorIteration(
              comm, compressor, rawBuffer, compressedBuffer, bytes,
              sampleSizes.back(), datatype, stream, nullptr,
              deviceFrameMetadata,
              &compressedBytes);
    }
    if (valid) valid = cudaStreamSynchronize(stream) == cudaSuccess;

    std::vector<double> times;
    std::vector<double> encodeTimes;
    std::vector<double> decodeTimes;
    for (int i = 0; i < config.iterations && valid; ++i) {
      valid = cudaMemsetAsync(rawBuffer, 0x3f, bytes, stream) == cudaSuccess &&
          cudaEventRecord(start, stream) == cudaSuccess &&
          runCompressorIteration(
              comm, compressor, rawBuffer, compressedBuffer, bytes,
              sampleSizes.back(), datatype, stream, encoded,
              deviceFrameMetadata,
              &compressedBytes) &&
          cudaEventRecord(stop, stream) == cudaSuccess &&
          cudaEventSynchronize(stop) == cudaSuccess;
      if (!valid) break;
      float elapsedMs = 0.0f;
      valid = cudaEventElapsedTime(&elapsedMs, start, stop) == cudaSuccess;
      float encodeMs = 0.0f;
      float decodeMs = 0.0f;
      valid = valid &&
          cudaEventElapsedTime(&encodeMs, start, encoded) == cudaSuccess &&
          cudaEventElapsedTime(&decodeMs, encoded, stop) == cudaSuccess;
      if (valid) {
        times.push_back((double)elapsedMs * 1000.0);
        encodeTimes.push_back((double)encodeMs * 1000.0);
        decodeTimes.push_back((double)decodeMs * 1000.0);
      }
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
      local.encodeTimeUs = median(std::move(encodeTimes));
      local.decodeTimeUs = median(std::move(decodeTimes));
      local.compressionRatio = (double)bytes / (double)compressedBytes;
      local.valid = 1;
    }

    cocclProfileObservation aggregate = {};
    NCCLCHECKGOTO(
        aggregateObservation(comm, local, &aggregate), ret, fail);
    if (aggregate.valid && aggregate.compressionRatio > 0.0) {
      points.push_back({(double)bytes, aggregate.timeUs});
      encodePoints.push_back({(double)bytes, aggregate.encodeTimeUs});
      decodePoints.push_back({(double)bytes, aggregate.decodeTimeUs});
      ratios.push_back(aggregate.compressionRatio);
    }

    if (profileReductions && !framed) {
      cocclProfileObservation drLocal = {};
      drLocal.active = active ? 1u : 0u;
      if (active) {
        const size_t elements = bytes / (size_t)ncclTypeSize(datatype);
        const size_t outputElements = elements / (size_t)comm->nRanks;
        const size_t outputCapacity = bytes / (size_t)comm->nRanks;
        const cocclCompressorView input = {
            rawBuffer, bytes, bytes, elements, (size_t)comm->nRanks,
            datatype, nullptr, 0};
        cocclCompressorView compressed = {
            compressedBuffer, bytes, 0, 0, (size_t)comm->nRanks,
            ncclInt8, nullptr, 0};
        bool drValid = elements % (size_t)comm->nRanks == 0 &&
            cudaMemsetAsync(rawBuffer, 0x3f, bytes, stream) == cudaSuccess &&
            ncclCompress(
                compressor, input, &compressed, comm->rank, stream) ==
                ncclSuccess &&
            compressed.datatype != COCCL_COMPRESSOR_RAW_PASSTHROUGH &&
            cudaStreamSynchronize(stream) == cudaSuccess;
        const cocclCompressorView compressedInput = {
            compressed.data, compressed.bytes, compressed.bytes,
            compressed.elements, compressed.chunks, compressed.datatype,
            nullptr, 0};
        for (int i = 0; i < config.warmup && drValid; ++i) {
          drValid = runDrIteration(
              comm, compressor, compressedInput, rawBuffer, reductionBuffer,
              outputCapacity, (size_t)comm->nRanks, datatype, outputElements,
              stream);
        }
        if (drValid) drValid = cudaStreamSynchronize(stream) == cudaSuccess;

        std::vector<double> drTimes;
        for (int i = 0; i < config.iterations && drValid; ++i) {
          drValid = cudaEventRecord(start, stream) == cudaSuccess &&
              runDrIteration(
                  comm, compressor, compressedInput, rawBuffer,
                  reductionBuffer, outputCapacity, (size_t)comm->nRanks,
                  datatype, outputElements, stream) &&
              cudaEventRecord(stop, stream) == cudaSuccess &&
              cudaEventSynchronize(stop) == cudaSuccess;
          if (!drValid) break;
          float elapsedMs = 0.0f;
          drValid =
              cudaEventElapsedTime(&elapsedMs, start, stop) == cudaSuccess;
          if (drValid) drTimes.push_back((double)elapsedMs * 1000.0);
        }
        if (drValid && !drTimes.empty()) {
          drLocal.timeUs = median(std::move(drTimes));
          drLocal.valid = 1;
        }
      }
      cocclProfileObservation drAggregate = {};
      NCCLCHECKGOTO(
          aggregateObservation(comm, drLocal, &drAggregate), ret, fail);
      if (drAggregate.valid) {
        drPoints.push_back({(double)bytes, drAggregate.timeUs});
      }
    }

    if (fusedDrc) {
      cocclProfileObservation drcLocal = {};
      drcLocal.active = active ? 1u : 0u;
      if (active) {
        const size_t elements = bytes / (size_t)ncclTypeSize(datatype);
        const size_t outputElements = elements / (size_t)comm->nRanks;
        const size_t outputCapacity = bytes / (size_t)comm->nRanks;
        const cocclCompressorView input = {
            rawBuffer, bytes, bytes, elements, (size_t)comm->nRanks,
            datatype, nullptr, 0};
        cocclCompressorView compressed = {
            compressedBuffer, bytes, 0, 0, (size_t)comm->nRanks,
            ncclInt8, nullptr, 0};
        bool drcValid = elements % (size_t)comm->nRanks == 0 &&
            cudaMemsetAsync(rawBuffer, 0x3f, bytes, stream) == cudaSuccess &&
            ncclCompress(
                compressor, input, &compressed, comm->rank, stream) ==
                ncclSuccess &&
            compressed.datatype != COCCL_COMPRESSOR_RAW_PASSTHROUGH &&
            cudaStreamSynchronize(stream) == cudaSuccess;
        const cocclCompressorView compressedInput = {
            compressed.data, compressed.bytes, compressed.bytes,
            compressed.elements, compressed.chunks, compressed.datatype,
            nullptr, 0};
        for (int i = 0; i < config.warmup && drcValid; ++i) {
          drcValid = runDrcIteration(
              compressor, compressedInput, reductionBuffer, outputCapacity,
              (size_t)comm->nRanks, datatype, outputElements, stream);
        }
        if (drcValid) {
          drcValid = cudaStreamSynchronize(stream) == cudaSuccess;
        }

        std::vector<double> drcTimes;
        for (int i = 0; i < config.iterations && drcValid; ++i) {
          drcValid = cudaEventRecord(start, stream) == cudaSuccess &&
              runDrcIteration(
                  compressor, compressedInput, reductionBuffer,
                  outputCapacity, (size_t)comm->nRanks, datatype,
                  outputElements, stream) &&
              cudaEventRecord(stop, stream) == cudaSuccess &&
              cudaEventSynchronize(stop) == cudaSuccess;
          if (!drcValid) break;
          float elapsedMs = 0.0f;
          drcValid =
              cudaEventElapsedTime(&elapsedMs, start, stop) == cudaSuccess;
          if (drcValid) drcTimes.push_back((double)elapsedMs * 1000.0);
        }
        if (drcValid && !drcTimes.empty()) {
          drcLocal.timeUs = median(std::move(drcTimes));
          drcLocal.valid = 1;
        }
      }
      cocclProfileObservation drcAggregate = {};
      NCCLCHECKGOTO(
          aggregateObservation(comm, drcLocal, &drcAggregate), ret, fail);
      if (drcAggregate.valid) {
        drcPoints.push_back({(double)bytes, drcAggregate.timeUs});
      }
    }
  }

  model->time = cocclAutotuneFitLinearModel(points);
  model->encodeTime = cocclAutotuneFitLinearModel(encodePoints);
  model->decodeTime = cocclAutotuneFitLinearModel(decodePoints);
  model->drcTime = cocclAutotuneFitLinearModel(drcPoints);
  model->drTime = cocclAutotuneFitLinearModel(drPoints);
  model->compressionRatio = median(std::move(ratios));
  model->valid = model->time.valid && model->compressionRatio > 0.0 &&
                 std::isfinite(model->compressionRatio);

exit:
  if (stop != nullptr) (void)cudaEventDestroy(stop);
  if (encoded != nullptr) (void)cudaEventDestroy(encoded);
  if (start != nullptr) (void)cudaEventDestroy(start);
  if (stream != nullptr) (void)cudaStreamDestroy(stream);
  if (hostFrameMetadata != nullptr) (void)cudaFreeHost(hostFrameMetadata);
  if (deviceFrameMetadata != nullptr) (void)cudaFree(deviceFrameMetadata);
  if (reductionBuffer != nullptr) (void)cudaFree(reductionBuffer);
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

void publishCompressorModel(const cocclProfiledCompressor& profiled,
                            ncclDataType_t datatype,
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
  cocclPerformanceModel.compressorModels[profiled.compressor][datatype] =
      model;
  pthread_mutex_unlock(&cocclAutotuneLock);
  if (rank == 0) {
    INFO(COCCL_TUNING,
         "COCCL compressor %s policy=%s-%s datatype=%d model: time_us=%g+%g*bytes encode_us=%g+%g*bytes decode_us=%g+%g*bytes dr_us=%g+%g*bytes ratio=%g",
         compressorName(compressor), operationName(profiled.policy.operation),
         scopeName(profiled.policy.scope), (int)datatype, model.time.alphaUs,
         model.time.betaUsPerByte, model.encodeTime.alphaUs,
         model.encodeTime.betaUsPerByte, model.decodeTime.alphaUs,
         model.decodeTime.betaUsPerByte, model.drTime.alphaUs,
         model.drTime.betaUsPerByte, model.compressionRatio);
  }
}

void copyCompressorModelLocked(void* compressor, ncclDataType_t datatype,
                               cocclCodecModel* model) {
  *model = {};
  const auto found = cocclPerformanceModel.compressorModels.find(compressor);
  if (found == cocclPerformanceModel.compressorModels.end()) return;
  const auto typed = found->second.find(datatype);
  if (typed != found->second.end()) *model = typed->second;
}

}  // namespace

void cocclAutotuneSnapshotCodecModels(
    void* defaultCompressor, void* intraCompressor, void* interCompressor,
    ncclDataType_t datatype,
    cocclCodecModel* defaultModel, cocclCodecModel* intraModel,
    cocclCodecModel* interModel) {
  pthread_mutex_lock(&cocclAutotuneLock);
  copyCompressorModelLocked(defaultCompressor, datatype, defaultModel);
  copyCompressorModelLocked(intraCompressor, datatype, intraModel);
  copyCompressorModelLocked(interCompressor, datatype, interModel);
  pthread_mutex_unlock(&cocclAutotuneLock);
}

cocclCodecModel cocclAutotuneSnapshotCodecModel(
    void* compressor, ncclDataType_t datatype) {
  cocclCodecModel model;
  pthread_mutex_lock(&cocclAutotuneLock);
  copyCompressorModelLocked(compressor, datatype, &model);
  pthread_mutex_unlock(&cocclAutotuneLock);
  return model;
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

  const uint32_t localNeeds = localProfileNeeds();
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
    INFO(COCCL_TUNING,
         "COCCL fitting global models at %zu sizes from %zu to %zu bytes",
         sampleSizes.size(), sampleSizes.front(), sampleSizes.back());
  }

  if ((needs & cocclProfileNeedCompressors) != 0) {
    for (ncclDataType_t datatype : {ncclFloat32, ncclBfloat16}) {
      for (const cocclProfiledCompressor& compressor :
           snapshotEnabledCompressors()) {
        cocclCodecModel model;
        const bool profileReductions =
            measurementComm->nNodes == 1 &&
            compressor.policy.operation == cocclOperation::AllReduce &&
            compressor.policy.scope != cocclCompressionScope::Inter;
        const ncclResult_t result = profileCompressor(
            measurementComm, compressor.compressor, sampleSizes, datatype,
            profileReductions, &model);
        if (result == ncclSuccess) {
          publishCompressorModel(
              compressor, datatype, model, measurementComm->rank);
        }
      }
    }
  }
  return ncclSuccess;
}
