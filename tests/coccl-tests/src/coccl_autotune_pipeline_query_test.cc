#include "core/tuning/coccl_autotune_pipeline.h"

#include "comm.h"
#include "core/compression/coccl_compressor_runtime.h"
#include "core/config/coccl_config.h"
#include "core/pipeline/coccl_pipeline.h"
#include "core/tuning/coccl_autotune_internal.h"
#include "debug.h"

#include <chrono>
#include <cstdio>

namespace {

cocclConfig config;

cocclLinearModel model() {
  cocclLinearModel result;
  result.alphaUs = 2.0;
  result.betaUsPerByte = 1.0e-5;
  result.valid = true;
  return result;
}

}  // namespace

thread_local int ncclDebugNoWarn = 0;
void ncclDebugLog(ncclDebugLogLevel, unsigned long, const char*, int,
                  const char*, ...) {}

const cocclConfig& cocclGetConfig() {
  return config;
}

bool cocclCompressorSupports(void*, cocclCompressorCapability) {
  return false;
}

cocclCodecModel cocclAutotuneSnapshotCodecModel(
    void*, ncclDataType_t) {
  cocclCodecModel result;
  result.time = model();
  result.encodeTime = model();
  result.decodeTime = model();
  result.compressionRatio = 4.0;
  result.valid = true;
  return result;
}

cocclLinearModel cocclAutotuneSnapshotTopologyStageModel(
    ncclComm_t, cocclAutotuneTopologyOperation) {
  return model();
}

int cocclPipelineStageCtaPolicy(
    ncclComm_t, const cocclPipelineStage&) {
  return NCCL_CTA_POLICY_DEFAULT;
}

int main() {
  ncclComm comm = {};
  comm.rank = 0;
  comm.nRanks = 8;
  comm.nNodes = 2;
  comm.node = 0;
  int rankToNode[8] = {0, 0, 0, 0, 1, 1, 1, 1};
  comm.rankToNode = rankToNode;

  void* compressor = reinterpret_cast<void*>(0x1000);
  cocclPipelineStage stages[] = {
      cocclPipelineCompress(compressor),
      cocclPipelineSendRecv(&comm, 4, cocclPipelineSend, compressor)};
  const cocclPipelineSpec spec = {
      "send", reinterpret_cast<void*>(0x2000),
      reinterpret_cast<void*>(0x3000), size_t{1} << 30, 1,
      ncclInt8, &comm, nullptr, stages, 2,
      cocclPipelineInPlaceNone, cocclPipelineInputContiguous, 0};

  (void)cocclAutotunePipelineLayout(&spec);
  constexpr int kIterations = 1000000;
  size_t checksum = 0;
  const auto begin = std::chrono::steady_clock::now();
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    checksum += cocclAutotunePipelineLayout(&spec).targetSliceBytes;
  }
  const auto end = std::chrono::steady_clock::now();
  const double nsPerCall =
      std::chrono::duration<double, std::nano>(end - begin).count() /
      kIterations;
  std::printf("sendrecv_pipeline_query_ns_per_call=%.2f checksum=%zu\n",
              nsPerCall, checksum);
  cocclAutotunePipelineCommDestroy(&comm);
  return nsPerCall < 1000.0 ? 0 : 1;
}
