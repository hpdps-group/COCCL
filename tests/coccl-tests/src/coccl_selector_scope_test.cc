#include "core/tuning/coccl_autotune_internal.h"

#include "core/compression/coccl_compressor_runtime.h"
#include "core/config/coccl_config.h"
#include "core/pipeline/coccl_pipeline.h"
#include "core/runtime/coccl_comm.h"
#include "core/runtime/coccl_primitive_dispatch.h"
#include "core/tuning/coccl_autotune_pipeline.h"
#include "comm.h"
#include "debug.h"

#include <chrono>
#include <cstdarg>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace {

void fail(const char* expression, int line) {
  std::fprintf(stderr, "line %d: %s\n", line, expression);
  std::exit(1);
}

#define EXPECT(expression) \
  do { if (!(expression)) fail(#expression, __LINE__); } while (0)

void* const kDefault = reinterpret_cast<void*>(0x10);
void* const kIntra = reinterpret_cast<void*>(0x11);
void* const kInter = reinterpret_cast<void*>(0x12);
cocclConfig config;
cocclSelectionPerformanceModel performance = {
    {1.0, 1e-6, true}, {4.0, 4e-6, true},
    {1.0, 1e-6, true}, {1.0, 1e-6, true}};
cocclCodecModel defaultModel = {{2.0, 2e-6, true}, 4.0, true};
cocclCodecModel intraModel = {{1.0, 1e-6, true}, 8.0, true};
cocclCodecModel interModel = {{3.0, 3e-6, true}, 4.0, true};
ncclDataType_t snapshotDatatype = ncclNumTypes;
bool framedCompressor;
double lastScores[3];
bool lastUsedModel;
bool captureSelectionLog = true;
double nativeAllGatherCost = 0.5;
double hierarchicalAllGatherCost = 1.0;

void setScope(cocclPreparedCall* prepared, cocclCompressionScope scope,
              void* compressor, bool datatypeSupported = true) {
  const size_t index = static_cast<size_t>(scope);
  prepared->compressors.handles[index] = compressor;
  prepared->compressors.datatypeSupported[index] = datatypeSupported;
}

cocclPreparedCall makePrepared(ncclComm_t comm, cocclOperation operation) {
  cocclPreparedCall prepared = {};
  prepared.info.sendbuff = reinterpret_cast<void*>(0x1000);
  prepared.info.recvbuff = reinterpret_cast<void*>(0x2000);
  prepared.info.count = 1024 * 1024;
  prepared.info.datatype = ncclFloat32;
  prepared.info.op = ncclSum;
  prepared.info.operation = operation;
  prepared.info.comm = comm;
  return prepared;
}

const cocclCodecModel& modelFor(void* compressor) {
  if (compressor == kIntra) return intraModel;
  if (compressor == kInter) return interModel;
  return defaultModel;
}

void checkSelection(cocclPreparedCall* prepared,
                    cocclAlgorithmKind expected, bool expectedModel) {
  EXPECT(cocclSelectAlgorithm(prepared) == ncclSuccess);
  if (prepared->algorithm != expected) {
    std::fprintf(stderr,
                 "selection mismatch operation=%d count=%zu expected=%d actual=%d one=%g two=%g triple=%g\n",
                 (int)prepared->info.operation, prepared->info.count,
                 (int)expected, (int)prepared->algorithm,
                 lastScores[0], lastScores[1],
                 lastScores[2]);
  }
  EXPECT(prepared->algorithm == expected);
  EXPECT(lastUsedModel == expectedModel);
}

}  // namespace

thread_local int ncclDebugNoWarn = 0;
int ncclDebugLevel = NCCL_LOG_INFO;
uint64_t ncclDebugMask = COCCL_TUNING;
void ncclDebugLogInternal(
    ncclDebugLogLevel, unsigned long, const char*, const char*, int,
    const char* format, ...) {
  if (!captureSelectionLog ||
      std::strncmp(format, "COCCL select ", 13) != 0) return;
  char message[512];
  va_list args;
  va_start(args, format);
  std::vsnprintf(message, sizeof(message), format, args);
  va_end(args);
  int usedModel = 0;
  EXPECT(std::sscanf(std::strstr(message, "one="),
                    "one=%lf two=%lf triple=%lf model=%d",
                    &lastScores[0], &lastScores[1], &lastScores[2],
                    &usedModel) == 4);
  lastUsedModel = usedModel != 0;
}

const cocclCompressorPlugin* cocclCompressorDescriptor(void* compressor) {
  return reinterpret_cast<const cocclCompressorPlugin*>(compressor);
}

bool cocclCompressorSupports(void*, cocclCompressorCapability capability) {
  return framedCompressor &&
      capability == cocclCompressorCapabilityFramed;
}

const cocclConfig& cocclGetConfig() {
  return config;
}

cocclSelectionPerformanceModel cocclAutotuneSnapshotPerformanceModel(
    ncclComm_t, ncclComm_t, ncclComm_t, ncclComm_t) {
  return performance;
}

void cocclAutotuneSnapshotCodecModels(
    void* defaultCompressor, void* intraCompressor, void* interCompressor,
    ncclDataType_t datatype,
    cocclCodecModel* defaultCodec, cocclCodecModel* intraCodec,
    cocclCodecModel* interCodec) {
  snapshotDatatype = datatype;
  if (defaultCompressor != nullptr) *defaultCodec = modelFor(defaultCompressor);
  if (intraCompressor != nullptr) *intraCodec = modelFor(intraCompressor);
  if (interCompressor != nullptr) *interCodec = modelFor(interCompressor);
}

ncclResult_t cocclCommGetHierarchicalComms(
    ncclComm_t comm, cocclHierarchicalComms* hierarchy) {
  *hierarchy = {comm, comm, comm};
  return ncclSuccess;
}

ncclResult_t cocclCommGetZeroCtaComm(ncclComm_t comm, ncclComm_t* child) {
  *child = comm;
  return ncclSuccess;
}

cocclPipelineTuningDecision cocclAutotunePipelineLayout(
    const cocclPipelineSpec* spec) {
  const double cost = spec->stageCount == 1
      ? nativeAllGatherCost
      : (spec->outputLayout == cocclPipelineOutputHierarchicalAllGather
             ? hierarchicalAllGatherCost : 2.0);
  return {spec->rawChunkCount, 1, cost};
}

int main() {
  ncclComm comm = {};
  comm.rank = 0;
  comm.nRanks = 8;
  comm.localRanks = 4;
  comm.nNodes = 2;
  ncclNodeRanks nodeRanks[2] = {};
  nodeRanks[0].localRanks = 4;
  nodeRanks[1].localRanks = 4;
  comm.nodeRanks = nodeRanks;
  // The selector and primitive share this recipe but supply their own comms.
  cocclPreparedCall recipe = makePrepared(&comm, cocclOperation::AllGather);
  ncclComm first = {}, second = {};
  cocclPipelineStage stages[4];
  cocclPipelineSpec spec = cocclBuildAllGatherSpec(
      recipe.info, cocclAlgorithmAllGatherOneShot, nullptr,
      &first, &second, stages);
  EXPECT(spec.stageCount == 1 && stages[0].kind == cocclPipelineStageAllGather);
  EXPECT(stages[0].comm == &first);
  EXPECT(std::strcmp(spec.name, "allgather-native") == 0);
  spec = cocclBuildAllGatherSpec(
      recipe.info, cocclAlgorithmAllGatherOneShot, kDefault,
      &first, &second, stages);
  EXPECT(spec.stageCount == 3 && stages[0].compressor == kDefault);
  EXPECT(stages[1].kind == cocclPipelineStageAllGather && stages[1].comm == &first);
  EXPECT(stages[2].kind == cocclPipelineStageDecompress);
  EXPECT(spec.rawChunkCount == recipe.info.count && spec.inputChunks == 1);
  EXPECT(spec.outputLayout == cocclPipelineOutputContiguous);
  spec = cocclBuildAllGatherSpec(
      recipe.info, cocclAlgorithmAllGatherTwoShot, kInter,
      &first, &second, stages);
  EXPECT(spec.stageCount == 4 && stages[0].compressor == kInter);
  EXPECT(stages[1].comm == &first && stages[2].comm == &second);
  EXPECT(stages[2].kind == cocclPipelineStageAllGather);
  EXPECT(stages[3].kind == cocclPipelineStageDecompress);
  EXPECT(spec.outputLayout == cocclPipelineOutputHierarchicalAllGather);
  config.autotune.enabled = true;

  cocclPreparedCall prepared =
      makePrepared(&comm, cocclOperation::AllGather);
  prepared.info.count = (size_t{64} << 20) / sizeof(float);
  setScope(&prepared, cocclCompressionScope::Default, kDefault);
  setScope(&prepared, cocclCompressionScope::Inter, kInter);
  checkSelection(&prepared, cocclAlgorithmAllGatherTwoShot, true);

  prepared = makePrepared(&comm, cocclOperation::AllGather);
  prepared.info.count = (size_t{64} << 20) / sizeof(float);
  setScope(&prepared, cocclCompressionScope::Inter, kInter);
  checkSelection(&prepared, cocclAlgorithmAllGatherOneShot, true);

  nativeAllGatherCost = 1.0;
  hierarchicalAllGatherCost = 0.97;
  prepared = makePrepared(&comm, cocclOperation::AllGather);
  prepared.info.count = (size_t{64} << 20) / sizeof(float);
  setScope(&prepared, cocclCompressionScope::Inter, kInter);
  checkSelection(&prepared, cocclAlgorithmAllGatherOneShot, true);

  hierarchicalAllGatherCost = 0.9;
  prepared = makePrepared(&comm, cocclOperation::AllGather);
  prepared.info.count = (size_t{64} << 20) / sizeof(float);
  setScope(&prepared, cocclCompressionScope::Inter, kInter);
  checkSelection(&prepared, cocclAlgorithmAllGatherTwoShot, true);
  nativeAllGatherCost = 0.5;
  hierarchicalAllGatherCost = 1.0;

  prepared = makePrepared(&comm, cocclOperation::AllGather);
  setScope(&prepared, cocclCompressionScope::Default, kDefault);
  setScope(&prepared, cocclCompressionScope::Inter, kInter);
  checkSelection(&prepared, cocclAlgorithmAllGatherOneShot, false);

  prepared =
      makePrepared(&comm, cocclOperation::ReduceScatter);
  setScope(&prepared, cocclCompressionScope::Inter, kInter);
  checkSelection(&prepared, cocclAlgorithmReduceScatterTwoShot, true);
  EXPECT(std::isinf(lastScores[0]));
  EXPECT(snapshotDatatype == ncclFloat32);
  EXPECT(std::isfinite(lastScores[1]));

  prepared = makePrepared(&comm, cocclOperation::ReduceScatter);
  setScope(&prepared, cocclCompressionScope::Intra, kIntra);
  checkSelection(&prepared, cocclAlgorithmReduceScatterTwoShot, true);

  config.autotune.enabled = false;
  prepared = makePrepared(&comm, cocclOperation::ReduceScatter);
  setScope(&prepared, cocclCompressionScope::Default, kDefault);
  checkSelection(&prepared, cocclAlgorithmReduceScatterOneShot, false);

  config.autotune.reduceScatterAlgorithm =
      cocclReduceScatterAlgorithmPolicy::TwoShot;
  prepared = makePrepared(&comm, cocclOperation::ReduceScatter);
  setScope(&prepared, cocclCompressionScope::Default, kDefault);
  checkSelection(&prepared, cocclAlgorithmReduceScatterOneShot, false);
  config.autotune.reduceScatterAlgorithm =
      cocclReduceScatterAlgorithmPolicy::Auto;

  prepared = makePrepared(&comm, cocclOperation::AllReduce);
  setScope(&prepared, cocclCompressionScope::Inter, kInter);
  checkSelection(&prepared, cocclAlgorithmAllReduceTripleShot, false);

  prepared = makePrepared(&comm, cocclOperation::AllReduce);
  setScope(&prepared, cocclCompressionScope::Intra, kIntra);
  checkSelection(&prepared, cocclAlgorithmAllReduceTripleShot, false);

  prepared = makePrepared(&comm, cocclOperation::AllReduce);
  setScope(&prepared, cocclCompressionScope::Default, kDefault);
  checkSelection(&prepared, cocclAlgorithmAllReduceTwoShot, false);

  config.autotune.enabled = true;
  interModel.valid = false;
  prepared = makePrepared(&comm, cocclOperation::ReduceScatter);
  setScope(&prepared, cocclCompressionScope::Inter, kInter);
  checkSelection(&prepared, cocclAlgorithmReduceScatterTwoShot, false);
  interModel.valid = true;

  framedCompressor = true;
  interModel.valid = false;
  prepared = makePrepared(&comm, cocclOperation::AllReduce);
  setScope(&prepared, cocclCompressionScope::Inter, kInter);
  checkSelection(&prepared, cocclAlgorithmAllReduceTripleShot, false);
  interModel.valid = true;

  defaultModel.valid = false;
  prepared = makePrepared(&comm, cocclOperation::AllReduce);
  prepared.info.count = (size_t{2} << 30) / sizeof(float);
  setScope(&prepared, cocclCompressionScope::Default, kDefault);
  checkSelection(&prepared, cocclAlgorithmAllReduceOneShot, false);

  prepared = makePrepared(&comm, cocclOperation::AllReduce);
  prepared.info.count = (size_t{3} << 30) / sizeof(float);
  setScope(&prepared, cocclCompressionScope::Default, kDefault);
  checkSelection(&prepared, cocclAlgorithmAllReduceTwoShot, false);
  defaultModel.valid = true;
  framedCompressor = false;

  prepared = makePrepared(&comm, cocclOperation::ReduceScatter);
  setScope(&prepared, cocclCompressionScope::Inter, kInter, false);
  EXPECT(cocclSelectAlgorithm(&prepared) == ncclInvalidArgument);

  prepared = makePrepared(&comm, cocclOperation::ReduceScatter);
  prepared.info.datatype = ncclBfloat16;
  setScope(&prepared, cocclCompressionScope::Inter, kInter);
  checkSelection(&prepared, cocclAlgorithmReduceScatterTwoShot, true);
  EXPECT(snapshotDatatype == ncclBfloat16);

  prepared = makePrepared(&comm, cocclOperation::AllGather);
  prepared.info.count = (size_t{64} << 20) / sizeof(float);
  setScope(&prepared, cocclCompressionScope::Default, kDefault);
  setScope(&prepared, cocclCompressionScope::Inter, kInter);
  captureSelectionLog = false;
  ncclDebugLevel = NCCL_LOG_NONE;
  constexpr int kIterations = 1000000;
  size_t checksum = 0;
  const auto begin = std::chrono::steady_clock::now();
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    EXPECT(cocclSelectAlgorithm(&prepared) == ncclSuccess);
    checksum += (size_t)prepared.algorithm;
  }
  const double nsPerCall =
      std::chrono::duration<double, std::nano>(
          std::chrono::steady_clock::now() - begin).count() /
      kIterations;
  std::printf("allgather_algorithm_query_ns_per_call=%.2f checksum=%zu\n",
              nsPerCall, checksum);
  EXPECT(nsPerCall < 1000.0);

  std::printf("coccl selector scopes: PASS\n");
  return 0;
}
