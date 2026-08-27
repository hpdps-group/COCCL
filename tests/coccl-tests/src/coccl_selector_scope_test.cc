#include "core/tuning/coccl_autotune_internal.h"

#include "core/compression/coccl_compressor_runtime.h"
#include "core/config/coccl_config.h"
#include "comm.h"
#include "debug.h"

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
  EXPECT(prepared->algorithm == expected);
  EXPECT(prepared->usedModel == expectedModel);
}

}  // namespace

thread_local int ncclDebugNoWarn = 0;
void ncclDebugLog(ncclDebugLogLevel, unsigned long, const char*, int,
                  const char*, ...) {}

const cocclCompressorPlugin* cocclCompressorDescriptor(void* compressor) {
  return reinterpret_cast<const cocclCompressorPlugin*>(compressor);
}

bool cocclCompressorSupports(void*, cocclCompressorCapability) {
  return false;
}

const cocclConfig& cocclGetConfig() {
  return config;
}

cocclSelectionPerformanceModel cocclAutotuneSnapshotPerformanceModel(
    void* defaultCompressor, void* intraCompressor, void* interCompressor,
    ncclDataType_t datatype,
    cocclCodecModel* defaultCodec, cocclCodecModel* intraCodec,
    cocclCodecModel* interCodec) {
  snapshotDatatype = datatype;
  if (defaultCompressor != nullptr) *defaultCodec = modelFor(defaultCompressor);
  if (intraCompressor != nullptr) *intraCodec = modelFor(intraCompressor);
  if (interCompressor != nullptr) *interCodec = modelFor(interCompressor);
  return performance;
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
  config.autotune.enabled = true;

  cocclPreparedCall prepared =
      makePrepared(&comm, cocclOperation::ReduceScatter);
  setScope(&prepared, cocclCompressionScope::Inter, kInter);
  checkSelection(&prepared, cocclAlgorithmReduceScatterTwoShot, true);
  EXPECT(std::isinf(prepared.oneShotUs));
  EXPECT(snapshotDatatype == ncclFloat32);
  EXPECT(std::isfinite(prepared.twoShotUs));

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

  prepared = makePrepared(&comm, cocclOperation::ReduceScatter);
  setScope(&prepared, cocclCompressionScope::Inter, kInter, false);
  EXPECT(cocclSelectAlgorithm(&prepared) == ncclInvalidArgument);

  prepared = makePrepared(&comm, cocclOperation::ReduceScatter);
  prepared.info.datatype = ncclBfloat16;
  setScope(&prepared, cocclCompressionScope::Inter, kInter);
  checkSelection(&prepared, cocclAlgorithmReduceScatterTwoShot, true);
  EXPECT(snapshotDatatype == ncclBfloat16);

  std::printf("coccl selector scopes: PASS\n");
  return 0;
}
