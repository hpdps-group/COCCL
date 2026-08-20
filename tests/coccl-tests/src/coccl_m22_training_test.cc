#include "coccl_config.h"
#include "coccl_group_internal.h"
#include "coccl_prepared_call.h"
#include "coccl_runtime.h"
#include "coccl_training_assist.h"
#include "coccl_training_classifier.h"
#include "comm.h"
#include "compress.h"
#include "debug.h"

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <fstream>
#include <string>
#include <vector>

void ncclDebugLog(ncclDebugLogLevel, unsigned long, const char*, int,
                  const char*, ...) {}
thread_local int ncclDebugNoWarn = 0;
__thread int ncclGroupDepth = 0;
__thread ncclResult_t ncclGroupError = ncclSuccess;

namespace {

cocclConfig testConfig;
size_t compressorThreshold = 0;
int dpAllGatherCompressor;
int dpReduceScatterCompressor;
int tpAllReduceCompressor;
int ppForwardCompressor;
int ppBackwardCompressor;
int executionCount = 0;
cocclPreparedCall lastExecution;

const cocclCompressorPlugin sdp4bitDescriptor = {
    COCCL_COMPRESSOR_ABI_VERSION,
    sizeof(cocclCompressorPlugin),
    "sdp4bit",
    COCCL_COMPRESSOR_REQUIRED_CAPABILITIES,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

const cocclCompressorPlugin zfpDescriptor = {
    COCCL_COMPRESSOR_ABI_VERSION,
    sizeof(cocclCompressorPlugin),
    "zfp",
    COCCL_COMPRESSOR_REQUIRED_CAPABILITIES,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

void* compressorFor(cocclTrainingRole role, cocclPolicyKey key) {
  if (role == cocclTrainingRoleDataParallel &&
      key.variant == cocclPolicyVariant::Default) {
    if (key.operation == cocclOperation::AllGather) {
      return &dpAllGatherCompressor;
    }
    if (key.operation == cocclOperation::ReduceScatter) {
      return &dpReduceScatterCompressor;
    }
  }
  if (role == cocclTrainingRoleTensorParallel &&
      key.variant == cocclPolicyVariant::Default &&
      key.operation == cocclOperation::AllReduce) {
    return &tpAllReduceCompressor;
  }
  if (role == cocclTrainingRolePipelineParallel &&
      key.operation == cocclOperation::SendRecv) {
    if (key.variant == cocclPolicyVariant::Forward) {
      return &ppForwardCompressor;
    }
    if (key.variant == cocclPolicyVariant::Backward) {
      return &ppBackwardCompressor;
    }
  }
  return nullptr;
}

ncclResult_t captureExecution(const cocclPreparedCall* prepared) {
  lastExecution = *prepared;
  executionCount++;
  return ncclSuccess;
}

}  // namespace

const cocclConfig& cocclGetConfig() { return testConfig; }
bool cocclCompressionEnabled() { return true; }

ncclResult_t cocclResolveCompressorPolicy(
    cocclTrainingRole role, cocclPolicyKey key,
    cocclResolvedCompressorPolicy* resolved) {
  void* compressor = compressorFor(role, key);
  if (compressor == nullptr) return ncclInvalidUsage;
  resolved->compressor = compressor;
  resolved->thresholdBytes = compressorThreshold;
  return ncclSuccess;
}

bool cocclCompressorSupports(void* compressor,
                             cocclCompressorCapability) {
  return compressor != nullptr;
}

const cocclCompressorPlugin* cocclCompressorDescriptor(void* compressor) {
  return compressor == &tpAllReduceCompressor ||
             compressor == &ppBackwardCompressor
      ? &zfpDescriptor
      : &sdp4bitDescriptor;
}

ncclResult_t cocclSelectAlgorithm(cocclPreparedCall* prepared) {
  prepared->algorithm = prepared->info.operation == cocclOperation::AllReduce
      ? cocclAlgorithmAllReduceOneShot
      : cocclAlgorithmReduceScatterOneShot;
  return ncclSuccess;
}

ncclResult_t cocclGroupEnqueue(const cocclPreparedCall*) {
  return ncclInternalError;
}
ncclResult_t cocclGroupEnqueueNative(const cocclInfo*) {
  return ncclInternalError;
}
ncclResult_t cocclExecuteAllGather(const cocclPreparedCall* prepared) {
  return captureExecution(prepared);
}
ncclResult_t cocclExecuteReduceScatter(const cocclPreparedCall* prepared) {
  return captureExecution(prepared);
}
ncclResult_t cocclExecuteAllReduce(const cocclPreparedCall* prepared) {
  return captureExecution(prepared);
}
ncclResult_t cocclExecuteAllToAll(const cocclPreparedCall* prepared) {
  return captureExecution(prepared);
}
ncclResult_t cocclExecuteSendRecv(const cocclPreparedCall* prepared) {
  return captureExecution(prepared);
}

ncclResult_t ncclAllGather(const void*, void*, size_t, ncclDataType_t,
                           ncclComm_t, cudaStream_t) {
  return ncclSuccess;
}
ncclResult_t ncclReduceScatter(const void*, void*, size_t, ncclDataType_t,
                               ncclRedOp_t, ncclComm_t, cudaStream_t) {
  return ncclSuccess;
}
ncclResult_t ncclAllReduce(const void*, void*, size_t, ncclDataType_t,
                           ncclRedOp_t, ncclComm_t, cudaStream_t) {
  return ncclSuccess;
}
ncclResult_t ncclAllToAll(const void*, void*, size_t, ncclDataType_t,
                          ncclComm_t, cudaStream_t) {
  return ncclSuccess;
}
ncclResult_t ncclSend(const void*, size_t, ncclDataType_t, int,
                      ncclComm_t, cudaStream_t) {
  return ncclSuccess;
}
ncclResult_t ncclRecv(void*, size_t, ncclDataType_t, int,
                      ncclComm_t, cudaStream_t) {
  return ncclSuccess;
}

namespace {

struct TraceBuilder {
  std::vector<cocclTrainingTraceEvent> events;
  std::vector<cocclTrainingIterationRange> iterations;
  uint64_t timestampNs = 0;

  void beginIteration() {
    if (!iterations.empty()) iterations.back().end = events.size();
    iterations.push_back({events.size(), events.size()});
    timestampNs += 1000;
  }

  void add(uint64_t communicatorId, ncclFunc_t operation, size_t bytes,
           int peer = -1) {
    cocclTrainingTraceEvent event;
    event.sequence = events.size() + 1;
    event.communicatorId = communicatorId;
    event.operation = operation;
    event.logicalBytes = bytes;
    event.datatype = ncclFloat32;
    event.peer = peer;
    event.timestampNs = timestampNs;
    timestampNs += 10;
    events.push_back(event);
  }

  void finish() {
    if (!iterations.empty()) iterations.back().end = events.size();
  }
};

static cocclTrainingTraceComm makeComm(uint64_t id, int ranks, int nodes) {
  cocclTrainingTraceComm comm;
  comm.communicatorId = id;
  comm.commHash = id * 101;
  comm.nRanks = ranks;
  comm.nNodes = nodes;
  comm.localRanks = nodes == 0 ? ranks : ranks / nodes;
  return comm;
}

static const cocclTrainingClassification* findClassification(
    uint64_t communicatorId,
    const std::vector<cocclTrainingTraceResult>& results) {
  for (const cocclTrainingTraceResult& result : results) {
    if (result.communicatorId == communicatorId) {
      return &result.classification;
    }
  }
  return nullptr;
}

static int expectRole(
    uint64_t communicatorId, cocclTrainingRole expected,
    const std::vector<cocclTrainingTraceResult>& results,
    const char* scenario) {
  const cocclTrainingClassification* classification =
      findClassification(communicatorId, results);
  if (classification == nullptr || classification->role != expected) {
    fprintf(stderr, "%s: comm %llu expected %s, got %s\n", scenario,
            (unsigned long long)communicatorId,
            cocclTrainingRoleName(expected),
            classification == nullptr
                ? "missing"
                : cocclTrainingRoleName(classification->role));
    return 1;
  }
  return 0;
}

static std::vector<cocclTrainingTraceResult> classify(
    const std::vector<cocclTrainingTraceComm>& comms,
    const TraceBuilder& trace) {
  std::vector<cocclTrainingTraceResult> results;
  cocclTrainingClassifyTrace(comms, trace.events, trace.iterations, 10,
                             testConfig.training, &results);
  return results;
}

static int testIterationDetection() {
  TraceBuilder trace;
  for (int iteration = 0; iteration < 10; ++iteration) {
    trace.beginIteration();
    trace.add(1, ncclFuncAllGather, 128);
    for (int i = 0; i < 6; ++i) trace.add(2, ncclFuncAllReduce, 64);
    trace.add(1, ncclFuncReduceScatter, 512);
  }
  trace.finish();

  std::vector<cocclTrainingIterationRange> detected;
  if (!cocclTrainingDetectIterations(trace.events, 10, &detected) ||
      detected.size() != 10) {
    fprintf(stderr, "iteration detection failed for exact trace\n");
    return 1;
  }
  for (const cocclTrainingIterationRange& iteration : detected) {
    if (iteration.end - iteration.begin != 8) {
      fprintf(stderr, "iteration detector selected an unexpected period\n");
      return 1;
    }
  }

  // One non-boundary bookkeeping change still leaves more than 90 percent of
  // canonical events identical across the ten cycles.
  trace.events[3 * 8 + 3].logicalBytes = 65;
  detected.clear();
  if (!cocclTrainingDetectIterations(trace.events, 10, &detected) ||
      detected.size() != 10) {
    fprintf(stderr, "iteration detection failed for tolerant trace\n");
    return 1;
  }

  std::vector<cocclTrainingTraceEvent> unstable;
  for (size_t i = 0; i < 100; ++i) {
    cocclTrainingTraceEvent event;
    event.communicatorId = 1;
    event.operation = ncclFuncAllReduce;
    event.logicalBytes = i + 1;
    unstable.push_back(event);
  }
  detected.clear();
  if (cocclTrainingDetectIterations(unstable, 10, &detected)) {
    fprintf(stderr, "iteration detector accepted an unstable trace\n");
    return 1;
  }

  std::vector<cocclTrainingTraceEvent> innerLayerRepeats(10);
  for (size_t i = 0; i < innerLayerRepeats.size(); ++i) {
    innerLayerRepeats[i].communicatorId = 2;
    innerLayerRepeats[i].operation = ncclFuncAllReduce;
    innerLayerRepeats[i].logicalBytes = 2ULL << 20;
    innerLayerRepeats[i].timestampNs = i * 10;
  }
  detected.clear();
  if (cocclTrainingDetectIterations(innerLayerRepeats, 5, &detected)) {
    fprintf(stderr, "iteration detector accepted a flat inner-layer loop\n");
    return 1;
  }
  return 0;
}

static int testTensorParallelCommunicationModes() {
  TraceBuilder trace;
  for (int iteration = 0; iteration < 10; ++iteration) {
    trace.beginIteration();
    trace.add(30, ncclFuncAllGather, 1ULL << 20);
    trace.add(31, ncclFuncAllReduce, 2ULL << 20);
    trace.add(31, ncclFuncAllReduce, 2ULL << 20);
    trace.add(30, ncclFuncReduceScatter, 2ULL << 20);
  }
  trace.finish();

  const bool savedSequenceParallel = testConfig.training.sequenceParallel;
  testConfig.training.sequenceParallel = false;
  auto results = classify(
      {makeComm(30, 8, 2), makeComm(31, 2, 1)}, trace);
  testConfig.training.sequenceParallel = savedSequenceParallel;
  return expectRole(30, cocclTrainingRoleDataParallel, results,
                    "SDP AG/RS") ||
         expectRole(31, cocclTrainingRoleTensorParallel, results,
                    "TP AllReduce");
}

static int testConfiguredDataParallelStrategies() {
  TraceBuilder ddpTrace;
  TraceBuilder fsdpTrace;
  for (int iteration = 0; iteration < 10; ++iteration) {
    ddpTrace.beginIteration();
    ddpTrace.add(32, ncclFuncAllReduce, 2ULL << 20);
    ddpTrace.add(32, ncclFuncAllReduce, 4ULL << 20);

    fsdpTrace.beginIteration();
    fsdpTrace.add(33, ncclFuncAllGather, 1ULL << 20);
    fsdpTrace.add(33, ncclFuncAllGather, 2ULL << 20);
    fsdpTrace.add(33, ncclFuncReduceScatter, 3ULL << 20);
  }
  ddpTrace.finish();
  fsdpTrace.finish();

  const cocclDataParallelStrategy savedStrategy =
      testConfig.training.dataParallelStrategy;
  testConfig.training.dataParallelStrategy = cocclDataParallelStrategy::Ddp;
  auto ddpResults = classify({makeComm(32, 8, 2)}, ddpTrace);
  testConfig.training.dataParallelStrategy = cocclDataParallelStrategy::Fsdp;
  auto fsdpResults = classify({makeComm(33, 8, 2)}, fsdpTrace);
  testConfig.training.dataParallelStrategy = savedStrategy;
  return expectRole(32, cocclTrainingRoleDataParallel, ddpResults, "DDP") ||
         expectRole(33, cocclTrainingRoleDataParallel, fsdpResults, "FSDP");
}

static int testConfiguredParallelSizes() {
  TraceBuilder first;
  TraceBuilder second;
  for (int iteration = 0; iteration < 10; ++iteration) {
    first.beginIteration();
    first.add(34, ncclFuncAllReduce, 2ULL << 20);
    first.add(35, ncclFuncAllReduce, 2ULL << 20);
    first.add(36, ncclFuncSend, 2ULL << 20, 1);
    first.add(36, ncclFuncRecv, 2ULL << 20, 1);

    second.beginIteration();
    second.add(37, ncclFuncReduceScatter, 2ULL << 20);
    second.add(38, ncclFuncAllReduce, 2ULL << 20);
  }
  first.finish();
  second.finish();

  const int savedDp = testConfig.training.dataParallelSize;
  const int savedTp = testConfig.training.tensorParallelSize;
  const int savedPp = testConfig.training.pipelineParallelSize;

  testConfig.training.dataParallelSize = 4;
  testConfig.training.tensorParallelSize = 2;
  testConfig.training.pipelineParallelSize = 2;
  auto firstResults = classify({makeComm(34, 4, 2), makeComm(35, 2, 1),
                                makeComm(36, 2, 2)}, first);

  testConfig.training.dataParallelSize = 2;
  testConfig.training.tensorParallelSize = 4;
  auto secondResults = classify(
      {makeComm(37, 2, 2), makeComm(38, 4, 1)}, second);

  testConfig.training.dataParallelSize = savedDp;
  testConfig.training.tensorParallelSize = savedTp;
  testConfig.training.pipelineParallelSize = savedPp;

  return expectRole(34, cocclTrainingRoleDataParallel, firstResults,
                    "configured DP=4") ||
      expectRole(35, cocclTrainingRoleTensorParallel, firstResults,
                 "configured TP=2") ||
      expectRole(36, cocclTrainingRolePipelineParallel, firstResults,
                 "configured PP=2") ||
      expectRole(37, cocclTrainingRoleDataParallel, secondResults,
                 "configured DP=2") ||
      expectRole(38, cocclTrainingRoleTensorParallel, secondResults,
                 "configured TP=4");
}

static int testCrossNodeDpAndTp() {
  TraceBuilder trace;
  for (int iteration = 0; iteration < 10; ++iteration) {
    trace.beginIteration();
    trace.add(1, ncclFuncAllGather, 128);
    // TP's aggregate traffic is deliberately larger than DP's. Classification
    // must follow topology/order/frequency rather than total transferred bytes.
    for (int i = 0; i < 12; ++i) trace.add(2, ncclFuncAllReduce, 64);
    trace.add(1, ncclFuncReduceScatter, 512);
  }
  trace.finish();
  auto results = classify({makeComm(1, 16, 2), makeComm(2, 8, 1)}, trace);
  return expectRole(1, cocclTrainingRoleDataParallel, results,
                    "cross-node DP") ||
         expectRole(2, cocclTrainingRoleTensorParallel, results,
                    "ordered TP");
}

static int testPipelineParallel() {
  TraceBuilder trace;
  for (int iteration = 0; iteration < 10; ++iteration) {
    trace.beginIteration();
    for (int microbatch = 0; microbatch < 4; ++microbatch) {
      trace.add(3, ncclFuncRecv, 1024, 0);
      trace.add(3, ncclFuncSend, 1024, 1);
    }
  }
  trace.finish();
  auto results = classify({makeComm(3, 4, 2)}, trace);
  return expectRole(3, cocclTrainingRolePipelineParallel, results, "PP");
}

static int testOverlapDp() {
  TraceBuilder trace;
  for (int iteration = 0; iteration < 10; ++iteration) {
    trace.beginIteration();
    trace.add(4, ncclFuncReduceScatter, 256);
    trace.add(4, ncclFuncReduceScatter, 768);
    trace.add(4, ncclFuncReduceScatter, 512);
  }
  trace.finish();
  auto results = classify({makeComm(4, 8, 1)}, trace);
  return expectRole(4, cocclTrainingRoleDataParallel, results,
                    "overlap DP");
}

static int testOverlapDpRequiresEightIterations() {
  TraceBuilder sevenOfTen;
  TraceBuilder eightOfTen;
  for (int iteration = 0; iteration < 10; ++iteration) {
    sevenOfTen.beginIteration();
    sevenOfTen.add(10, ncclFuncReduceScatter,
                   iteration < 7 ? 256 : (size_t)1024 + iteration);
    sevenOfTen.add(10, ncclFuncReduceScatter,
                   iteration < 7 ? 768 : (size_t)2048 + iteration);

    eightOfTen.beginIteration();
    eightOfTen.add(11, ncclFuncReduceScatter,
                   iteration < 8 ? 256 : (size_t)1024 + iteration);
    eightOfTen.add(11, ncclFuncReduceScatter,
                   iteration < 8 ? 768 : (size_t)2048 + iteration);
  }
  sevenOfTen.finish();
  eightOfTen.finish();

  auto sevenResults = classify({makeComm(10, 8, 1)}, sevenOfTen);
  auto eightResults = classify({makeComm(11, 8, 1)}, eightOfTen);
  return expectRole(10, cocclTrainingRoleUnknown, sevenResults,
                    "overlap DP 7/10") ||
         expectRole(11, cocclTrainingRoleDataParallel, eightResults,
                    "overlap DP 8/10");
}

static int testAgRsRatioRequiresSixIterations() {
  TraceBuilder fiveOfTen;
  TraceBuilder sixOfTen;
  for (int iteration = 0; iteration < 10; ++iteration) {
    fiveOfTen.beginIteration();
    fiveOfTen.add(12, ncclFuncAllGather, 100);
    fiveOfTen.add(12, ncclFuncReduceScatter,
                  iteration < 5 ? 200 : (size_t)301 + iteration);

    sixOfTen.beginIteration();
    sixOfTen.add(13, ncclFuncAllGather, 100);
    sixOfTen.add(13, ncclFuncReduceScatter,
                 iteration < 6 ? 200 : (size_t)301 + iteration);
  }
  fiveOfTen.finish();
  sixOfTen.finish();

  auto fiveResults = classify({makeComm(12, 8, 1)}, fiveOfTen);
  auto sixResults = classify({makeComm(13, 8, 1)}, sixOfTen);
  return expectRole(12, cocclTrainingRoleUnknown, fiveResults,
                    "AG/RS ratio 5/10") ||
         expectRole(13, cocclTrainingRoleDataParallel, sixResults,
                    "AG/RS ratio 6/10");
}

static int testAgRsPrecisionRatios() {
  TraceBuilder halfTrace;
  TraceBuilder quarterTrace;
  for (int iteration = 0; iteration < 10; ++iteration) {
    halfTrace.beginIteration();
    halfTrace.add(5, ncclFuncAllGather, 100);
    halfTrace.add(5, ncclFuncReduceScatter, 200);

    quarterTrace.beginIteration();
    quarterTrace.add(6, ncclFuncAllGather, 100);
    quarterTrace.add(6, ncclFuncReduceScatter, 400);
  }
  halfTrace.finish();
  quarterTrace.finish();

  auto halfResults = classify({makeComm(5, 8, 1)}, halfTrace);
  auto quarterResults = classify({makeComm(6, 8, 1)}, quarterTrace);
  if (expectRole(5, cocclTrainingRoleDataParallel, halfResults,
                 "AG/RS half") ||
      expectRole(6, cocclTrainingRoleDataParallel, quarterResults,
                 "AG/RS quarter")) {
    return 1;
  }
  const auto* half = findClassification(5, halfResults);
  const auto* quarter = findClassification(6, quarterResults);
  if (half == nullptr || quarter == nullptr ||
      half->agToRsRatio < 0.49 || half->agToRsRatio > 0.51 ||
      quarter->agToRsRatio < 0.24 || quarter->agToRsRatio > 0.26) {
    fprintf(stderr, "AG/RS precision ratios were not retained\n");
    return 1;
  }
  return 0;
}

static int testNodeLocalDdpTail() {
  TraceBuilder trace;
  for (int iteration = 0; iteration < 10; ++iteration) {
    trace.beginIteration();
    for (int i = 0; i < 4; ++i) trace.add(8, ncclFuncAllReduce, 64);
    trace.add(7, ncclFuncAllReduce, 256);
  }
  trace.finish();
  auto results = classify({makeComm(7, 8, 1), makeComm(8, 4, 1)}, trace);
  return expectRole(7, cocclTrainingRoleDataParallel, results,
                    "node-local DDP") ||
         expectRole(8, cocclTrainingRoleTensorParallel, results,
                    "node-local TP");
}

static int testQwenTp2Pp2Trace() {
  TraceBuilder trace;
  trace.add(26, ncclFuncAllReduce, 128);
  for (int iteration = 0; iteration < 10; ++iteration) {
    trace.beginIteration();
    trace.add(20, ncclFuncAllReduce, 4);
    trace.add(20, ncclFuncAllGather, 768);
    trace.add(21, ncclFuncAllReduce, 311164928);
    trace.add(22, ncclFuncAllGather, 1048576);
    trace.add(22, ncclFuncAllReduce, 129024);
    trace.add(22, ncclFuncReduceScatter, 1048576);
    trace.add(27, ncclFuncAllGather, 1048576);
    trace.add(27, ncclFuncAllReduce, 2048);
    trace.add(27, ncclFuncAllReduce, 4096);
    trace.add(27, ncclFuncAllGather, 1048576);
    trace.add(27, ncclFuncReduceScatter, 1048576);
    trace.add(27, ncclFuncAllReduce, 133120);
    trace.add(23, ncclFuncSend, 524288, 1);
    trace.add(23, ncclFuncRecv, 524288, 1);
    trace.add(24, ncclFuncReduceScatter, 440530944);
    trace.add(24, ncclFuncReduceScatter, 311164928);
    trace.add(24, ncclFuncAllGather, 220265472);
    trace.add(24, ncclFuncAllGather, 155582464);
    trace.add(25, ncclFuncAllReduce, 4);
    trace.add(25, ncclFuncAllReduce, 4);
    trace.add(25, ncclFuncAllReduce, 4);
  }
  trace.finish();

  auto results = classify({
      makeComm(20, 8, 2),
      makeComm(21, 2, 2),
      makeComm(22, 2, 1),
      makeComm(23, 2, 2),
      makeComm(24, 2, 1),
      makeComm(25, 4, 2),
      makeComm(26, 2, 1),
      makeComm(27, 2, 1),
  }, trace);
  return expectRole(20, cocclTrainingRoleUnknown, results,
                    "Qwen world control") ||
      expectRole(21, cocclTrainingRoleUnknown, results,
                 "Qwen embedding collective") ||
      expectRole(22, cocclTrainingRoleTensorParallel, results,
                 "Qwen TP") ||
      expectRole(23, cocclTrainingRolePipelineParallel, results,
                 "Qwen PP") ||
      expectRole(24, cocclTrainingRoleDataParallel, results,
                 "Qwen distributed optimizer DP") ||
      expectRole(25, cocclTrainingRoleUnknown, results,
                 "Qwen tiny cross-node control") ||
      expectRole(26, cocclTrainingRoleUnknown, results,
                 "Qwen startup auxiliary") ||
      expectRole(27, cocclTrainingRoleTensorParallel, results,
                 "Qwen multi-size TP");
}

static cocclInfo collectiveInfo(ncclComm_t comm, cocclOperation operation,
                                ncclFunc_t func, size_t count,
                                const void* sendbuff, void* recvbuff) {
  cocclInfo info;
  info.sendbuff = sendbuff;
  info.recvbuff = recvbuff;
  info.count = count;
  info.datatype = ncclFloat32;
  info.op = ncclSum;
  info.func = func;
  info.operation = operation;
  info.comm = comm;
  return info;
}

static int expectCommittedRole(ncclComm_t comm, cocclTrainingRole role,
                               const char* scenario) {
  cocclTrainingClassification classification;
  if (!cocclTrainingAssistQuery(comm, &classification) ||
      classification.role != role) {
    fprintf(stderr, "%s: expected committed role %s\n", scenario,
            cocclTrainingRoleName(role));
    return 1;
  }
  return 0;
}

static int expectRouted(const cocclInfo& info, cocclTrainingRole role,
                        cocclPolicyVariant variant, void* compressor,
                        const char* scenario) {
  const int previousExecutions = executionCount;
  bool isEnqueued = false;
  if (cocclEnqueueCheck(&info, &isEnqueued) != ncclSuccess || !isEnqueued ||
      executionCount != previousExecutions + 1 ||
      lastExecution.trainingRole != role ||
      lastExecution.policy.variant != variant ||
      lastExecution.compressors.get(
          info.operation == cocclOperation::SendRecv
              ? cocclCompressionScope::Inter
              : info.comm->nNodes == 1
                  ? cocclCompressionScope::Intra
                  : cocclCompressionScope::Default) != compressor) {
    fprintf(stderr, "%s: routed to the wrong role or compressor\n", scenario);
    return 1;
  }
  return 0;
}

static int testRoleSpecificCompressorSelection() {
  constexpr size_t kAllGatherCount =
      kCocclTrainingMinimumObservedBytes / sizeof(float) / 8;
  constexpr size_t kReduceScatterCount = 2 * kAllGatherCount;
  constexpr size_t kAllReduceOrP2pCount =
      kCocclTrainingMinimumObservedBytes / sizeof(float);
  int dpRankToNode[8] = {0, 0, 0, 0, 1, 1, 1, 1};
  int tpRankToNode[2] = {0, 0};
  int ppRankToNode[2] = {0, 1};

  ncclComm dp = {};
  dp.rank = 0;
  dp.nRanks = 8;
  dp.nNodes = 2;
  dp.localRanks = 4;
  dp.node = 0;
  dp.commHash = 0x1001;
  dp.rankToNode = dpRankToNode;

  ncclComm tp = {};
  tp.rank = 0;
  tp.nRanks = 2;
  tp.nNodes = 1;
  tp.localRanks = 2;
  tp.node = 0;
  tp.commHash = 0x1002;
  tp.rankToNode = tpRankToNode;

  ncclComm pp = {};
  pp.rank = 0;
  pp.nRanks = 2;
  pp.nNodes = 2;
  pp.localRanks = 1;
  pp.node = 0;
  pp.commHash = 0x1003;
  pp.rankToNode = ppRankToNode;

  ncclComm unknown = {};
  unknown.rank = 0;
  unknown.nRanks = 8;
  unknown.nNodes = 1;
  unknown.localRanks = 8;
  unknown.node = 0;
  unknown.commHash = 0x1004;

  cocclTrainingAssistRegister(&dp);
  cocclTrainingAssistRegister(&tp);
  cocclTrainingAssistRegister(&pp);
  cocclTrainingAssistRegister(&unknown);

  float buffer[512] = {};
  cocclInfo dpAllGather = collectiveInfo(
      &dp, cocclOperation::AllGather, ncclFuncAllGather,
      kAllGatherCount,
      buffer, buffer);
  cocclInfo dpReduceScatter = collectiveInfo(
      &dp, cocclOperation::ReduceScatter, ncclFuncReduceScatter,
      kReduceScatterCount,
      buffer, buffer);
  cocclInfo tpAllReduce = collectiveInfo(
      &tp, cocclOperation::AllReduce, ncclFuncAllReduce,
      kAllReduceOrP2pCount,
      buffer, buffer);
  cocclInfo ppRecv = collectiveInfo(
      &pp, cocclOperation::SendRecv, ncclFuncRecv, kAllReduceOrP2pCount,
      nullptr, buffer);
  ppRecv.peer = 1;
  cocclInfo ppSend = collectiveInfo(
      &pp, cocclOperation::SendRecv, ncclFuncSend, kAllReduceOrP2pCount,
      buffer, nullptr);
  ppSend.peer = 1;

  // Sixteen identical events form one training iteration. At 256 events the
  // runtime detector sees sixteen complete iterations and commits all roles.
  for (int iteration = 0; iteration < 16; ++iteration) {
    cocclTrainingAssistObserve(&dpAllGather, 0);
    for (int tensorOp = 0; tensorOp < 6; ++tensorOp) {
      cocclTrainingAssistObserve(&tpAllReduce, 0);
    }
    for (int microbatch = 0; microbatch < 4; ++microbatch) {
      cocclTrainingAssistObserve(&ppRecv, 1);
      cocclTrainingAssistObserve(&ppSend, 1);
    }
    cocclTrainingAssistObserve(&dpReduceScatter, 0);
    usleep(1000);
  }

  if (expectCommittedRole(&dp, cocclTrainingRoleDataParallel, "DP") ||
      expectCommittedRole(&tp, cocclTrainingRoleTensorParallel, "TP") ||
      expectCommittedRole(&pp, cocclTrainingRolePipelineParallel, "PP") ||
      expectCommittedRole(&unknown, cocclTrainingRoleUnknown, "unknown")) {
    return 1;
  }

  if (expectRouted(dpAllGather, cocclTrainingRoleDataParallel,
                   cocclPolicyVariant::Default, &dpAllGatherCompressor,
                   "DP AllGather") ||
      expectRouted(tpAllReduce, cocclTrainingRoleTensorParallel,
                   cocclPolicyVariant::Default, &tpAllReduceCompressor,
                   "TP AllReduce") ||
      expectRouted(ppSend, cocclTrainingRolePipelineParallel,
                   cocclPolicyVariant::Forward, &ppForwardCompressor,
                   "PP forward Send") ||
      expectRouted(ppRecv, cocclTrainingRolePipelineParallel,
                   cocclPolicyVariant::Backward, &ppBackwardCompressor,
                   "PP backward Recv")) {
    return 1;
  }

  cocclInfo unknownCall = collectiveInfo(
      &unknown, cocclOperation::AllGather, ncclFuncAllGather,
      kAllGatherCount,
      buffer, buffer);
  bool isEnqueued = true;
  if (cocclEnqueueCheck(&unknownCall, &isEnqueued) != ncclSuccess ||
      isEnqueued) {
    fprintf(stderr, "unknown role did not fall back to native NCCL\n");
    return 1;
  }

  cocclInfo missingPolicy = collectiveInfo(
      &dp, cocclOperation::AllReduce, ncclFuncAllReduce,
      kAllReduceOrP2pCount,
      buffer, buffer);
  isEnqueued = true;
  if (cocclEnqueueCheck(&missingPolicy, &isEnqueued) != ncclSuccess ||
      isEnqueued) {
    fprintf(stderr, "missing role/operation policy did not fall back\n");
    return 1;
  }

  compressorThreshold = 1ULL << 30;
  isEnqueued = true;
  if (cocclEnqueueCheck(&dpAllGather, &isEnqueued) != ncclSuccess ||
      isEnqueued) {
    fprintf(stderr, "automatic routing ignored the compression threshold\n");
    return 1;
  }
  const int previousExecutions = executionCount;
  if (cocclEnqueueExplicitCall(
          &dpAllGather, cocclAlgorithmNone) != ncclSuccess ||
      executionCount != previousExecutions + 1 ||
      lastExecution.trainingRole != cocclTrainingRoleDataParallel) {
    fprintf(stderr, "explicit COCCL call did not bypass the threshold\n");
    return 1;
  }
  compressorThreshold = 0;

  cocclTrainingAssistUnregister(&dp);
  cocclTrainingAssistUnregister(&tp);
  cocclTrainingAssistUnregister(&pp);
  cocclTrainingAssistUnregister(&unknown);
  cocclTrainingClassification classification;
  if (cocclTrainingAssistQuery(&dp, &classification)) {
    fprintf(stderr, "unregistered communicator remained queryable\n");
    return 1;
  }
  return 0;
}

static int testRuntimeObservationMinimumBytes() {
  int rankToNode[2] = {0, 0};
  ncclComm comm = {};
  comm.rank = 0;
  comm.nRanks = 2;
  comm.nNodes = 1;
  comm.localRanks = 2;
  comm.rankToNode = rankToNode;
  comm.commHash = 0x1005;
  cocclTrainingAssistRegister(&comm);

  float buffer[1] = {};
  cocclInfo belowMinimum = collectiveInfo(
      &comm, cocclOperation::AllReduce, ncclFuncAllReduce,
      kCocclTrainingMinimumObservedBytes / sizeof(float) - 1,
      buffer, buffer);
  for (int call = 0; call < 256; ++call) {
    cocclTrainingAssistObserve(&belowMinimum, 0);
  }
  cocclTrainingClassification classification;
  if (cocclTrainingAssistQuery(&comm, &classification)) {
    fprintf(stderr, "sub-1MiB calls entered the training trace\n");
    return 1;
  }

  cocclInfo atMinimum = collectiveInfo(
      &comm, cocclOperation::AllReduce, ncclFuncAllReduce,
      kCocclTrainingMinimumObservedBytes / sizeof(float),
      buffer, buffer);
  for (int iteration = 0; iteration < 10; ++iteration) {
    cocclTrainingAssistObserve(&atMinimum, 0);
    cocclTrainingAssistObserve(&atMinimum, 0);
    usleep(1000);
  }
  if (!cocclTrainingAssistQuery(&comm, &classification)) {
    fprintf(stderr, "1MiB calls were not observed by the training trace\n");
    return 1;
  }
  cocclTrainingAssistUnregister(&comm);
  return 0;
}

static int testImmediateConfiguredSizeClassification() {
  const int savedDp = testConfig.training.dataParallelSize;
  const int savedTp = testConfig.training.tensorParallelSize;
  const int savedPp = testConfig.training.pipelineParallelSize;
  testConfig.training.dataParallelSize = 4;
  testConfig.training.tensorParallelSize = 2;
  testConfig.training.pipelineParallelSize = 2;

  ncclComm dp = {};
  dp.nRanks = 4;
  dp.nNodes = 2;
  dp.localRanks = 2;
  dp.commHash = 0x1010;
  ncclComm tp = {};
  tp.nRanks = 2;
  tp.nNodes = 1;
  tp.localRanks = 2;
  tp.commHash = 0x1011;
  ncclComm pp = tp;
  pp.nNodes = 2;
  pp.localRanks = 1;
  pp.commHash = 0x1012;

  float value = 0.0f;
  cocclInfo dpCall = collectiveInfo(
      &dp, cocclOperation::AllReduce, ncclFuncAllReduce, 1, &value, &value);
  cocclInfo tpCall = collectiveInfo(
      &tp, cocclOperation::AllReduce, ncclFuncAllReduce, 1, &value, &value);
  cocclInfo ppCall = collectiveInfo(
      &pp, cocclOperation::SendRecv, ncclFuncSend, 1, &value, nullptr);
  ppCall.peer = 1;

  cocclTrainingAssistRegister(&dp);
  cocclTrainingAssistObserve(&dpCall, 0);
  int result = expectCommittedRole(
      &dp, cocclTrainingRoleDataParallel, "immediate DP size");

  constexpr int kSteadyCalls = 100000;
  struct timespec begin = {};
  struct timespec end = {};
  volatile int roleSum = 0;
  clock_gettime(CLOCK_MONOTONIC, &begin);
  for (int call = 0; call < kSteadyCalls; ++call) {
    cocclTrainingClassification classification;
    cocclTrainingAssistObserve(&dpCall, 0);
    if (cocclTrainingAssistQuery(&dp, &classification)) {
      roleSum += classification.role;
    }
  }
  clock_gettime(CLOCK_MONOTONIC, &end);
  const uint64_t elapsedNs =
      (uint64_t)(end.tv_sec - begin.tv_sec) * 1000000000ULL +
      (uint64_t)end.tv_nsec - (uint64_t)begin.tv_nsec;
  printf("training_steady_observe_query_ns_per_call=%.2f checksum=%d\n",
         (double)elapsedNs / (double)kSteadyCalls, roleSum);
  cocclTrainingAssistUnregister(&dp);

  cocclTrainingAssistRegister(&tp);
  cocclTrainingAssistObserve(&tpCall, 0);
  result |= expectCommittedRole(
      &tp, cocclTrainingRoleTensorParallel, "immediate TP size");
  cocclTrainingAssistUnregister(&tp);

  cocclTrainingAssistRegister(&pp);
  cocclTrainingAssistObserve(&ppCall, 0);
  result |= expectCommittedRole(
      &pp, cocclTrainingRolePipelineParallel, "immediate PP size");
  cocclTrainingAssistUnregister(&pp);

  testConfig.training.dataParallelSize = savedDp;
  testConfig.training.tensorParallelSize = savedTp;
  testConfig.training.pipelineParallelSize = savedPp;
  return result;
}

static int testAmbiguousConstantCollective() {
  TraceBuilder trace;
  for (int iteration = 0; iteration < 10; ++iteration) {
    trace.beginIteration();
    trace.add(9, ncclFuncAllReduce, 64);
  }
  trace.finish();
  auto results = classify({makeComm(9, 8, 1)}, trace);
  return expectRole(9, cocclTrainingRoleUnknown, results,
                    "ambiguous constant collective");
}

static int expectConfigCompressor(const cocclPrimitivePolicy& policy,
                                  cocclCompressionScope scope,
                                  const char* compressor,
                                  const char* scenario) {
  const cocclEffectiveCompressorScope effective =
      cocclEffectiveCompressorScopeFor(policy, scope);
  if (!effective.enabled() || effective.entry->name != compressor ||
      effective.source != cocclCompressionScope::Default) {
    fprintf(stderr, "%s: unexpected configured compressor\n", scenario);
    return 1;
  }
  return 0;
}

static int testTrainingConfig() {
  cocclConfig config;
  std::string error;
  if (!cocclLoadConfigFile(COCCL_M22_CONFIG_FILE, &config, &error) ||
      config.runtime.mode != cocclRuntimeMode::Training ||
      config.runtime.compressionThresholdBytes != 0 ||
      config.training.observationIterations != 5 ||
      config.training.dataParallelSize != 2 ||
      config.training.tensorParallelSize != 2 ||
      config.training.pipelineParallelSize != 2 ||
      config.training.dataParallelStrategy !=
          cocclDataParallelStrategy::Sdp ||
      !config.training.sequenceParallel ||
      config.training.contextParallel) {
    fprintf(stderr, "M22 training config failed to parse: %s\n",
            error.c_str());
    return 1;
  }
  return expectConfigCompressor(
             config.trainingPolicies.dataParallel.allGather,
             cocclCompressionScope::Default, "sdp4bit", "DP AllGather") ||
      expectConfigCompressor(
             config.trainingPolicies.dataParallel.reduceScatter,
             cocclCompressionScope::Inter, "sdp4bit", "DP inter RS") ||
      expectConfigCompressor(
             config.trainingPolicies.tensorParallel.allReduce,
             cocclCompressionScope::Intra, "zfp", "TP intra AR") ||
      expectConfigCompressor(
             config.trainingPolicies.pipelineSendRecvForward,
             cocclCompressionScope::Inter, "sdp4bit", "PP forward") ||
      expectConfigCompressor(
             config.trainingPolicies.pipelineSendRecvBackward,
             cocclCompressionScope::Inter, "zfp", "PP backward");
}

static int testTrainingConfigRequiresParallelSizes() {
  const char* keys[] = {
      "data_parallel_size",
      "tensor_parallel_size",
      "pipeline_parallel_size",
  };
  for (size_t missing = 0; missing < 3; ++missing) {
    std::string contents =
        "schema_version = 3\n"
        "runtime.mode = \"training\"\n"
        "compressor_plugins.compressors = []\n"
        "compressor_plugins.library_path = \".\"\n"
        "[training.classifier]\n";
    for (size_t index = 0; index < 3; ++index) {
      if (index != missing) {
        contents += keys[index];
        contents += " = 2\n";
      }
    }

    std::ofstream output(COCCL_M22_TMP_CONFIG);
    output << contents;
    output.close();

    cocclConfig config;
    std::string error;
    if (cocclLoadConfigFile(COCCL_M22_TMP_CONFIG, &config, &error) ||
        error.find(std::string(keys[missing]) + " is required") ==
            std::string::npos) {
      fprintf(stderr, "training config accepted missing %s\n", keys[missing]);
      ::remove(COCCL_M22_TMP_CONFIG);
      return 1;
    }
  }
  ::remove(COCCL_M22_TMP_CONFIG);
  return 0;
}

static int testOperationDescriptors() {
  for (int value = 0; value < (int)cocclOperation::Count; ++value) {
    const cocclOperation operation = (cocclOperation)value;
    const cocclOperationDescriptor* descriptor =
        cocclGetOperationDescriptor(operation);
    if (descriptor == nullptr || descriptor->operation != operation ||
        descriptor->name == nullptr ||
        !cocclOperationSupportsPolicy(
            descriptor, cocclPolicyVariant::Default)) {
      fprintf(stderr, "operation %d has an incomplete descriptor\n", value);
      return 1;
    }
  }
  const cocclOperationDescriptor* allReduce =
      cocclGetOperationDescriptor(cocclOperation::AllReduce);
  const cocclOperationDescriptor* reduceScatter =
      cocclGetOperationDescriptor(cocclOperation::ReduceScatter);
  const cocclOperationDescriptor* sendRecv =
      cocclGetOperationDescriptor(cocclOperation::SendRecv);
  if (cocclGetOperationDescriptor(cocclOperation::Count) != nullptr ||
      !cocclOperationHasTrait(allReduce, cocclOperationTraitReduction) ||
      !cocclOperationHasTrait(
          allReduce, cocclOperationTraitCountDivisibleByRanks) ||
      !cocclOperationHasTrait(
          reduceScatter, cocclOperationTraitReduction) ||
      !cocclOperationHasTrait(
          reduceScatter, cocclOperationTraitHierarchicalPolicy) ||
      !cocclOperationHasTrait(sendRecv, cocclOperationTraitGrouped) ||
      !cocclOperationSupportsPolicy(
          sendRecv, cocclPolicyVariant::Forward) ||
      cocclOperationSupportsPolicy(
          allReduce, cocclPolicyVariant::Forward)) {
    fprintf(stderr, "operation descriptor semantics are inconsistent\n");
    return 1;
  }
  return 0;
}

}  // namespace

int main() {
  testConfig.runtime.mode = cocclRuntimeMode::Training;
  testConfig.training.observationIterations = 10;
  testConfig.training.maxEvents = 65536;
  testConfig.training.dataParallelSize = 31;
  testConfig.training.tensorParallelSize = 29;
  testConfig.training.pipelineParallelSize = 23;
  testConfig.training.dataParallelStrategy = cocclDataParallelStrategy::Sdp;
  testConfig.training.sequenceParallel = true;
  testConfig.training.contextParallel = false;
  if (testTrainingConfig() ||
      testTrainingConfigRequiresParallelSizes() ||
      testOperationDescriptors() ||
      testIterationDetection() ||
      testTensorParallelCommunicationModes() ||
      testConfiguredDataParallelStrategies() ||
      testConfiguredParallelSizes() ||
      testCrossNodeDpAndTp() ||
      testPipelineParallel() ||
      testOverlapDp() ||
      testOverlapDpRequiresEightIterations() ||
      testAgRsRatioRequiresSixIterations() ||
      testAgRsPrecisionRatios() ||
      testNodeLocalDdpTail() ||
      testQwenTp2Pp2Trace() ||
      testRoleSpecificCompressorSelection() ||
      testRuntimeObservationMinimumBytes() ||
      testImmediateConfiguredSizeClassification() ||
      testAmbiguousConstantCollective()) {
    return 1;
  }
  printf("COCCL training classifier tests passed\n");
  return 0;
}
