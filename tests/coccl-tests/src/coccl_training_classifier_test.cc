#include "coccl_training_assist.h"
#include "coccl_training_classifier.h"
#include "coccl_comm.h"
#include "coccl_runtime.h"
#include "comm.h"
#include "debug.h"
#include "param.h"

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

// training_assist is compiled directly into this host-only test. Runtime
// logging and NCCL_PARAM loading are irrelevant to the pure trace classifier,
// so small stubs keep the test independent of CUDA devices and libnccl.so.
void ncclDebugLog(ncclDebugLogLevel, unsigned long, const char*, int,
                  const char*, ...) {}
thread_local int ncclDebugNoWarn = 0;

// coccl_comm.cc's detached cleanup references the NCCL split-communicator
// destroy entry point. This host-only test never creates a split communicator.
ncclResult_t ncclCommDestroy(ncclComm_t) { return ncclSuccess; }

void ncclLoadParam(const char* env, int64_t defaultValue, int64_t,
                   int64_t* cache) {
  *cache = strcmp(env, "NCCL_COCCL_TRAINING_MODE") == 0 ? 1 : defaultValue;
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
                             &results);
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
  return 0;
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

struct ChainVisitResult {
  ncclCompressor_t* expected = nullptr;
  int calls = 0;
  bool matched = true;
};

static ncclResult_t recordVisitedCompressor(
    ncclCompressor_t* compressor, void*, void* context) {
  ChainVisitResult* result = static_cast<ChainVisitResult*>(context);
  result->calls++;
  result->matched = result->matched && compressor == result->expected;
  return ncclSuccess;
}

static int expectVisitedCompressor(
    ncclComm_t comm, ncclCommOp_t op, ncclCompressor_t* compressor,
    const char* scenario) {
  ChainVisitResult result;
  result.expected = compressor;
  if (cocclVisitCompressorChain(comm, op, false,
                                recordVisitedCompressor,
                                &result) != ncclSuccess ||
      result.calls != 1 || !result.matched) {
    fprintf(stderr, "%s: selected an unexpected compressor chain\n", scenario);
    return 1;
  }
  return 0;
}

static int testRoleSpecificCompressorSelection() {
  ncclComm comm = {};
  comm.rank = 0;
  comm.nRanks = 8;
  comm.nNodes = 2;
  comm.localRanks = 4;
  comm.commHash = 0x1234;

  ncclCompressor_t defaultCompressor = {};
  defaultCompressor.name = "default";
  ncclCompressor_t dpCompressor = {};
  dpCompressor.name = "dp";

  // The assist registry is intentionally independent from cocclComm. Register
  // twice to verify idempotence before creating the compressor sidecar.
  cocclTrainingAssistRegister(&comm);
  cocclTrainingAssistRegister(&comm);
  cocclTrainingClassification classification;
  if (cocclTrainingAssistQuery(&comm, &classification)) {
    fprintf(stderr, "unclassified training assist query unexpectedly succeeded\n");
    return 1;
  }

  if (cocclCommCreate(&comm) != ncclSuccess ||
      cocclCommAppendDefaultCompressor(
          &comm, &defaultCompressor, nullptr) != ncclSuccess) {
    return 1;
  }
  if (expectVisitedCompressor(&comm, ReduceScatter, &defaultCompressor,
                              "pre-commit autotune chain")) {
    return 1;
  }

  cocclRuntimeCommConfig config = {};
  config.enabled = true;
  config.enableDataParallelComp = true;
  config.enableAllGatherComp = true;
  config.enableAllReduceComp = true;
  config.enableReduceScatterComp = true;
  if (cocclCommSetConfig(&comm, &config) != ncclSuccess ||
      cocclCommResetTrainingRoleChain(
          &comm, cocclTrainingRoleDataParallel,
          ReduceScatter) != ncclSuccess ||
      cocclCommAppendTrainingRoleCompressor(
          &comm, cocclTrainingRoleDataParallel, ReduceScatter,
          &dpCompressor, nullptr) != ncclSuccess ||
      cocclCommSetTrainingCompressionThreshold(
          &comm, cocclTrainingRoleDataParallel, ReduceScatter,
          1024) != ncclSuccess) {
    return 1;
  }

  ChainVisitResult unclassifiedVisit;
  unclassifiedVisit.expected = &defaultCompressor;
  if (cocclTrainingAssistQuery(&comm, &classification) ||
      cocclVisitCompressorChain(&comm, ReduceScatter, false,
                                recordVisitedCompressor,
                                &unclassifiedVisit) != ncclInvalidUsage ||
      unclassifiedVisit.calls != 0) {
    fprintf(stderr, "unclassified training comm was allowed to compress\n");
    return 1;
  }

  cocclRuntimeArgs args = {};
  args.count = 64;
  args.datatype = ncclFloat32;
  args.func = ncclFuncReduceScatter;
  args.comm = &comm;
  // Eight events per iteration match the runtime detector's minimum complete
  // schedule. Ten repetitions commit the cross-node communicator as DP.
  for (int iteration = 0; iteration < 10; ++iteration) {
    for (int event = 0; event < 8; ++event) {
      cocclTrainingAssistObserve(&args, 0);
    }
  }

  if (!cocclTrainingAssistQuery(&comm, &classification) ||
      classification.role != cocclTrainingRoleDataParallel) {
    fprintf(stderr, "role-specific chain test did not classify DP\n");
    return 1;
  }
  if (!cocclCommRoleAllowsCompression(
          &comm, classification.role, ReduceScatter) ||
      cocclCommRoleAllowsCompression(
          &comm, classification.role, SendRecv)) {
    fprintf(stderr, "classified DP comm has incorrect operation permissions\n");
    return 1;
  }
  if (!cocclCommShouldCompress(
          &comm, ReduceScatter, classification.role, 1025,
          ncclFloat16, ncclSum) ||
      !cocclCommShouldCompress(
          &comm, ReduceScatter, classification.role, 1025,
          ncclFloat32, ncclSum) ||
      cocclCommShouldCompress(
          &comm, ReduceScatter, classification.role, 1024,
          ncclFloat32, ncclSum) ||
      cocclCommShouldCompress(
          &comm, ReduceScatter, classification.role, 2048,
          ncclFloat64, ncclSum) ||
      cocclCommShouldCompress(
          &comm, ReduceScatter, classification.role, 2048,
          ncclFloat32, ncclProd)) {
    fprintf(stderr, "unified compression predicate has incorrect threshold, datatype, or reduction-op behavior\n");
    return 1;
  }
#if defined(__CUDA_BF16_TYPES_EXIST__)
  if (!cocclCommShouldCompress(
          &comm, ReduceScatter, classification.role, 1025,
          ncclBfloat16, ncclSum)) {
    fprintf(stderr, "unified compression predicate rejected BF16\n");
    return 1;
  }
#endif
  comm.nRanks = 1;
  bool compressedSingleRank = cocclCommShouldCompress(
      &comm, ReduceScatter, classification.role, 2048,
      ncclFloat32, ncclSum);
  comm.nRanks = 8;
  if (compressedSingleRank) {
    fprintf(stderr, "unified compression predicate accepted one rank\n");
    return 1;
  }
  int result = expectVisitedCompressor(&comm, ReduceScatter, &dpCompressor,
                                       "classified DP chain");
  ChainVisitResult missingRoleOpVisit;
  missingRoleOpVisit.expected = &defaultCompressor;
  if (result == 0 &&
      (cocclCommRoleAllowsCompression(
           &comm, classification.role, AllGather) ||
       cocclVisitCompressorChain(
           &comm, AllGather, false, recordVisitedCompressor,
           &missingRoleOpVisit) != ncclInvalidUsage ||
       missingRoleOpVisit.calls != 0)) {
    fprintf(stderr, "missing role/op chain was allowed to use the default compressor\n");
    result = 1;
  }

  config.enableDataParallelComp = false;
  if (cocclCommSetConfig(&comm, &config) != ncclSuccess ||
      cocclCommRoleAllowsCompression(
          &comm, classification.role, ReduceScatter) ||
      cocclVisitCompressorChain(&comm, ReduceScatter, false,
                                recordVisitedCompressor,
                                &unclassifiedVisit) != ncclInvalidUsage) {
    fprintf(stderr, "disabled DP role was still allowed to compress\n");
    return 1;
  }
  cocclTrainingAssistUnregister(&comm);
  cocclTrainingAssistUnregister(&comm);
  if (cocclTrainingAssistQuery(&comm, &classification)) {
    fprintf(stderr, "destroyed training assist state remained queryable\n");
    return 1;
  }
  bool registryEmpty = false;
  cocclCommDetachedResources* detachedResources = nullptr;
  if (cocclCommDestroy(&comm, &registryEmpty, &detachedResources) !=
          ncclSuccess ||
      cocclCommDestroyDetachedResources(detachedResources) != ncclSuccess ||
      !registryEmpty) {
    fprintf(stderr, "role-specific chain test failed to destroy sidecar state\n");
    return 1;
  }
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

}  // namespace

int main() {
  if (testIterationDetection() ||
      testCrossNodeDpAndTp() ||
      testPipelineParallel() ||
      testOverlapDp() ||
      testOverlapDpRequiresEightIterations() ||
      testAgRsRatioRequiresSixIterations() ||
      testAgRsPrecisionRatios() ||
      testNodeLocalDdpTail() ||
      testRoleSpecificCompressorSelection() ||
      testAmbiguousConstantCollective()) {
    return 1;
  }
  printf("COCCL training classifier tests passed\n");
  return 0;
}
