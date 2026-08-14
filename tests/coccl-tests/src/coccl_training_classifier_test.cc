#include "training/coccl_training_assist.h"
#include "training/coccl_training_classifier.h"
#include "runtime/coccl_comm.h"
#include "config/coccl_config.h"
#include "runtime/coccl_runtime.h"
#include "comm.h"
#include "debug.h"

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

// training_assist is compiled directly into this host-only test. A logging
// stub keeps the test independent of CUDA devices and libnccl.so.
void ncclDebugLog(ncclDebugLogLevel, unsigned long, const char*, int,
                  const char*, ...) {}
thread_local int ncclDebugNoWarn = 0;

// coccl_comm.cc's detached cleanup references the NCCL split-communicator
// destroy entry point. This host-only test never creates a split communicator.
ncclResult_t ncclCommDestroy(ncclComm_t) { return ncclSuccess; }

// Runtime dependencies below the threshold gate must never execute in this
// host-only test. Returning an error makes a missed gate fail the assertion.
__thread int ncclGroupDepth = 0;
__thread ncclResult_t ncclGroupError = ncclSuccess;
struct ncclCudaGraph;
bool cocclCompressionEnabled() { return true; }
ncclResult_t ncclCudaGetCapturingGraph(ncclCudaGraph*, cudaStream_t) {
  return ncclSuccess;
}
ncclResult_t cocclSelectAlgorithm(cocclPreparedCall*) {
  return ncclInternalError;
}
ncclResult_t cocclGroupEnqueue(const cocclPreparedCall*) {
  return ncclInternalError;
}
ncclResult_t cocclGroupEnqueueNative(const cocclInfo*) {
  return ncclInternalError;
}
ncclResult_t cocclExecuteAllGather(const cocclPreparedCall*) {
  return ncclInternalError;
}
ncclResult_t cocclExecuteReduceScatter(const cocclPreparedCall*) {
  return ncclInternalError;
}
ncclResult_t cocclExecuteAllReduce(const cocclPreparedCall*) {
  return ncclInternalError;
}
ncclResult_t cocclExecuteAllToAll(const cocclPreparedCall*) {
  return ncclInternalError;
}
ncclResult_t cocclExecuteSendRecv(const cocclPreparedCall*) {
  return ncclInternalError;
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

static cocclCompressorHandle makeTestCompressorHandle(const void* identity) {
  cocclCompressorHandle handle;
  handle.state = std::shared_ptr<cocclCompressorRuntimeState>(
      reinterpret_cast<cocclCompressorRuntimeState*>(
          const_cast<void*>(identity)),
      [](cocclCompressorRuntimeState*) {});
  return handle;
}

static int expectConfiguredCompressor(
    ncclComm_t comm, cocclPolicyKey policy,
    const cocclCompressorHandle& expected, const char* scenario) {
  cocclCompressorHandle actual;
  if (cocclCommGetCompressor(comm, policy, &actual) != ncclSuccess ||
      actual.state.get() != expected.state.get()) {
    fprintf(stderr, "%s: selected an unexpected compressor\n", scenario);
    return 1;
  }
  return 0;
}

static int testRoleSpecificCompressorSelection() {
  constexpr size_t kExplicitBypassThreshold = 1ULL << 30;
  ncclComm comm = {};
  comm.rank = 0;
  comm.nRanks = 8;
  comm.nNodes = 2;
  comm.localRanks = 4;
  comm.commHash = 0x1234;

  int dpCompressorIdentity = 0;
  int dpAllGatherCompressorIdentity = 0;
  const cocclCompressorHandle dpCompressor =
      makeTestCompressorHandle(&dpCompressorIdentity);
  const cocclCompressorHandle dpAllGatherCompressor =
      makeTestCompressorHandle(&dpAllGatherCompressorIdentity);

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
      cocclCommSetCompressorPolicy(
          &comm, cocclTrainingRoleDataParallel,
          cocclDefaultPolicy(cocclOperation::ReduceScatter),
          kExplicitBypassThreshold, dpCompressor) != ncclSuccess ||
      cocclCommSetCompressorPolicy(
          &comm, cocclTrainingRoleDataParallel,
          cocclDefaultPolicy(cocclOperation::AllGather),
          0, dpAllGatherCompressor) != ncclSuccess) {
    return 1;
  }

  cocclCompressorHandle selected;
  if (cocclCommGetCompressor(
          &comm, cocclDefaultPolicy(cocclOperation::ReduceScatter),
          &selected) !=
          ncclInvalidUsage ||
      selected || cocclCommCommit(&comm) != ncclSuccess) {
    fprintf(stderr, "pre-commit compressor was executable\n");
    return 1;
  }

  if (cocclTrainingAssistQuery(&comm, &classification) ||
      cocclCommGetCompressor(
          &comm, cocclDefaultPolicy(cocclOperation::ReduceScatter),
          &selected) !=
          ncclInvalidUsage ||
      selected) {
    fprintf(stderr, "unclassified training comm was allowed to compress\n");
    return 1;
  }

  cocclInfo args = {};
  args.count = 64;
  args.datatype = ncclFloat32;
  args.func = ncclFuncReduceScatter;
  args.operation = cocclOperation::ReduceScatter;
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
  cocclResolvedCompressorPolicy resolved;
  if (cocclCommResolveCompressorPolicy(
          &comm, classification.role,
          cocclDefaultPolicy(cocclOperation::ReduceScatter), &resolved) !=
          ncclSuccess ||
      resolved.compressor.state.get() != dpCompressor.state.get() ||
      resolved.thresholdBytes != kExplicitBypassThreshold) {
    fprintf(stderr, "classified DP policy did not resolve correctly\n");
    return 1;
  }
  float sendbuff[512] = {};
  float recvbuff[64] = {};
  cocclInfo routed = {};
  routed.sendbuff = sendbuff;
  routed.recvbuff = recvbuff;
  routed.count = 64;
  routed.datatype = ncclFloat32;
  routed.op = ncclSum;
  routed.func = ncclFuncReduceScatter;
  routed.operation = cocclOperation::ReduceScatter;
  routed.comm = &comm;
  bool isEnqueued = true;
  if (cocclEnqueueCheck(&routed, &isEnqueued) != ncclSuccess || isEnqueued) {
    fprintf(stderr, "automatic routing ignored the compression threshold\n");
    return 1;
  }
  // Direct lookup is used by explicit nccl*Comp* primitives and must not
  // filter a compressor merely because this message is below threshold.
  int result = expectConfiguredCompressor(
      &comm, cocclDefaultPolicy(cocclOperation::ReduceScatter),
      dpCompressor, "classified DP policy");
  if (result == 0 &&
      expectConfiguredCompressor(
           &comm, cocclDefaultPolicy(cocclOperation::AllGather),
           dpAllGatherCompressor,
           "zero-threshold DP AllGather policy")) {
    fprintf(stderr, "zero-threshold role/op policy was not enabled\n");
    result = 1;
  }
  selected = {};
  if (result == 0 &&
      (cocclCommGetCompressor(
           &comm, cocclDefaultPolicy(cocclOperation::AllReduce),
           &selected) !=
           ncclInvalidUsage ||
       selected)) {
    fprintf(stderr, "missing role/op policy was allowed to compress\n");
    result = 1;
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
      !cocclOperationSupportsPolicy(
          allReduce, cocclPolicyVariant::Hierarchical) ||
      !cocclOperationHasTrait(
          reduceScatter, cocclOperationTraitReduction) ||
      !cocclOperationHasTrait(
          reduceScatter, cocclOperationTraitHierarchicalPolicy) ||
      !cocclOperationHasTrait(sendRecv, cocclOperationTraitGrouped) ||
      !cocclOperationSupportsPolicy(
          sendRecv, cocclPolicyVariant::Forward) ||
      cocclOperationSupportsPolicy(
          sendRecv, cocclPolicyVariant::Hierarchical)) {
    fprintf(stderr, "operation descriptor semantics are inconsistent\n");
    return 1;
  }
  return 0;
}

}  // namespace

int main() {
  if (setenv("COCCL_ENABLE", "1", 1) != 0 ||
      setenv("COCCL_CONFIG_FILE", COCCL_TEST_CONFIG_FILE, 1) != 0 ||
      !cocclConfigInitialize() ||
      cocclGetConfig().runtime.mode != cocclRuntimeMode::Training) {
    fprintf(stderr, "failed to load the training test configuration\n");
    return 1;
  }
  if (testOperationDescriptors() ||
      testIterationDetection() ||
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
