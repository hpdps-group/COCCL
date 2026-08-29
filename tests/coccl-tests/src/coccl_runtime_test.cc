#include "core/config/coccl_config.h"
#include "core/runtime/coccl_group_internal.h"
#include "core/runtime/coccl_prepared_call.h"
#include "comm.h"
#include "core/compression/compress.h"
#include "debug.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace {

bool enabled = true;
bool bytewiseLossless = false;
size_t thresholdBytes = 0;
int descriptorQueries = 0;
int policyQueries = 0;
int compressedCalls = 0;
int nativeCalls = 0;
uint64_t lastProfilerTag = 0;
ncclResult_t resolverResult = ncclSuccess;
cocclAlgorithmKind executedAlgorithm = cocclAlgorithmNone;
std::array<void*, 3> scopeCompressors;
std::vector<cocclPolicyKey> queriedPolicies;
cocclConfig config;
std::vector<cocclPreparedCall> grouped;

constexpr cocclOperationDescriptor descriptors[] = {
    {cocclOperation::AllGather, "AllGather",
     cocclOperationTraitScaleBytesByRanks | cocclOperationTraitGrouped},
    {cocclOperation::ReduceScatter, "ReduceScatter",
     cocclOperationTraitScaleBytesByRanks | cocclOperationTraitReduction |
         cocclOperationTraitGrouped |
         cocclOperationTraitHierarchicalPolicy},
    {cocclOperation::AllReduce, "AllReduce",
     cocclOperationTraitReduction | cocclOperationTraitGrouped |
         cocclOperationTraitCountDivisibleByRanks |
         cocclOperationTraitHierarchicalPolicy},
    {cocclOperation::AllToAll, "AllToAll",
     cocclOperationTraitScaleBytesByRanks | cocclOperationTraitGrouped},
    {cocclOperation::SendRecv, "SendRecv",
     cocclOperationTraitDirectionalPolicy | cocclOperationTraitGrouped},
};

void fail(const char* expression, int line) {
  std::fprintf(stderr, "line %d: %s\n", line, expression);
  std::exit(1);
}

#define EXPECT(expression) \
  do { if (!(expression)) fail(#expression, __LINE__); } while (0)

void reset() {
  enabled = true;
  bytewiseLossless = false;
  thresholdBytes = 0;
  descriptorQueries = 0;
  policyQueries = 0;
  compressedCalls = 0;
  nativeCalls = 0;
  lastProfilerTag = 0;
  resolverResult = ncclSuccess;
  executedAlgorithm = cocclAlgorithmNone;
  scopeCompressors = {
      reinterpret_cast<void*>(0x10), reinterpret_cast<void*>(0x11),
      reinterpret_cast<void*>(0x12)};
  queriedPolicies.clear();
  config = {};
  grouped.clear();
}

cocclInfo allToAllInfo(ncclComm_t comm) {
  cocclInfo info;
  info.sendbuff = reinterpret_cast<void*>(0x1000);
  info.recvbuff = reinterpret_cast<void*>(0x2000);
  info.count = 1024;
  info.datatype = ncclFloat32;
  info.operation = cocclOperation::AllToAll;
  info.comm = comm;
  return info;
}

void enableOnly(cocclCompressionScope scope) {
  scopeCompressors = {nullptr, nullptr, nullptr};
  scopeCompressors[static_cast<size_t>(scope)] =
      reinterpret_cast<void*>(0x10 + static_cast<size_t>(scope));
}

cocclInfo sendInfo(ncclComm_t comm, int peer) {
  cocclInfo info;
  info.sendbuff = reinterpret_cast<void*>(0x3000);
  info.count = 1024;
  info.datatype = ncclFloat32;
  info.peer = peer;
  info.func = ncclFuncSend;
  info.operation = cocclOperation::SendRecv;
  info.comm = comm;
  return info;
}

double benchmark(const cocclInfo& info, int iterations) {
  const auto begin = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; ++i) {
    bool enqueued = false;
    EXPECT(cocclEnqueueCheck(&info, &enqueued) == ncclSuccess);
  }
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::nano>(end - begin).count() /
      iterations;
}

}  // namespace

__thread int ncclGroupDepth = 0;
thread_local int ncclDebugNoWarn = 0;

void ncclDebugLog(ncclDebugLogLevel, unsigned long, const char*, int,
                  const char*, ...) {}

bool cocclCompressionEnabled() {
  return enabled;
}

bool cocclCompressorSupports(
    void*, cocclCompressorCapability capability) {
  return capability == cocclCompressorCapabilityBytewiseLossless &&
      bytewiseLossless;
}

ncclResult_t cocclResolveCompressorPolicy(
    cocclTrainingRole, cocclPolicyKey key,
    cocclResolvedCompressorPolicy* resolved) {
  ++policyQueries;
  queriedPolicies.push_back(key);
  if (resolverResult != ncclSuccess) return resolverResult;
  resolved->compressor = scopeCompressors[static_cast<size_t>(key.scope)];
  if (resolved->compressor == nullptr) return ncclInvalidUsage;
  resolved->thresholdBytes = thresholdBytes;
  return ncclSuccess;
}

const cocclCompressorPlugin* cocclCompressorDescriptor(void*) {
  static cocclCompressorPlugin plugin = {};
  plugin.name = "test";
  return &plugin;
}

bool cocclTrainingAssistEnabled() {
  return false;
}

void cocclTrainingAssistObserve(const cocclInfo*, int) {}

bool cocclTrainingAssistQuery(
    ncclComm_t, cocclTrainingClassification*) {
  return false;
}

const char* cocclTrainingRoleName(cocclTrainingRole) {
  return "Unknown";
}

const cocclOperationDescriptor* cocclGetOperationDescriptor(
    cocclOperation operation) {
  ++descriptorQueries;
  const size_t index = static_cast<size_t>(operation);
  return index < static_cast<size_t>(cocclOperation::Count)
      ? &descriptors[index] : nullptr;
}

bool cocclOperationSupportsPolicy(const cocclOperationDescriptor*,
                                  cocclPolicyVariant) {
  return true;
}

const cocclConfig& cocclGetConfig() {
  return config;
}

ncclResult_t cocclSelectAlgorithm(cocclPreparedCall* prepared) {
  const bool hierarchical = prepared->info.comm->nNodes > 1 &&
      prepared->info.comm->localRanks > 1;
  if (prepared->info.operation == cocclOperation::ReduceScatter) {
    prepared->algorithm =
        config.autotune.reduceScatterAlgorithm ==
                cocclReduceScatterAlgorithmPolicy::OneShot ||
            !hierarchical
        ? cocclAlgorithmReduceScatterOneShot
        : cocclAlgorithmReduceScatterTwoShot;
  } else {
    switch (config.autotune.allReduceAlgorithm) {
      case cocclAllReduceAlgorithmPolicy::OneShot:
        prepared->algorithm = cocclAlgorithmAllReduceOneShot;
        break;
      case cocclAllReduceAlgorithmPolicy::TripleShot:
        prepared->algorithm = hierarchical
            ? cocclAlgorithmAllReduceTripleShot
            : cocclAlgorithmAllReduceTwoShot;
        break;
      case cocclAllReduceAlgorithmPolicy::Auto:
      case cocclAllReduceAlgorithmPolicy::TwoShot:
        prepared->algorithm = cocclAlgorithmAllReduceTwoShot;
        break;
    }
  }
  return ncclSuccess;
}

ncclResult_t cocclGroupEnqueue(const cocclPreparedCall* prepared) {
  grouped.push_back(*prepared);
  return ncclSuccess;
}

ncclResult_t cocclGroupEnqueueNative(const cocclInfo* info) {
  cocclPreparedCall pending;
  pending.info = *info;
  grouped.push_back(pending);
  return ncclSuccess;
}

ncclResult_t cocclExecuteAllGather(const cocclPreparedCall*) {
  ++compressedCalls;
  return ncclSuccess;
}

ncclResult_t cocclExecuteAllToAll(const cocclPreparedCall*) {
  ++compressedCalls;
  return ncclSuccess;
}

ncclResult_t cocclExecuteReduceScatter(const cocclPreparedCall* prepared) {
  ++compressedCalls;
  executedAlgorithm = prepared->algorithm;
  return ncclSuccess;
}

ncclResult_t cocclExecuteAllReduce(const cocclPreparedCall* prepared) {
  ++compressedCalls;
  executedAlgorithm = prepared->algorithm;
  return ncclSuccess;
}

ncclResult_t cocclExecuteSendRecv(const cocclPreparedCall*) {
  ++compressedCalls;
  return ncclSuccess;
}

ncclResult_t ncclAllGatherConfig(
    const void*, void*, size_t, ncclDataType_t, ncclComm_t, cudaStream_t,
    const ncclCollConfig_t* config) {
  ++nativeCalls;
  lastProfilerTag = config->userProfilerTag;
  return ncclSuccess;
}

ncclResult_t ncclReduceScatterConfig(
    const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t,
    cudaStream_t, const ncclCollConfig_t* config) {
  ++nativeCalls;
  lastProfilerTag = config->userProfilerTag;
  return ncclSuccess;
}

ncclResult_t ncclAllReduceConfig(
    const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t,
    cudaStream_t, const ncclCollConfig_t* config) {
  ++nativeCalls;
  lastProfilerTag = config->userProfilerTag;
  return ncclSuccess;
}

ncclResult_t ncclAlltoAllConfig(
    const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream,
    const ncclCollConfig_t* config) {
  ++nativeCalls;
  lastProfilerTag = config->userProfilerTag;
  cocclInfo nested;
  nested.sendbuff = sendbuff;
  nested.recvbuff = recvbuff;
  nested.count = count;
  nested.datatype = datatype;
  nested.operation = cocclOperation::AllToAll;
  nested.comm = comm;
  nested.stream = stream;
  bool enqueued = true;
  EXPECT(cocclEnqueueCheck(&nested, &enqueued) == ncclSuccess);
  EXPECT(!enqueued);
  return ncclSuccess;
}

ncclResult_t ncclSend(
    const void*, size_t, ncclDataType_t, int, ncclComm_t,
    cudaStream_t) {
  ++nativeCalls;
  return ncclSuccess;
}

ncclResult_t ncclRecv(
    void*, size_t, ncclDataType_t, int, ncclComm_t,
    cudaStream_t) {
  ++nativeCalls;
  return ncclSuccess;
}

int main() {
  ncclComm comm = {};
  comm.nRanks = 4;
  comm.nNodes = 1;
  comm.localRanks = 4;
  comm.rank = 0;
  comm.node = 0;
  int rankToNode[] = {0, 0, 0, 0};
  comm.rankToNode = rankToNode;
  cocclInfo info = allToAllInfo(&comm);
  bool enqueued = false;

  reset();
  enabled = false;
  info.operation = cocclOperation::Count;
  EXPECT(cocclEnqueueCheck(&info, &enqueued) == ncclSuccess);
  EXPECT(!enqueued && descriptorQueries == 0 && policyQueries == 0);

  reset();
  info = allToAllInfo(&comm);
  info.datatype = ncclInt8;
  EXPECT(cocclEnqueueCheck(&info, &enqueued) == ncclSuccess);
  EXPECT(!enqueued && policyQueries == 3);

  for (ncclDataType_t datatype : {ncclInt8, ncclInt32, ncclInt64}) {
    reset();
    bytewiseLossless = true;
    info = allToAllInfo(&comm);
    info.datatype = datatype;
    EXPECT(cocclEnqueueCheck(&info, &enqueued) == ncclSuccess);
    EXPECT(enqueued && policyQueries == 3 && compressedCalls == 1);
  }

  reset();
  info = allToAllInfo(&comm);
  info.datatype = ncclInt32;
  EXPECT(cocclEnqueueExplicitCall(&info, cocclAlgorithmNone) == ncclSuccess);
  EXPECT(policyQueries == 3 && nativeCalls == 1 && compressedCalls == 0);

  reset();
  bytewiseLossless = true;
  info = allToAllInfo(&comm);
  info.datatype = ncclInt64;
  EXPECT(cocclEnqueueExplicitCall(&info, cocclAlgorithmNone) == ncclSuccess);
  EXPECT(policyQueries == 3 && compressedCalls == 1);

  reset();
  info = allToAllInfo(&comm);
  resolverResult = ncclInvalidUsage;
  EXPECT(cocclEnqueueCheck(&info, &enqueued) == ncclSuccess);
  EXPECT(!enqueued && policyQueries == 3 && compressedCalls == 0);

  reset();
  info = allToAllInfo(&comm);
  thresholdBytes = std::numeric_limits<size_t>::max();
  EXPECT(cocclEnqueueCheck(&info, &enqueued) == ncclSuccess);
  EXPECT(!enqueued && policyQueries == 3 && compressedCalls == 0);

  reset();
  info = allToAllInfo(&comm);
  EXPECT(cocclEnqueueCheck(&info, &enqueued) == ncclSuccess);
  EXPECT(enqueued && policyQueries == 3 && compressedCalls == 1);

  reset();
  enableOnly(cocclCompressionScope::Inter);
  info = allToAllInfo(&comm);
  EXPECT(cocclEnqueueCheck(&info, &enqueued) == ncclSuccess);
  EXPECT(!enqueued && compressedCalls == 0);

  reset();
  enableOnly(cocclCompressionScope::Intra);
  info = allToAllInfo(&comm);
  EXPECT(cocclEnqueueCheck(&info, &enqueued) == ncclSuccess);
  EXPECT(enqueued && compressedCalls == 1);

  reset();
  thresholdBytes = std::numeric_limits<size_t>::max();
  info = allToAllInfo(&comm);
  EXPECT(cocclEnqueueExplicitCall(&info, cocclAlgorithmNone) == ncclSuccess);
  EXPECT(policyQueries == 3 && compressedCalls == 1);

  reset();
  info = allToAllInfo(&comm);
  info.profilerTag = 0x1234;
  EXPECT(cocclReplayNativeCall(info) == ncclSuccess);
  EXPECT(nativeCalls == 1 && policyQueries == 0 && compressedCalls == 0 &&
         lastProfilerTag == 0x1234);

  reset();
  info = allToAllInfo(&comm);
  ncclGroupDepth = 1;
  EXPECT(cocclEnqueueCheck(&info, &enqueued) == ncclSuccess);
  ncclGroupDepth = 0;
  EXPECT(enqueued && grouped.size() == 1 && compressedCalls == 0);

  reset();
  cocclInfo send = sendInfo(&comm, 1);
  thresholdBytes = std::numeric_limits<size_t>::max();
  ncclGroupDepth = 1;
  EXPECT(cocclEnqueueCheck(&send, &enqueued) == ncclSuccess);
  ncclGroupDepth = 0;
  EXPECT(enqueued && grouped.size() == 1 &&
         !grouped[0].compressors.anyEnabled());

  rankToNode[2] = 1;
  comm.nNodes = 2;
  comm.localRanks = 2;

  reset();
  enableOnly(cocclCompressionScope::Intra);
  send = sendInfo(&comm, 1);
  EXPECT(cocclEnqueueCheck(&send, &enqueued) == ncclSuccess);
  EXPECT(enqueued && compressedCalls == 1);

  reset();
  enableOnly(cocclCompressionScope::Intra);
  send = sendInfo(&comm, 2);
  EXPECT(cocclEnqueueCheck(&send, &enqueued) == ncclSuccess);
  EXPECT(!enqueued && compressedCalls == 0);

  reset();
  enableOnly(cocclCompressionScope::Inter);
  send = sendInfo(&comm, 2);
  EXPECT(cocclEnqueueCheck(&send, &enqueued) == ncclSuccess);
  EXPECT(enqueued && compressedCalls == 1);

  reset();
  enableOnly(cocclCompressionScope::Inter);
  send = sendInfo(&comm, 1);
  EXPECT(cocclEnqueueCheck(&send, &enqueued) == ncclSuccess);
  EXPECT(!enqueued && compressedCalls == 0);

  comm.nNodes = 1;
  comm.localRanks = 4;
  rankToNode[2] = 0;

  reset();
  cocclInfo reduction = allToAllInfo(&comm);
  reduction.operation = cocclOperation::AllReduce;
  reduction.op = ncclSum;
  reduction.count = 1024;
  config.autotune.allReduceAlgorithm =
      cocclAllReduceAlgorithmPolicy::OneShot;
  EXPECT(cocclEnqueueCheck(&reduction, &enqueued) == ncclSuccess);
  EXPECT(enqueued && executedAlgorithm == cocclAlgorithmAllReduceOneShot);

  reset();
  reduction = allToAllInfo(&comm);
  reduction.operation = cocclOperation::AllReduce;
  reduction.op = ncclSum;
  reduction.count = 1025;
  EXPECT(cocclEnqueueCheck(&reduction, &enqueued) == ncclSuccess);
  EXPECT(!enqueued && policyQueries == 0);

  reset();
  comm.nNodes = 2;
  comm.localRanks = 2;
  reduction = allToAllInfo(&comm);
  reduction.operation = cocclOperation::ReduceScatter;
  reduction.op = ncclSum;
  EXPECT(cocclEnqueueCheck(&reduction, &enqueued) == ncclSuccess);
  EXPECT(enqueued && policyQueries == 3 &&
         executedAlgorithm == cocclAlgorithmReduceScatterTwoShot);

  reset();
  enableOnly(cocclCompressionScope::Intra);
  reduction = allToAllInfo(&comm);
  reduction.operation = cocclOperation::ReduceScatter;
  reduction.op = ncclSum;
  EXPECT(cocclEnqueueCheck(&reduction, &enqueued) == ncclSuccess);
  EXPECT(enqueued &&
         executedAlgorithm == cocclAlgorithmReduceScatterTwoShot);

  reset();
  enableOnly(cocclCompressionScope::Inter);
  reduction = allToAllInfo(&comm);
  reduction.operation = cocclOperation::ReduceScatter;
  reduction.op = ncclSum;
  EXPECT(cocclEnqueueCheck(&reduction, &enqueued) == ncclSuccess);
  EXPECT(enqueued &&
         executedAlgorithm == cocclAlgorithmReduceScatterTwoShot);

  reset();
  scopeCompressors = {nullptr, nullptr, nullptr};
  reduction = allToAllInfo(&comm);
  reduction.operation = cocclOperation::ReduceScatter;
  reduction.op = ncclSum;
  EXPECT(cocclEnqueueCheck(&reduction, &enqueued) == ncclSuccess);
  EXPECT(!enqueued && compressedCalls == 0);

  reset();
  enableOnly(cocclCompressionScope::Default);
  reduction = allToAllInfo(&comm);
  reduction.operation = cocclOperation::AllReduce;
  reduction.op = ncclSum;
  config.autotune.allReduceAlgorithm =
      cocclAllReduceAlgorithmPolicy::TripleShot;
  EXPECT(cocclEnqueueCheck(&reduction, &enqueued) == ncclSuccess);
  EXPECT(enqueued &&
         executedAlgorithm == cocclAlgorithmAllReduceTripleShot);

  reset();
  enableOnly(cocclCompressionScope::Inter);
  info = allToAllInfo(&comm);
  EXPECT(cocclEnqueueCheck(&info, &enqueued) == ncclSuccess);
  EXPECT(!enqueued && compressedCalls == 0);
  comm.nNodes = 1;
  comm.localRanks = 4;

  reset();
  reduction = allToAllInfo(&comm);
  reduction.operation = cocclOperation::ReduceScatter;
  reduction.op = ncclMax;
  EXPECT(cocclEnqueueCheck(&reduction, &enqueued) == ncclSuccess);
  EXPECT(!enqueued && policyQueries == 0);

  reset();
  enabled = false;
  info = allToAllInfo(&comm);
  const double disabledNs = benchmark(info, 500000);
  reset();
  thresholdBytes = std::numeric_limits<size_t>::max();
  const double thresholdNs = benchmark(info, 500000);
  std::printf("disabled_ns_per_call=%.2f threshold_ns_per_call=%.2f\n",
              disabledNs, thresholdNs);
  return 0;
}
