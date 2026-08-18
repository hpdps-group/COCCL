#include "coccl_config.h"
#include "coccl_group_internal.h"
#include "coccl_prepared_call.h"
#include "comm.h"
#include "compress.h"
#include "debug.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace {

bool enabled = true;
size_t thresholdBytes = 0;
int descriptorQueries = 0;
int policyQueries = 0;
int compressedCalls = 0;
int nativeCalls = 0;
ncclResult_t resolverResult = ncclSuccess;
cocclAlgorithmKind executedAlgorithm = cocclAlgorithmNone;
cocclPolicyKey lastPolicy;
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
  thresholdBytes = 0;
  descriptorQueries = 0;
  policyQueries = 0;
  compressedCalls = 0;
  nativeCalls = 0;
  resolverResult = ncclSuccess;
  executedAlgorithm = cocclAlgorithmNone;
  lastPolicy = {};
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

ncclResult_t cocclResolveCompressorPolicy(
    cocclPolicyKey key, cocclResolvedCompressorPolicy* resolved) {
  ++policyQueries;
  lastPolicy = key;
  if (resolverResult != ncclSuccess) return resolverResult;
  resolved->compressor = reinterpret_cast<void*>(0x1);
  resolved->thresholdBytes = thresholdBytes;
  return ncclSuccess;
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

ncclResult_t ncclAllGather(
    const void*, void*, size_t, ncclDataType_t, ncclComm_t, cudaStream_t) {
  ++nativeCalls;
  return ncclSuccess;
}

ncclResult_t ncclReduceScatter(
    const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t,
    cudaStream_t) {
  ++nativeCalls;
  return ncclSuccess;
}

ncclResult_t ncclAllReduce(
    const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t,
    cudaStream_t) {
  ++nativeCalls;
  return ncclSuccess;
}

ncclResult_t ncclAllToAll(
    const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  ++nativeCalls;
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
  EXPECT(!enqueued && policyQueries == 0);

  reset();
  info = allToAllInfo(&comm);
  resolverResult = ncclInvalidUsage;
  EXPECT(cocclEnqueueCheck(&info, &enqueued) == ncclSuccess);
  EXPECT(!enqueued && policyQueries == 1 && compressedCalls == 0);

  reset();
  info = allToAllInfo(&comm);
  thresholdBytes = std::numeric_limits<size_t>::max();
  EXPECT(cocclEnqueueCheck(&info, &enqueued) == ncclSuccess);
  EXPECT(!enqueued && policyQueries == 1 && compressedCalls == 0);

  reset();
  info = allToAllInfo(&comm);
  EXPECT(cocclEnqueueCheck(&info, &enqueued) == ncclSuccess);
  EXPECT(enqueued && policyQueries == 1 && compressedCalls == 1);

  reset();
  thresholdBytes = std::numeric_limits<size_t>::max();
  info = allToAllInfo(&comm);
  EXPECT(cocclEnqueueExplicitCall(&info, cocclAlgorithmNone) == ncclSuccess);
  EXPECT(policyQueries == 1 && compressedCalls == 1);

  reset();
  info = allToAllInfo(&comm);
  EXPECT(cocclReplayNativeCall(info) == ncclSuccess);
  EXPECT(nativeCalls == 1 && policyQueries == 0 && compressedCalls == 0);

  reset();
  info = allToAllInfo(&comm);
  ncclGroupDepth = 1;
  EXPECT(cocclEnqueueCheck(&info, &enqueued) == ncclSuccess);
  ncclGroupDepth = 0;
  EXPECT(enqueued && grouped.size() == 1 && compressedCalls == 0);

  reset();
  cocclInfo send;
  send.sendbuff = reinterpret_cast<void*>(0x3000);
  send.count = 1024;
  send.datatype = ncclFloat32;
  send.peer = 1;
  send.func = ncclFuncSend;
  send.operation = cocclOperation::SendRecv;
  send.comm = &comm;
  thresholdBytes = std::numeric_limits<size_t>::max();
  ncclGroupDepth = 1;
  EXPECT(cocclEnqueueCheck(&send, &enqueued) == ncclSuccess);
  ncclGroupDepth = 0;
  EXPECT(enqueued && grouped.size() == 1 && grouped[0].compressor == nullptr);

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
  EXPECT(enqueued && policyQueries == 1 &&
         lastPolicy.variant == cocclPolicyVariant::Hierarchical &&
         executedAlgorithm == cocclAlgorithmReduceScatterTwoShot);
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
