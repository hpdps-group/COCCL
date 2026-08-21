#include "core/runtime/coccl_primitive_dispatch.h"

#include "core/pipeline/coccl_pipeline.h"
#include "core/runtime/coccl_comm.h"
#include "core/runtime/coccl_prepared_call.h"
#include "comm.h"
#include "debug.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

cocclAlgorithmKind expectedAlgorithm = cocclAlgorithmNone;
cocclHierarchicalComms hierarchicalComms = {};
int pipelineCalls = 0;

void fail(const char* expression, int line) {
  std::fprintf(stderr, "line %d: %s\n", line, expression);
  std::exit(1);
}

#define EXPECT(expression) \
  do { if (!(expression)) fail(#expression, __LINE__); } while (0)

void expectStage(const cocclPipelineSpec* spec, int index,
                 cocclPipelineStageKind kind, size_t reduceChunks = 0) {
  EXPECT(spec->stages[index].kind == kind);
  EXPECT(spec->stages[index].reduceChunks == reduceChunks);
}

cocclPreparedCall makeCall(ncclComm_t comm, cocclAlgorithmKind algorithm) {
  cocclPreparedCall prepared;
  prepared.info.sendbuff = reinterpret_cast<void*>(0x100000);
  prepared.info.recvbuff = reinterpret_cast<void*>(0x200000);
  prepared.info.count = 1024;
  prepared.info.datatype = ncclFloat32;
  prepared.info.op = ncclSum;
  prepared.info.operation = cocclOperation::ReduceScatter;
  prepared.info.comm = comm;
  prepared.algorithm = algorithm;
  prepared.policy = cocclDefaultPolicy(cocclOperation::ReduceScatter);
  prepared.compressors.handles = {
      reinterpret_cast<void*>(0x1),
      reinterpret_cast<void*>(0x1),
      reinterpret_cast<void*>(0x1)};
  prepared.compressors.datatypeSupported = {true, true, true};
  return prepared;
}

}  // namespace

thread_local int ncclDebugNoWarn = 0;

void ncclDebugLog(ncclDebugLogLevel, unsigned long, const char*, int,
                  const char*, ...) {}

ncclResult_t cocclRunPipeline(const cocclPipelineSpec* spec) {
  ++pipelineCalls;
  EXPECT(spec->rawChunkCount == 1024);
  EXPECT(spec->inputChunks == (size_t)spec->ownerComm->nRanks);
  EXPECT(spec->inPlaceLayout == cocclPipelineInPlaceOutputRankChunk);

  if (expectedAlgorithm == cocclAlgorithmReduceScatterOneShot) {
    EXPECT(std::strcmp(spec->name, "reducescatter-oneshot") == 0);
    EXPECT(spec->inputLayout == cocclPipelineInputContiguous);
    EXPECT(spec->stageCount == 3);
    expectStage(spec, 0, cocclPipelineStageCompress);
    expectStage(spec, 1, cocclPipelineStageAllToAll);
    expectStage(spec, 2, cocclPipelineStageDecompressReduce, 4);
  } else {
    EXPECT(std::strcmp(spec->name, "reducescatter-twoshot") == 0);
    EXPECT(spec->inputLayout == cocclPipelineInputHierarchicalSwizzle);
    EXPECT(spec->stageCount == 5);
    expectStage(spec, 0, cocclPipelineStageCompress);
    expectStage(spec, 1, cocclPipelineStageAllToAll);
    expectStage(spec, 2, cocclPipelineStageDecompReduceComp, 4);
    expectStage(spec, 3, cocclPipelineStageAllToAll);
    expectStage(spec, 4, cocclPipelineStageDecompressReduce, 2);
  }
  return ncclSuccess;
}

ncclResult_t cocclCommGetHierarchicalComms(
    ncclComm_t, cocclHierarchicalComms* comms) {
  *comms = hierarchicalComms;
  return ncclSuccess;
}

int main() {
  ncclComm owner = {};
  owner.nRanks = 4;
  owner.localRanks = 4;
  owner.nNodes = 1;

  expectedAlgorithm = cocclAlgorithmReduceScatterOneShot;
  cocclPreparedCall oneShot = makeCall(
      &owner, cocclAlgorithmReduceScatterOneShot);
  EXPECT(cocclExecuteReduceScatter(&oneShot) == ncclSuccess);

  ncclComm intra = {};
  intra.nRanks = 4;
  ncclComm inter = {};
  inter.nRanks = 2;
  owner.nRanks = 8;
  owner.localRanks = 4;
  owner.nNodes = 2;
  hierarchicalComms = {&owner, &intra, &inter};
  expectedAlgorithm = cocclAlgorithmReduceScatterTwoShot;
  cocclPreparedCall twoShot = makeCall(
      &owner, cocclAlgorithmReduceScatterTwoShot);
  EXPECT(cocclExecuteReduceScatter(&twoShot) == ncclSuccess);

  EXPECT(pipelineCalls == 2);
  std::printf("COCCL reducescatter recipe: PASS\n");
  return 0;
}
