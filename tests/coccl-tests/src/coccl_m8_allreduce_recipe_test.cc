#include "coccl_allreduce.h"

#include "coccl_pipeline.h"
#include "coccl_prepared_call.h"
#include "comm.h"
#include "debug.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

int pipelineCalls = 0;
int serialPipelineCalls = 0;
cocclAlgorithmKind expectedAlgorithm = cocclAlgorithmNone;

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

cocclPreparedCall makeCall(ncclComm_t comm, cocclAlgorithmKind algorithm,
                           size_t count = 1024) {
  cocclPreparedCall prepared;
  prepared.info.sendbuff = reinterpret_cast<void*>(0x100000);
  prepared.info.recvbuff = reinterpret_cast<void*>(0x200000);
  prepared.info.count = count;
  prepared.info.datatype = ncclFloat32;
  prepared.info.op = ncclSum;
  prepared.info.operation = cocclOperation::AllReduce;
  prepared.info.comm = comm;
  prepared.algorithm = algorithm;
  prepared.policy = cocclDefaultPolicy(cocclOperation::AllReduce);
  prepared.compressors.handles = {
      reinterpret_cast<void*>(0x1),
      reinterpret_cast<void*>(0x1),
      reinterpret_cast<void*>(0x1)};
  prepared.compressors.datatypeSupported = {true, true, true};
  return prepared;
}

}  // namespace

__thread ncclComm_t InterSubComm = nullptr;
__thread ncclComm_t IntraSubComm = nullptr;
thread_local int ncclDebugNoWarn = 0;

void ncclDebugLog(ncclDebugLogLevel, unsigned long, const char*, int,
                  const char*, ...) {}

ncclResult_t cocclRunPipeline(const cocclPipelineSpec* spec) {
  ++pipelineCalls;
  EXPECT(spec->rawChunkCount == 256 && spec->inputChunks == 4);
  EXPECT(spec->inPlaceLayout == cocclPipelineInPlaceSameBuffer);
  EXPECT(spec->inputLayout ==
         (expectedAlgorithm == cocclAlgorithmAllReduceTripleShot
              ? cocclPipelineInputHierarchicalSwizzle
              : cocclPipelineInputContiguous));
  if (expectedAlgorithm == cocclAlgorithmAllReduceOneShot) {
    EXPECT(std::strcmp(spec->name, "allreduce-oneshot") == 0);
    EXPECT(spec->stageCount == 3);
    expectStage(spec, 0, cocclPipelineStageCompress);
    expectStage(spec, 1, cocclPipelineStageAllGather);
    expectStage(spec, 2, cocclPipelineStageDecompressReduce, 4);
  } else if (expectedAlgorithm == cocclAlgorithmAllReduceTwoShot) {
    EXPECT(std::strcmp(spec->name, "allreduce-twoshot") == 0);
    EXPECT(spec->stageCount == 5);
    expectStage(spec, 0, cocclPipelineStageCompress);
    expectStage(spec, 1, cocclPipelineStageAllToAll);
    expectStage(spec, 2, cocclPipelineStageDecompReduceComp, 4);
    expectStage(spec, 3, cocclPipelineStageAllGather);
    expectStage(spec, 4, cocclPipelineStageDecompress);
  } else {
    EXPECT(std::strcmp(spec->name, "allreduce-tripleshot") == 0);
    EXPECT(spec->stageCount == 7);
    expectStage(spec, 0, cocclPipelineStageCompress);
    expectStage(spec, 1, cocclPipelineStageAllToAll);
    expectStage(spec, 2, cocclPipelineStageDecompReduceComp, 2);
    expectStage(spec, 3, cocclPipelineStageAllToAll);
    expectStage(spec, 4, cocclPipelineStageDecompReduceComp, 2);
    expectStage(spec, 5, cocclPipelineStageAllGather);
    expectStage(spec, 6, cocclPipelineStageDecompress);
  }
  return ncclSuccess;
}

ncclResult_t cocclRunPipelineSerial(const cocclPipelineSpec* spec) {
  ++serialPipelineCalls;
  EXPECT(expectedAlgorithm == cocclAlgorithmAllReduceOneShot);
  return cocclRunPipeline(spec);
}

ncclResult_t ncclCommSplit(ncclComm_t, int, int, ncclComm_t*,
                           ncclConfig_t*) {
  return ncclInternalError;
}

int main() {
  ncclComm owner = {};
  owner.nRanks = 4;
  owner.localRanks = 4;
  owner.nNodes = 1;

  for (cocclAlgorithmKind algorithm : {
           cocclAlgorithmAllReduceOneShot,
           cocclAlgorithmAllReduceTwoShot}) {
    expectedAlgorithm = algorithm;
    cocclPreparedCall prepared = makeCall(&owner, algorithm);
    EXPECT(cocclExecuteAllReduce(&prepared) == ncclSuccess);
  }

  ncclComm intra = {};
  intra.nRanks = 2;
  ncclComm inter = {};
  inter.nRanks = 2;
  owner.localRanks = 2;
  owner.nNodes = 2;
  IntraSubComm = &intra;
  InterSubComm = &inter;
  expectedAlgorithm = cocclAlgorithmAllReduceTripleShot;
  cocclPreparedCall triple = makeCall(
      &owner, cocclAlgorithmAllReduceTripleShot);
  EXPECT(cocclExecuteAllReduce(&triple) == ncclSuccess);

  const int callsBeforeInvalid = pipelineCalls;
  cocclPreparedCall tail = makeCall(
      &owner, cocclAlgorithmAllReduceOneShot, 1025);
  EXPECT(cocclExecuteAllReduce(&tail) == ncclInvalidArgument);
  EXPECT(pipelineCalls == callsBeforeInvalid);

  owner.localRanks = 3;
  EXPECT(cocclExecuteAllReduce(&triple) == ncclInvalidArgument);
  EXPECT(pipelineCalls == callsBeforeInvalid);
  EXPECT(pipelineCalls == 3);
  EXPECT(serialPipelineCalls == 1);
  std::printf("coccl M8 allreduce recipe: PASS\n");
  return 0;
}
