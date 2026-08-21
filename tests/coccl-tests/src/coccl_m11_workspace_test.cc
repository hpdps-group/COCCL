#include "coccl_m11_size_query_stub.h"
#include "core/pipeline/coccl_pipeline_internal.h"

#include "comm.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

void fail(const char* expression, int line) {
  std::fprintf(stderr, "line %d: %s\n", line, expression);
  std::exit(1);
}

#define EXPECT(expression) \
  do { if (!(expression)) fail(#expression, __LINE__); } while (0)

bool rangesOverlap(size_t firstOffset, size_t firstBytes,
                   size_t secondOffset, size_t secondBytes) {
  return firstOffset < secondOffset + secondBytes &&
      secondOffset < firstOffset + firstBytes;
}

void validateLayout(const cocclPipelinePlan& plan, int depth) {
  EXPECT(plan.registeredBytes ==
         (size_t)depth * plan.registeredSliceBytes);
  EXPECT(plan.totalBytes == plan.registeredBytes + plan.rawBytes);
  for (int temp = 0; temp < plan.tempCount; ++temp) {
    const cocclPipelineTempPlan& item = plan.temps[temp];
    EXPECT(item.logicalBytes <= item.alignedBytes);
    EXPECT(item.alignedBytes % kCocclPipelineAlignment == 0);
    EXPECT(item.offset % kCocclPipelineAlignment == 0);
    if (item.storage == cocclPipelineRawRing) {
      EXPECT(item.offset + 2 * item.alignedBytes <= plan.rawBytes);
      EXPECT(item.offset == item.offset + (2 % 2) * item.alignedBytes);
      EXPECT(item.offset != item.offset + (1 % 2) * item.alignedBytes);
    } else {
      EXPECT(item.offset + item.alignedBytes <=
             plan.registeredSliceBytes);
    }
    if (temp > 0) {
      const cocclPipelineTempPlan& previous = plan.temps[temp - 1];
      if (previous.storage == item.storage) {
        EXPECT(!rangesOverlap(previous.offset, previous.alignedBytes,
                              item.offset, item.alignedBytes));
      }
    }
  }
}

void expectWorkspace(const size_t* sizes, int count, int depth,
                     int inputStaging, int outputStaging,
                     cocclPipelineWorkspaceKind kind,
                     size_t registeredBytes, size_t rawBytes) {
  cocclPipelinePlan plan = {};
  plan.tempCount = count;
  plan.inputStagingTemp = inputStaging;
  plan.outputStagingTemp = outputStaging;
  for (int temp = 0; temp < count; ++temp) {
    plan.temps[temp].logicalBytes = sizes[temp];
    plan.temps[temp].alignedBytes = sizes[temp];
  }
  EXPECT(cocclPlanPipelineWorkspace(&plan, depth) == ncclSuccess);
  EXPECT(plan.workspaceKind == kind);
  EXPECT(plan.registeredBytes == registeredBytes);
  EXPECT(plan.rawBytes == rawBytes);
  validateLayout(plan, depth);
}

void checkWorkspaceSelection() {
  constexpr size_t raw = 1024;
  constexpr size_t encoded = 256;
  const size_t allToAll[] = {raw, encoded, encoded, raw};
  const size_t allGather[] = {encoded, 4 * encoded, 4 * raw};
  const size_t tie[] = {encoded, encoded};
  const size_t single[] = {raw};

  expectWorkspace(allToAll, 4, 4, 0, 3,
                  cocclPipelineWorkspaceUnified, 5120, 0);
  expectWorkspace(allToAll, 4, 8, 0, 3,
                  cocclPipelineWorkspaceSplit, 4096, 4096);
  expectWorkspace(allGather, 3, 4, -1, 2,
                  cocclPipelineWorkspaceSplit, 5120, 8192);
  expectWorkspace(tie, 2, 4, -1, -1,
                  cocclPipelineWorkspaceUnified, 2048, 0);
  expectWorkspace(single, 1, 4, -1, -1,
                  cocclPipelineWorkspaceUnified, 4096, 0);

  cocclPipelinePlan overflow = {};
  overflow.tempCount = 2;
  overflow.inputStagingTemp = -1;
  overflow.outputStagingTemp = -1;
  overflow.temps[0].alignedBytes = SIZE_MAX - 255;
  overflow.temps[1].alignedBytes = 256;
  EXPECT(cocclPlanPipelineWorkspace(&overflow, 1) == ncclInvalidArgument);
}

cocclPipelineSpec makeSpec(const char* name, ncclComm_t comm,
                           const cocclPipelineStage* stages, int stageCount,
                           size_t totalInputBytes, size_t inputChunks,
                           cocclOperation operation) {
  return {
      name,
      reinterpret_cast<const void*>(0x100000000ULL),
      reinterpret_cast<void*>(0x400000000ULL),
      totalInputBytes / inputChunks / sizeof(float),
      inputChunks,
      ncclFloat32,
      comm,
      nullptr,
      stages,
      stageCount,
  };
}

void checkQueryShapes() {
  ncclComm comm = {};
  comm.nRanks = 4;
  constexpr size_t inputBytes = 4096;

  const cocclPipelineStage allToAllStages[] = {
      cocclPipelineCompress(reinterpret_cast<void*>(0x1)), cocclPipelineAllToAll(&comm),
      cocclPipelineDecompress()};
  cocclPipelineSpec allToAll = makeSpec(
      "alltoall", &comm, allToAllStages, 3, inputBytes, 4,
      cocclOperation::AllToAll);
  cocclM11ConfigureSizeQueryStub(1, 4, 1, 1, false);
  cocclPipelineContext context = {};
  EXPECT(cocclPreparePipeline(&allToAll, 4, &context) == ncclSuccess);
  const cocclM11SizeQueryObservation& compress =
      cocclM11CompressQueryObservation();
  EXPECT(compress.calls == 1 && compress.elements == 256);
  EXPECT(compress.chunks == 4 && compress.datatype == ncclFloat32);
  EXPECT(context.plan.stageOutputCapacityBytes[0] == 256);
  EXPECT(context.plan.workspaceKind == cocclPipelineWorkspaceUnified);
  EXPECT(context.plan.totalBytes == 5120);

  const cocclPipelineStage allReduceStages[] = {
      cocclPipelineCompress(reinterpret_cast<void*>(0x1)), cocclPipelineAllToAll(&comm),
      cocclPipelineDecompReduceComp(4, reinterpret_cast<void*>(0x1)),
      cocclPipelineAllGather(&comm), cocclPipelineDecompress()};
  cocclPipelineSpec allReduce = makeSpec(
      "allreduce-twoshot", &comm, allReduceStages, 5,
      inputBytes, 4, cocclOperation::AllReduce);
  cocclM11ConfigureSizeQueryStub(1, 4, 1, 2, true);
  context = {};
  EXPECT(cocclPreparePipeline(&allReduce, 4, &context) == ncclSuccess);
  const cocclM11SizeQueryObservation& drc =
      cocclM11DrcQueryObservation();
  EXPECT(drc.calls == 1 && drc.elements == 64 && drc.chunks == 1);
  EXPECT(drc.datatype == ncclFloat32);
  EXPECT(cocclM11CompressQueryObservation().calls == 2);
  EXPECT(context.plan.stageOutputCapacityBytes[2] == 128);
}

void dumpPlan(const char* operation, size_t benchmarkBytes,
              cocclPipelineSpec* spec, int depth) {
  cocclPipelineContext context = {};
  EXPECT(cocclPreparePipeline(spec, depth, &context) == ncclSuccess);
  std::printf("%s,%zu,%d,%d,%s,%zu,%zu,%zu,%zu\n", operation,
              benchmarkBytes, depth, context.depth,
              context.plan.workspaceKind == cocclPipelineWorkspaceUnified
                  ? "unified" : "split",
              context.plan.registeredSliceBytes,
              context.plan.registeredBytes, context.plan.rawBytes,
              context.plan.totalBytes);
}

void dumpPlans() {
  ncclComm comm = {};
  comm.nRanks = 4;
  const cocclPipelineStage allToAllStages[] = {
      cocclPipelineCompress(reinterpret_cast<void*>(0x1)), cocclPipelineAllToAll(&comm),
      cocclPipelineDecompress()};
  const cocclPipelineStage allGatherStages[] = {
      cocclPipelineCompress(reinterpret_cast<void*>(0x1)), cocclPipelineAllGather(&comm),
      cocclPipelineDecompress()};
  const cocclPipelineStage reduceScatterStages[] = {
      cocclPipelineCompress(reinterpret_cast<void*>(0x1)), cocclPipelineAllToAll(&comm),
      cocclPipelineDecompressReduce(4)};
  const cocclPipelineStage allReduceStages[] = {
      cocclPipelineCompress(reinterpret_cast<void*>(0x1)), cocclPipelineAllToAll(&comm),
      cocclPipelineDecompReduceComp(4, reinterpret_cast<void*>(0x1)),
      cocclPipelineAllGather(&comm), cocclPipelineDecompress()};
  const size_t sizes[] = {64ULL << 20, 1ULL << 30, 8ULL << 30};
  const int depths[] = {1, 2, 4, 8};

  cocclM11ConfigureSizeQueryStub(1, 4, 1, 4, true);
  std::printf("operation,input_bytes,requested_depth,effective_depth,"
              "layout,registered_slice_bytes,registered_bytes,raw_bytes,"
              "total_bytes\n");
  for (size_t bytes : sizes) {
    for (int depth : depths) {
      cocclPipelineSpec allToAll = makeSpec(
          "alltoall", &comm, allToAllStages, 3, bytes, 4,
          cocclOperation::AllToAll);
      cocclPipelineSpec allGather = makeSpec(
          "allgather", &comm, allGatherStages, 3, bytes / 4, 1,
          cocclOperation::AllGather);
      cocclPipelineSpec reduceScatter = makeSpec(
          "reducescatter-oneshot", &comm, reduceScatterStages, 3,
          bytes * 4, 4, cocclOperation::ReduceScatter);
      cocclPipelineSpec allReduce = makeSpec(
          "allreduce-twoshot", &comm, allReduceStages, 5,
          bytes, 4, cocclOperation::AllReduce);
      dumpPlan("alltoall", bytes, &allToAll, depth);
      dumpPlan("allgather", bytes, &allGather, depth);
      dumpPlan("reducescatter-oneshot", bytes, &reduceScatter, depth);
      dumpPlan("allreduce-twoshot", bytes, &allReduce, depth);
    }
  }
}

}  // namespace

thread_local int ncclDebugNoWarn = 0;
void ncclDebugLog(ncclDebugLogLevel, unsigned long, const char*, int,
                  const char*, ...) {}

int main(int argc, char** argv) {
  if (argc == 2 && std::strcmp(argv[1], "--csv") == 0) {
    dumpPlans();
    return 0;
  }
  checkWorkspaceSelection();
  checkQueryShapes();
  std::printf("coccl M11 workspace: PASS\n");
  return 0;
}
