#include "pipeline/coccl_pipeline_internal.h"

#include <stdio.h>

namespace {

bool rangesOverlap(size_t firstOffset, size_t firstBytes,
                   size_t secondOffset, size_t secondBytes) {
  return firstOffset < secondOffset + secondBytes &&
      secondOffset < firstOffset + firstBytes;
}

int validateLayout(const char* name, const cocclPipelinePlan& plan) {
  for (int i = 0; i < plan.tempCount; ++i) {
    const auto& temp = plan.temps[i];
    if (temp.offset % kPipelineBufferAlignment != 0 ||
        temp.bytes % kPipelineBufferAlignment != 0) {
      fprintf(stderr, "%s: temp %d is not aligned\n", name, i);
      return 1;
    }
    if (temp.logicalBytes > temp.bytes ||
        (temp.frameMetadataBytes != 0 &&
         (temp.frameMetadataOffset < temp.payloadBytes ||
          temp.frameMetadataOffset + temp.frameMetadataBytes >
              temp.logicalBytes))) {
      fprintf(stderr, "%s: temp %d has an invalid framed layout\n", name, i);
      return 1;
    }
    const size_t arenaBytes = temp.storage == cocclPipelineRawRing
        ? plan.rawBytes : plan.registeredSliceBytes;
    const size_t spanBytes = temp.storage == cocclPipelineRawRing
        ? (size_t)kPipelineRawRingSlots * temp.bytes : temp.bytes;
    if (temp.offset > arenaBytes || spanBytes > arenaBytes - temp.offset) {
      fprintf(stderr, "%s: temp %d exceeds its arena\n", name, i);
      return 1;
    }
    if (temp.storage == cocclPipelineRawRing) {
      const size_t slot0 = temp.offset;
      const size_t slot1 = temp.offset + temp.bytes;
      const size_t slot2 = temp.offset;
      if (slot0 == slot1 || slot0 != slot2) {
        fprintf(stderr, "%s: temp %d has an invalid raw ring\n", name, i);
        return 1;
      }
    }
  }

  for (int i = 1; i < plan.tempCount; ++i) {
    const auto& previous = plan.temps[i - 1];
    const auto& current = plan.temps[i];
    if (previous.storage == current.storage &&
        rangesOverlap(previous.offset, previous.bytes,
                      current.offset, current.bytes)) {
      fprintf(stderr, "%s: adjacent temps %d and %d overlap\n",
              name, i - 1, i);
      return 1;
    }
  }
  return 0;
}

int expectWorkspace(const char* name, const size_t* buffers, int bufferCount,
                    int depth, int inputStaging, int outputStaging,
                    cocclPipelineWorkspaceKind expectedKind,
                    size_t expectedRegistered, size_t expectedRaw) {
  cocclPipelinePlan plan = {};
  plan.tempCount = bufferCount;
  plan.inputStagingTemp = inputStaging;
  plan.outputStagingTemp = outputStaging;
  for (int i = 0; i < kMaxPipelineStages; ++i) {
    plan.stageOutputTemp[i] = -1;
  }
  for (int i = 0; i < bufferCount; ++i) {
    plan.temps[i].logicalBytes = buffers[i];
    plan.temps[i].bytes = buffers[i];
    plan.temps[i].payloadBytes = buffers[i];
  }

  const ncclResult_t result = cocclPlanPipelineWorkspace(&plan, depth);
  const size_t expectedTotal = expectedRegistered + expectedRaw;
  if (result != ncclSuccess || plan.workspaceKind != expectedKind ||
      plan.registeredBytes != expectedRegistered ||
      plan.rawBytes != expectedRaw || plan.totalBytes != expectedTotal) {
    fprintf(stderr,
            "%s: expected result=%d kind=%d registered=%zu raw=%zu total=%zu, "
            "got result=%d kind=%d registered=%zu raw=%zu total=%zu\n",
            name, (int)ncclSuccess, (int)expectedKind, expectedRegistered,
            expectedRaw, expectedTotal, (int)result, (int)plan.workspaceKind,
            plan.registeredBytes, plan.rawBytes, plan.totalBytes);
    return 1;
  }
  return validateLayout(name, plan);
}

int testFramedWorkspace() {
  constexpr size_t rawBytes = 1024;
  constexpr size_t frameStrideBytes = 256;
  constexpr size_t metadataBytes =
      4 * sizeof(cocclCompressorFrameMetadata);
  constexpr size_t framedBytes = 1280;

  cocclPipelinePlan plan = {};
  plan.tempCount = 4;
  plan.inputStagingTemp = 0;
  plan.outputStagingTemp = 3;
  for (int i = 0; i < kMaxPipelineStages; ++i) {
    plan.stageOutputTemp[i] = -1;
  }
  plan.temps[0] = {0, rawBytes, rawBytes, rawBytes, 0, 0, 0,
                   cocclPipelineRegisteredArena};
  plan.temps[1] = {0, rawBytes + metadataBytes, framedBytes, rawBytes,
                   rawBytes, metadataBytes, frameStrideBytes,
                   cocclPipelineRegisteredArena};
  plan.temps[2] = plan.temps[1];
  plan.temps[3] = plan.temps[0];

  if (cocclPlanPipelineWorkspace(&plan, 4) != ncclSuccess ||
      plan.workspaceKind != cocclPipelineWorkspaceUnified ||
      plan.registeredSliceBytes != 2 * framedBytes ||
      plan.registeredBytes != 8 * framedBytes || plan.rawBytes != 0 ||
      validateLayout("framed workspace", plan) != 0) {
    fprintf(stderr, "framed workspace planning failed\n");
    return 1;
  }

  return 0;
}

int testWorkspaceSelection() {
  constexpr size_t raw = 1024;
  constexpr size_t encoded = 256;

  const size_t allToAll[] = {raw, encoded, encoded, raw};
  const size_t allGather[] = {encoded, 4 * encoded, 4 * raw};
  const size_t reduceScatterOneShot[] = {raw, encoded, encoded};
  const size_t reduceScatterTwoShot[] = {
      raw, encoded, encoded, encoded, encoded};
  const size_t allReduceOneShot[] = {encoded, 4 * encoded};
  const size_t allReduceTwoShot[] = {
      raw, encoded, encoded, encoded, encoded, raw};
  const size_t allReduceTripleShot[] = {
      raw, encoded, encoded, encoded, encoded, encoded, encoded, raw};
  const size_t unknownEstimate[] = {raw, raw, raw, raw};
  const size_t tie[] = {encoded, encoded};

  return
      expectWorkspace("AllToAll Unified", allToAll, 4, 4, 0, 3,
                      cocclPipelineWorkspaceUnified, 5120, 0) ||
      expectWorkspace("AllToAll Split", allToAll, 4, 8, 0, 3,
                      cocclPipelineWorkspaceSplit, 4096, 4096) ||
      expectWorkspace("AllGather", allGather, 3, 4, -1, 2,
                      cocclPipelineWorkspaceSplit, 5120, 8192) ||
      expectWorkspace("ReduceScatter OneShot", reduceScatterOneShot, 3,
                      4, 0, -1, cocclPipelineWorkspaceSplit, 2048, 2048) ||
      expectWorkspace("ReduceScatter TwoShot", reduceScatterTwoShot, 5,
                      4, 0, -1, cocclPipelineWorkspaceSplit, 2048, 2048) ||
      expectWorkspace("AllReduce OneShot", allReduceOneShot, 2, 1, -1, -1,
                      cocclPipelineWorkspaceUnified, 1280, 0) ||
      expectWorkspace("AllReduce TwoShot", allReduceTwoShot, 6, 4, 0, 5,
                      cocclPipelineWorkspaceUnified, 5120, 0) ||
      expectWorkspace("AllReduce TripleShot", allReduceTripleShot, 8, 4,
                      0, 7, cocclPipelineWorkspaceUnified, 5120, 0) ||
      expectWorkspace("unknown estimate", unknownEstimate, 4, 8, 0, 3,
                      cocclPipelineWorkspaceUnified, 16384, 0) ||
      expectWorkspace("tie keeps Unified", tie, 2, 4, -1, -1,
                      cocclPipelineWorkspaceUnified, 2048, 0);
}

int expectUserBuffers(const char* name, uintptr_t input, size_t inputChunks,
                      uintptr_t output, size_t outputChunks,
                      size_t rawChunkBytes, int rank,
                      cocclPipelineInPlaceLayout inPlaceLayout,
                      ncclResult_t expectedResult, bool expectedSerial) {
  bool requireSerial = !expectedSerial;
  const ncclResult_t result = cocclPipelineUserBuffersRequireSerial(
      reinterpret_cast<const void*>(input), inputChunks,
      reinterpret_cast<void*>(output), outputChunks, rawChunkBytes, rank,
      inPlaceLayout, &requireSerial);
  if (result != expectedResult ||
      (result == ncclSuccess && requireSerial != expectedSerial)) {
    fprintf(stderr,
            "%s: expected result=%d serial=%d, got result=%d serial=%d\n",
            name, (int)expectedResult, (int)expectedSerial, (int)result,
            (int)requireSerial);
    return 1;
  }
  return 0;
}

int testUserBufferLayouts() {
  constexpr uintptr_t base = 0x100000;
  constexpr size_t chunk = 1024;

  return
      expectUserBuffers("disjoint", base, 4, base + 8 * chunk, 4, chunk, 0,
                        cocclPipelineInPlaceNone, ncclSuccess, false) ||
      expectUserBuffers("AllReduce in-place", base, 4, base, 4, chunk, 0,
                        cocclPipelineInPlaceSameBuffer, ncclSuccess, false) ||
      expectUserBuffers("AllGather rank 0", base, 1, base, 4, chunk, 0,
                        cocclPipelineInPlaceInputRankChunk, ncclSuccess,
                        false) ||
      expectUserBuffers("AllGather rank 3", base + 3 * chunk, 1, base, 4,
                        chunk, 3, cocclPipelineInPlaceInputRankChunk,
                        ncclSuccess, false) ||
      expectUserBuffers("ReduceScatter rank 0", base, 4, base, 1, chunk, 0,
                        cocclPipelineInPlaceOutputRankChunk, ncclSuccess,
                        false) ||
      expectUserBuffers("ReduceScatter rank 3", base, 4, base + 3 * chunk, 1,
                        chunk, 3, cocclPipelineInPlaceOutputRankChunk,
                        ncclSuccess, false) ||
      expectUserBuffers("AllGather byte offset", base + 3 * chunk + 1, 1,
                        base, 4, chunk, 3,
                        cocclPipelineInPlaceInputRankChunk, ncclSuccess,
                        true) ||
      expectUserBuffers("AllGather wrong rank chunk", base + 3 * chunk, 1,
                        base, 4, chunk, 2,
                        cocclPipelineInPlaceInputRankChunk, ncclSuccess,
                        true) ||
      expectUserBuffers("partial overlap", base + chunk / 2, 4, base, 4,
                        chunk, 0, cocclPipelineInPlaceSameBuffer, ncclSuccess,
                        true) ||
      expectUserBuffers("AllToAll in-place", base, 4, base, 4, chunk, 0,
                        cocclPipelineInPlaceNone, ncclSuccess, true) ||
      expectUserBuffers("same-buffer shape mismatch", base, 4, base, 1,
                        chunk, 0, cocclPipelineInPlaceSameBuffer, ncclSuccess,
                        true) ||
      expectUserBuffers("size overflow", base, 2, base, 1, SIZE_MAX, 0,
                        cocclPipelineInPlaceSameBuffer, ncclInvalidArgument,
                        false) ||
      expectUserBuffers("address overflow", UINTPTR_MAX - chunk / 2, 1, base,
                        1, chunk, 0, cocclPipelineInPlaceNone,
                        ncclInvalidArgument, false) ||
      expectUserBuffers("invalid rank", base, 1, base, 4, chunk, -1,
                        cocclPipelineInPlaceInputRankChunk,
                        ncclInvalidArgument, false);
}

}  // namespace

int main() {
  if (testWorkspaceSelection() || testFramedWorkspace() ||
      testUserBufferLayouts()) {
    return 1;
  }
  printf("coccl pipeline plan tests passed\n");
  return 0;
}
