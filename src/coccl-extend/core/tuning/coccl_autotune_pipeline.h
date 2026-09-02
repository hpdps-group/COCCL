#ifndef COCCL_AUTOTUNE_PIPELINE_H_
#define COCCL_AUTOTUNE_PIPELINE_H_

#include "nccl.h"

#include <algorithm>
#include <stddef.h>

struct cocclPipelineSpec;

constexpr size_t kCocclAutotuneSliceStepBytes = size_t{16} << 20;
constexpr int kCocclAutotuneMaxPipelineDepth = 16;

struct cocclPipelineTuningDecision {
  size_t targetSliceBytes;
  int maxDepth;
  double predictedTimeUs;
};

inline int cocclPipelineDepthForSlice(size_t totalBytes,
                                      size_t targetSliceBytes,
                                      int maxDepth =
                                          kCocclAutotuneMaxPipelineDepth) {
  const size_t slices =
      (totalBytes + targetSliceBytes - 1) / targetSliceBytes;
  return (int)std::min<size_t>((size_t)maxDepth, std::max<size_t>(1, slices));
}

cocclPipelineTuningDecision cocclAutotunePipelineLayout(
    const cocclPipelineSpec* spec);
void cocclAutotunePipelineCommDestroy(ncclComm_t comm);

#endif
