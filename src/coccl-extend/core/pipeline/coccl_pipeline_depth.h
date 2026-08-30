#ifndef COCCL_PIPELINE_DEPTH_H_
#define COCCL_PIPELINE_DEPTH_H_

#include "nccl.h"

#include <initializer_list>
#include <stddef.h>

struct cocclPipelineSpec;

constexpr size_t kCocclPipelineTargetSliceBytes = size_t{32} << 20;

inline int cocclChoosePipelineDepthForBytes(size_t bytes, int maxDepth = 8) {
  int selected = 1;
  for (int depth : {2, 4, 8}) {
    if (depth <= maxDepth &&
        bytes / (size_t)depth >= kCocclPipelineTargetSliceBytes) {
      selected = depth;
    }
  }
  return selected;
}

int cocclAutotunePipelineDepth(const cocclPipelineSpec* spec);
void cocclPipelineDepthCommDestroy(ncclComm_t comm);

#endif
