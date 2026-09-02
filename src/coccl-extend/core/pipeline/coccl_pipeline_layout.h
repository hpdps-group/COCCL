#ifndef COCCL_PIPELINE_LAYOUT_H_
#define COCCL_PIPELINE_LAYOUT_H_

#include <stddef.h>

#include "core/pipeline/coccl_pipeline.h"
#include "nccl.h"

// Packs one slice from rank-major pitched chunks into one contiguous edge.
ncclResult_t cocclLaunchPackSlice(const void* source,
                                  size_t sourcePitchBytes, void* destination,
                                  size_t sliceBytes, size_t chunkCount,
                                  cocclPipelineInputLayout inputLayout,
                                  int nNodes, int ranksPerNode,
                                  cudaStream_t stream);

// Scatters one contiguous edge back into the same slice of each rank chunk.
ncclResult_t cocclLaunchUnpackSlice(const void* source, void* destination,
                                    size_t destinationPitchBytes,
                                    size_t sliceBytes, size_t chunkCount,
                                    cocclPipelineOutputLayout outputLayout,
                                    int nNodes, int ranksPerNode,
                                    cudaStream_t stream);

#endif
