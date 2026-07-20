#ifndef COCCL_PIPELINE_LAYOUT_H_
#define COCCL_PIPELINE_LAYOUT_H_

#include <stddef.h>

#include "nccl.h"

// Packs one slice from chunk-major rows into a contiguous pipeline edge.
// source points at the slice offset in row zero; sourcePitchBytes advances to
// the same slice in the next logical chunk.
ncclResult_t cocclLaunchPackSlice(const void* source,
                                  size_t sourcePitchBytes, void* destination,
                                  size_t sliceBytes, size_t chunkCount,
                                  cudaStream_t stream);

// Scatters one contiguous pipeline edge back to the same slice in each
// chunk-major output row.
ncclResult_t cocclLaunchUnpackSlice(const void* source, void* destination,
                                    size_t destinationPitchBytes,
                                    size_t sliceBytes, size_t chunkCount,
                                    cudaStream_t stream);

#endif
