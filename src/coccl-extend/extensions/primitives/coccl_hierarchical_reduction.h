#ifndef COCCL_HIERARCHICAL_REDUCTION_H_
#define COCCL_HIERARCHICAL_REDUCTION_H_

#include "core/pipeline/coccl_pipeline.h"

struct cocclPreparedCall;

// Builds the two reduction phases shared by ReduceScatter TwoShot and
// AllReduce TripleShot. finalCompressor is null when the caller needs raw
// output, or the encoder required by the following collective.
int cocclBuildHierarchicalReduction(
    const cocclPreparedCall* prepared, ncclComm_t intraComm,
    ncclComm_t interComm, void* finalCompressor,
    cocclPipelineStage* stages);

#endif
