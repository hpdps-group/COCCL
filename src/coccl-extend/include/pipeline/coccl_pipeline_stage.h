#ifndef COCCL_PIPELINE_STAGE_H_
#define COCCL_PIPELINE_STAGE_H_

#include "pipeline/coccl_pipeline.h"
#include "compression/coccl_compressor.h"

// Minimal execution state shared by every pipeline stage. Workspace planning,
// slicing, and stream/event scheduling remain private to coccl_pipeline.cc.
struct cocclPipelineStageContext {
  size_t rawSliceCount;
  ncclDataType_t rawDatatype;
  ncclComm_t ownerComm;
  cocclCompressorHandle compressor;
};

// Runtime metadata for the linear edge passed from one stage to the next.
// bytes, totalElements, and datatype describe the current physical encoding;
// logicalChunks counts equally sized, independently communicable encodings.
// A reduction stage may group several of them into each reduction operand;
// that operand count is carried separately by stage.reduceChunks.
struct cocclPipelineEdge {
  void* ptr;
  size_t bytes;
  size_t totalElements;
  ncclDataType_t datatype;
  size_t logicalChunks;
};

// Shape propagation is part of stage semantics and is shared by the workspace
// planner and executor.
ncclResult_t cocclPipelineStageOutputChunks(
    const cocclPipelineStage& stage, size_t inputChunks,
    size_t* outputChunks);

// Dispatches one common Compress/Collective/Decompress stage and updates edge
// metadata. Stream ordering is owned by coccl_pipeline.cc.
ncclResult_t cocclExecutePipelineStage(
    const cocclPipelineStageContext* context,
    const cocclPipelineStage* stage, cocclPipelineEdge* edge,
    void* outputPtr, size_t outputCapacityBytes, cudaStream_t stream);

#endif
