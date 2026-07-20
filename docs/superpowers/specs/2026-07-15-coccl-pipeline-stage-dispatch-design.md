# COCCL Pipeline Stage Dispatch Design

## Goal

Replace the operation switch inside `cocclRunPipelineStage` with private,
uniform stage handler functions selected through a static function pointer
table.

## Design

- Keep `cocclPipelineStage`, its builders, and all primitive flow declarations
  unchanged.
- Define one private handler for each stage kind in `coccl_pipeline.cc`:
  Compress, AllToAll, AllGather, DecompReduceComp, DecompressReduce, and
  Decompress.
- Give every handler the same signature. The executor supplies the pipeline
  context, stage description, input edge metadata, planned output pointer,
  output edge metadata, and CUDA stream.
- Index a static handler table by `cocclPipelineStageKind` after validating the
  enum value. `cocclRunPipelineStage` remains responsible only for resolving
  slice input/output, invoking the selected handler, final output scatter, and
  storing the resulting edge metadata.

## Constraints

- Do not change primitive declarations, public NCCL ABI, compressor ABI,
  workspace planning, stream/event ordering, or stage semantics.
- Keep handlers private to the implementation file.
- Preserve existing validation and error propagation.

## Verification

- A source contract checks for all six handlers and the dispatch table, and
  rejects a stage-operation switch inside `cocclRunPipelineStage`.
- Compile the existing private API contract.
- Build NCCL/COCCL and relink the six overlap tests.
