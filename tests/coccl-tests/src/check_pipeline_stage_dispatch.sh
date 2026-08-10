#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/../../.." && pwd)
executor_file="$root/src/coccl-extend/primitives/coccl_pipeline.cc"
stage_file="$root/src/coccl-extend/primitives/coccl_pipeline_stage.cc"

for symbol in \
  cocclRunCompressStage \
  cocclRunAllToAllStage \
  cocclRunAllGatherStage \
  cocclRunDecompReduceCompStage \
  cocclRunDecompressReduceStage \
  cocclRunDecompressStage \
  cocclPipelineStageFn \
  cocclPipelineStageHandlers; do
  rg -q "$symbol" "$stage_file" || {
    echo "missing pipeline dispatch symbol: $symbol" >&2
    exit 1
  }
done

run_stage_body=$(sed -n \
  '/ncclResult_t cocclExecutePipelineStage(/,/^}/p' "$stage_file")
if grep -q 'switch (stage.kind)' <<<"$run_stage_body"; then
  echo "cocclExecutePipelineStage still switches on stage.kind" >&2
  exit 1
fi

rg -q 'cocclExecutePipelineStage' "$executor_file" || {
  echo "pipeline executor does not call the shared stage dispatcher" >&2
  exit 1
}
