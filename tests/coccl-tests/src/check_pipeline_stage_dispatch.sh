#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/../../.." && pwd)
source_file="$root/src/coccl-extend/primitives/coccl_pipeline.cc"

for symbol in \
  cocclRunCompressStage \
  cocclRunAllToAllStage \
  cocclRunAllGatherStage \
  cocclRunDecompReduceCompStage \
  cocclRunDecompressReduceStage \
  cocclRunDecompressStage \
  cocclPipelineStageFn \
  cocclPipelineStageHandlers; do
  rg -q "$symbol" "$source_file" || {
    echo "missing pipeline dispatch symbol: $symbol" >&2
    exit 1
  }
done

run_stage_body=$(sed -n \
  '/ncclResult_t cocclRunPipelineStage(/,/^}/p' "$source_file")
if grep -q 'switch (stage.kind)' <<<"$run_stage_body"; then
  echo "cocclRunPipelineStage still switches on stage.kind" >&2
  exit 1
fi
