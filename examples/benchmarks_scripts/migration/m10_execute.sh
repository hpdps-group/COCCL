#!/usr/bin/env bash
set -euo pipefail

source_root=${M10_SOURCE_ROOT:-/data/home/scyb672/run/lxc/COCCL-migrate}
result_root=${M10_RESULT_ROOT:-/data/home/scyb672/run/lxc/COCCL-migrate/results/M10}
build_root=${M10_BUILD_ROOT:-$source_root/build}
host_test_dir=${M10_HOST_TEST_DIR:-/data/home/scyb672/run/codex-tmp/coccl-m10-host}
tests=${M10_TESTS_DIR:-$source_root/tests/coccl-tests/build}
script_root="$source_root/examples/benchmarks_scripts/migration"
status_file="$result_root/execution.status"

export CUDA_HOME=${M10_CUDA_HOME:-/data/apps/cuda/12.4}
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$build_root/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

mkdir -p "$result_root"
"$host_test_dir/coccl_m10_unified_arena_test" \
  >"$result_root/unified-host.txt"
"$tests/coccl_m5_pipeline_layout_test" \
  >"$result_root/layout-correctness.txt"

printf '%s layout-start\n' "$(date -Is)" >>"$status_file"
bash "$script_root/m10_matrix.sh" "$source_root" layout layout
printf '%s layout-complete\n' "$(date -Is)" >>"$status_file"

for compressor in sdp4bit zfp; do
  printf '%s smoke-%s-start\n' "$(date -Is)" "$compressor" >>"$status_file"
  bash "$script_root/m10_matrix.sh" "$source_root" "$compressor" smoke
  printf '%s smoke-%s-complete\n' "$(date -Is)" "$compressor" >>"$status_file"
done

for compressor in sdp4bit zfp; do
  printf '%s performance-%s-start\n' "$(date -Is)" "$compressor" >>"$status_file"
  bash "$script_root/m10_matrix.sh" "$source_root" "$compressor" performance
  printf '%s performance-%s-complete\n' "$(date -Is)" "$compressor" >>"$status_file"
done

for compressor in sdp4bit zfp; do
  printf '%s memory-%s-start\n' "$(date -Is)" "$compressor" >>"$status_file"
  bash "$script_root/m10_matrix.sh" "$source_root" "$compressor" memory
  printf '%s memory-%s-complete\n' "$(date -Is)" "$compressor" >>"$status_file"
done

python3 "$script_root/m10_report.py" "$result_root" \
  /data/home/scyb672/run/lxc/COCCL-migrate/results
printf '%s complete\n' "$(date -Is)" >>"$status_file"
