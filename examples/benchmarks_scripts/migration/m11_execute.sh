#!/usr/bin/env bash
set -euo pipefail

source_root=${M11_SOURCE_ROOT:-/data/home/scyb672/run/codex-tmp/coccl-valid-publish-20260816}
result_root=${M11_RESULT_ROOT:-/data/home/scyb672/run/lxc/COCCL-migrate/results/M11}
build_root=${M11_BUILD_ROOT:-/data/home/scyb672/run/lxc/COCCL-migrate/build}
host_test_dir=${M11_HOST_TEST_DIR:-/data/home/scyb672/run/codex-tmp/coccl-m11-host}
tests=${M11_TESTS_DIR:-/data/home/scyb672/run/lxc/COCCL-migrate/tests/coccl-tests/build}
script_root="$source_root/examples/benchmarks_scripts/migration"
status_file="$result_root/execution.status"

export CUDA_HOME=${M11_CUDA_HOME:-/data/apps/cuda/12.4}
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$build_root/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

mkdir -p "$result_root"
"$host_test_dir/coccl_m11_size_query_test" >"$result_root/host-size-query.txt"
"$host_test_dir/coccl_m11_workspace_test" >"$result_root/host-workspace.txt"
"$tests/coccl_m5_pipeline_layout_test" >"$result_root/layout-correctness.txt"

for compressor in sdp4bit zfp; do
  printf '%s smoke-%s-start\n' "$(date -Is)" "$compressor" >>"$status_file"
  bash "$script_root/m11_matrix.sh" "$source_root" "$compressor" smoke
  printf '%s smoke-%s-complete\n' "$(date -Is)" "$compressor" >>"$status_file"
done
for compressor in sdp4bit zfp; do
  printf '%s performance-%s-start\n' "$(date -Is)" "$compressor" >>"$status_file"
  bash "$script_root/m11_matrix.sh" "$source_root" "$compressor" performance
  printf '%s performance-%s-complete\n' "$(date -Is)" "$compressor" >>"$status_file"
done
for compressor in sdp4bit zfp; do
  printf '%s memory-%s-start\n' "$(date -Is)" "$compressor" >>"$status_file"
  bash "$script_root/m11_matrix.sh" "$source_root" "$compressor" memory
  printf '%s memory-%s-complete\n' "$(date -Is)" "$compressor" >>"$status_file"
done

python3 "$script_root/m11_report.py" "$result_root" \
  /data/home/scyb672/run/lxc/COCCL-migrate/results
printf '%s complete\n' "$(date -Is)" >>"$status_file"
