#!/usr/bin/env bash
set -euo pipefail

source_root=${M5_SOURCE_ROOT:-/data/home/scyb672/run/lxc/COCCL-migrate}
result_root=${M5_RESULT_ROOT:-/data/home/scyb672/run/lxc/COCCL-migrate/results/M5}
build_root=${M5_BUILD_ROOT:-$source_root/build}
host_test_dir=${M5_HOST_TEST_DIR:-/data/home/scyb672/run/codex-tmp/coccl-m5-host}
tests=${M5_TESTS_DIR:-$source_root/tests/coccl-tests/build}
script_root="$source_root/examples/benchmarks_scripts/migration"
status_file="$result_root/execution.status"

export CUDA_HOME=${CUDA_HOME:-/data/apps/cuda/12.4}
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$build_root/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

mkdir -p "$result_root"
"$host_test_dir/coccl_m5_pipeline_plan_test" >"$result_root/host-plan.txt"
"$host_test_dir/coccl_m5_pipeline_stage_test" >"$result_root/host-stage.txt"
"$host_test_dir/coccl_m5_pipeline_plan_test" --csv \
  >"$result_root/workspace-plan.csv"
"$tests/coccl_m5_pipeline_layout_test" \
  >"$result_root/layout-correctness.txt"

for mode in smoke performance layout memory; do
  printf '%s %s-start\n' "$(date -Is)" "$mode" >>"$status_file"
  bash "$script_root/m5_matrix.sh" "$source_root" "$mode"
  printf '%s %s-complete\n' "$(date -Is)" "$mode" >>"$status_file"
done

python3 "$script_root/m5_report.py" "$result_root" \
  /data/home/scyb672/run/lxc/COCCL-migrate/results/M0
printf '%s complete\n' "$(date -Is)" >>"$status_file"
