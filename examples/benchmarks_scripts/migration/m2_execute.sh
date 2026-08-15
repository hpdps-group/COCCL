#!/usr/bin/env bash
set -euo pipefail

source_root=${M2_SOURCE_ROOT:-/data/home/scyb672/run/lxc/COCCL-migrate}
result_root=${M2_RESULT_ROOT:-/data/home/scyb672/run/lxc/COCCL-migrate/results/M2}
host_test_dir=${M2_HOST_TEST_DIR:-/data/home/scyb672/run/codex-tmp/coccl-m2-host}
script_root="$source_root/examples/benchmarks_scripts/migration"
status_file="$result_root/execution.status"

mkdir -p "$result_root"
"$host_test_dir/coccl_m2_runtime_test" >"$result_root/host-overhead.txt"
"$host_test_dir/coccl_m2_group_test"

for mode in performance memory; do
  printf '%s %s-start\n' "$(date -Is)" "$mode" >>"$status_file"
  bash "$script_root/m2_matrix.sh" "$source_root" "$mode"
  printf '%s %s-complete\n' "$(date -Is)" "$mode" >>"$status_file"
done

python3 "$script_root/m2_report.py" "$result_root" \
  /data/home/scyb672/run/lxc/COCCL-migrate/results/M0 \
  /data/home/scyb672/run/lxc/COCCL-migrate/results/M1
printf '%s complete\n' "$(date -Is)" >>"$status_file"
