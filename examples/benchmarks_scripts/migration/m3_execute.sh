#!/usr/bin/env bash
set -euo pipefail

source_root=${M3_SOURCE_ROOT:-/data/home/scyb672/run/lxc/COCCL-migrate}
result_root=${M3_RESULT_ROOT:-/data/home/scyb672/run/lxc/COCCL-migrate/results/M3}
host_test_dir=${M3_HOST_TEST_DIR:-/data/home/scyb672/run/codex-tmp/coccl-m3-host}
script_root="$source_root/examples/benchmarks_scripts/migration"
status_file="$result_root/execution.status"

mkdir -p "$result_root"
"$host_test_dir/coccl_m3_buffer_test" >"$result_root/host-buffer.txt"

for mode in performance memory lifecycle; do
  printf '%s %s-start\n' "$(date -Is)" "$mode" >>"$status_file"
  bash "$script_root/m3_matrix.sh" "$source_root" "$mode"
  printf '%s %s-complete\n' "$(date -Is)" "$mode" >>"$status_file"
done

python3 "$script_root/m3_report.py" "$result_root" \
  /data/home/scyb672/run/lxc/COCCL-migrate/results/M2
printf '%s complete\n' "$(date -Is)" >>"$status_file"
