#!/usr/bin/env bash
set -euo pipefail

source_root=${M4_SOURCE_ROOT:-/data/home/scyb672/run/lxc/COCCL-migrate}
result_root=${M4_RESULT_ROOT:-/data/home/scyb672/run/lxc/COCCL-migrate/results/M4}
host_test_dir=${M4_HOST_TEST_DIR:-/data/home/scyb672/run/codex-tmp/coccl-m4-host}
script_root="$source_root/examples/benchmarks_scripts/migration"
status_file="$result_root/execution.status"
control_result="$result_root/m3-control"

mkdir -p "$result_root"
"$host_test_dir/coccl_m4_vmm_test" >"$result_root/host-vmm.txt"

for mode in smoke performance memory lifecycle; do
  printf '%s %s-start\n' "$(date -Is)" "$mode" >>"$status_file"
  bash "$script_root/m4_matrix.sh" "$source_root" "$mode"
  printf '%s %s-complete\n' "$(date -Is)" "$mode" >>"$status_file"
done

if [[ -n ${M4_M3_CONTROL_SOURCE_ROOT:-} ]]; then
  env NCCL_CUMEM_ENABLE=1 \
    M3_RESULT_ROOT="$control_result" \
    M3_BUILD_ROOT="${M4_M3_CONTROL_BUILD_ROOT:?}" \
    M3_TESTS_DIR="${M4_TESTS_DIR:?}" \
    bash "$M4_M3_CONTROL_SOURCE_ROOT/examples/benchmarks_scripts/migration/m3_matrix.sh" \
      "$M4_M3_CONTROL_SOURCE_ROOT" performance
fi

python3 "$script_root/m4_report.py" "$result_root" \
  /data/home/scyb672/run/lxc/COCCL-migrate/results/M3 "$control_result"
printf '%s complete\n' "$(date -Is)" >>"$status_file"
