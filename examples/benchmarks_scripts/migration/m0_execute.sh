#!/usr/bin/env bash
set -euo pipefail

root=/data/home/scyb672/run/lxc/COCCL-migrate
runner="$root/examples/benchmarks_scripts/migration/m0_matrix.sh"
result_root="$root/results/M0"
status_file="$result_root/execution.status"

mkdir -p "$result_root"
printf '%s original-full-start\n' "$(date -Is)" >>"$status_file"
bash "$runner" original /data/home/scyb672/run/lxc/COCCL full
printf '%s original-full-complete\n' "$(date -Is)" >>"$status_file"

printf '%s migrate-full-start\n' "$(date -Is)" >>"$status_file"
bash "$runner" migrate-copy "$root" full
printf '%s migrate-full-complete\n' "$(date -Is)" >>"$status_file"

printf '%s original-memory-start\n' "$(date -Is)" >>"$status_file"
bash "$runner" original /data/home/scyb672/run/lxc/COCCL memory
printf '%s original-memory-complete\n' "$(date -Is)" >>"$status_file"

printf '%s migrate-memory-start\n' "$(date -Is)" >>"$status_file"
bash "$runner" migrate-copy "$root" memory
printf '%s migrate-memory-complete\n' "$(date -Is)" >>"$status_file"

python3 "$root/examples/benchmarks_scripts/migration/m0_parse.py" "$result_root"
python3 "$root/examples/benchmarks_scripts/migration/m0_report.py" "$result_root"

printf '%s performance-confirmation-start\n' "$(date -Is)" >>"$status_file"
bash "$root/examples/benchmarks_scripts/migration/m0_confirm_performance.sh" performance
printf '%s performance-confirmation-complete\n' "$(date -Is)" >>"$status_file"

printf '%s memory-confirmation-start\n' "$(date -Is)" >>"$status_file"
bash "$root/examples/benchmarks_scripts/migration/m0_confirm_performance.sh" memory
printf '%s memory-confirmation-complete\n' "$(date -Is)" >>"$status_file"

python3 "$root/examples/benchmarks_scripts/migration/m0_report.py" "$result_root"
printf '%s complete\n' "$(date -Is)" >>"$status_file"
