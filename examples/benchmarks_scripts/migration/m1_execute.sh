#!/usr/bin/env bash
set -euo pipefail

root=/data/home/scyb672/run/lxc/COCCL-migrate
original=/data/home/scyb672/run/lxc/COCCL
runner="$root/examples/benchmarks_scripts/migration/m1_matrix.sh"
result_root="$root/results/M1"
status_file="$result_root/execution.status"

mkdir -p "$result_root"
for mode in performance memory; do
  for variant in original migrate; do
    source_root=$root
    [[ "$variant" != original ]] || source_root=$original
    printf '%s %s-%s-start\n' "$(date -Is)" "$variant" "$mode" >>"$status_file"
    bash "$runner" "$variant" "$source_root" "$mode"
    printf '%s %s-%s-complete\n' "$(date -Is)" "$variant" "$mode" >>"$status_file"
  done
done

python3 "$root/examples/benchmarks_scripts/migration/m1_report.py" "$result_root"
printf '%s complete\n' "$(date -Is)" >>"$status_file"
