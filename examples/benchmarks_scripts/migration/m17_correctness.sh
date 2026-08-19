#!/usr/bin/env bash
set -euo pipefail

source_root=${1:?usage: m17_correctness.sh SOURCE_ROOT}
hostfile=${M17_HOSTFILE:?M17_HOSTFILE must name the two-node hostfile}
current_root=${M17_CURRENT_ROOT:-/data/home/scyb672/run/lxc/COCCL-migrate}
baseline_root=${M17_BASELINE_ROOT:-/data/home/scyb672/run/lxc/COCCL}
temp_root=${M17_TEMP_ROOT:-/data/home/scyb672/run/codex-tmp/coccl-m17-hierarchical}
result_root=${M17_RESULT_ROOT:-$current_root/results/M17}
correctness_root="$temp_root/correctness"
failures=0

run_stem() {
  local implementation=$1
  local compressor=$2
  local depth=$3
  local datatype=$4
  local stem="2x4__hierarchical__${compressor}__${datatype}__d${depth}"
  local log="$result_root/raw/$implementation/hierarchical/$stem.log"
  local marker="$result_root/raw/$implementation/hierarchical/$stem.ok"

  if [[ -n ${M17_ONLY_IMPLEMENTATION:-} &&
        "$implementation" != "$M17_ONLY_IMPLEMENTATION" ]]; then
    return
  fi
  if [[ -n ${M17_ONLY_COMPRESSOR:-} &&
        "$compressor" != "$M17_ONLY_COMPRESSOR" ]]; then
    return
  fi

  if M16_HOSTFILE="$hostfile" M16_RANKS=8 M16_TOPOLOGY=2x4 \
      M16_CURRENT_ROOT="$current_root" M16_BASELINE_ROOT="$baseline_root" \
      M16_TEMP_ROOT="$correctness_root" M16_RESULT_ROOT="$result_root" \
      M16_ONLY_STEM="$stem" M16_FORCE=${M17_FORCE:-0} \
      bash "$source_root/examples/benchmarks_scripts/migration/m16_matrix.sh" \
        "$source_root" "$implementation" hierarchical; then
    return
  fi

  # Initial COCCL can fail during teardown after both complete results exist.
  if [[ "$implementation" == baseline ]] &&
      grep -q 'operation=reducescatter algorithm=twoshot' "$log" &&
      grep -q 'operation=allreduce algorithm=tripleshot' "$log"; then
    printf 'teardown failure accepted after complete output\n' \
      >"${log%.log}.teardown"
    touch "$marker"
    return
  fi
  failures=$((failures + 1))
}

for compressor in sdp4bit zfp; do
  for datatype in float half bfloat16; do
    run_stem baseline "$compressor" 1 "$datatype"
  done
done

for compressor in sdp4bit zfp; do
  for depth in 1 2 4 8; do
    for datatype in float half bfloat16; do
      run_stem current "$compressor" "$depth" "$datatype"
    done
  done
done

exit "$failures"
