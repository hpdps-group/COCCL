#!/usr/bin/env bash
set -euo pipefail

root=/data/home/scyb672/run/lxc/COCCL-migrate
result_root=${M0_RESULT_ROOT:-$root/results/M0}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mode=${1:-performance}
case_timeout=${M0_CASE_TIMEOUT:-1800}
cooldown_seconds=${M0_8G_COOLDOWN_SECONDS:-2}

source "$script_dir/m0_env.sh" "$root"
base_library_path=$LD_LIBRARY_PATH

executable_for() {
  case "$1:$2" in
    alltoall:native) echo alltoall_p2p_perf ;;
    alltoall:pipeline) echo alltoall_comp_overlap_perf ;;
    allgather:native) echo all_gather_perf ;;
    allgather:pipeline) echo all_gather_comp_overlap_perf ;;
    reducescatter:native) echo reduce_scatter_perf ;;
    reducescatter:oneshot) echo reduce_scatter_comp_oneshot_overlap_perf ;;
    allreduce:native) echo all_reduce_perf ;;
    allreduce:oneshot) echo all_reduce_comp_oneshot_perf ;;
    allreduce:twoshot) echo all_reduce_comp_twoshot_overlap_perf ;;
  esac
}

set_compressor_env() {
  local collective=$1
  local compressor=$2

  if [[ "$compressor" == native ]]; then
    export NCCL_ENABLE_COMPRESS=0
    return
  fi

  export NCCL_ENABLE_COMPRESS=1
  case "$collective" in
    alltoall)
      export NCCL_ALLTOALL_COMPRESSORS="$compressor"
      ;;
    allgather)
      export NCCL_ALLGATHER_COMPRESSORS="$compressor"
      export NCCL_ALLGATHER_INTER_COMPRESSORS="$compressor"
      ;;
    reducescatter)
      export NCCL_REDUCESCATTER_COMPRESSORS="$compressor"
      export NCCL_REDUCESCATTER_INTER_COMPRESSORS="$compressor"
      ;;
    allreduce)
      export NCCL_ALLREDUCE_COMPRESSORS="$compressor"
      export NCCL_ALLREDUCE_INTER_COMPRESSORS="$compressor"
      ;;
  esac
}

sample_memory() {
  local benchmark_pid=$1
  local output=$2

  while kill -0 "$benchmark_pid" 2>/dev/null; do
    nvidia-smi \
      --query-compute-apps=pid,gpu_uuid,used_gpu_memory \
      --format=csv,noheader,nounits 2>/dev/null |
      while IFS= read -r line; do
        printf '%s,%s\n' "$(date +%s.%N)" "$line"
      done
    sleep 0.05
  done >"$output"
}

run_case() {
  local variant=$1
  local source_root=$2
  local collective=$3
  local algorithm=$4
  local compressor=$5
  local depth=$6
  local bytes=$7
  local run_id=$8
  local executable
  executable=$(executable_for "$collective" "$algorithm")

  local stem="${collective}__${algorithm}__${compressor}__d${depth}__b${bytes}__r${run_id}"
  local output_root="$result_root/confirmation/current/$mode/$variant"
  local command_root="$result_root/confirmation/current/commands-$mode/$variant"
  local debug_root="$result_root/debug/confirmation-current/$mode/$variant"
  local sample_root="$result_root/confirmation/current/memory-samples/$variant"
  local log="$output_root/$stem.log"
  local marker="$output_root/$stem.ok"

  if [[ -f "$marker" && ${M0_FORCE:-0} != 1 ]]; then
    return
  fi

  mkdir -p "$output_root" "$command_root" "$debug_root"
  [[ "$mode" != memory ]] || mkdir -p "$sample_root"
  export NCCL_HOME="$source_root/build"
  export LD_LIBRARY_PATH="$NCCL_HOME/lib:$base_library_path"
  export NCCL_COMPRESSORS_CONFIG_PATH="$source_root/examples/benchmarks_scripts/configs"
  export NCCL_COMPRESSORS_LIB_PATH="$source_root/build/obj/device/compress/libcompress"
  export NCCL_DEBUG_FILE="$debug_root/nccl.%h.%p.log"
  export NCCL_PIPELINE_DEPTH="$depth"
  set_compressor_env "$collective" "$compressor"

  local command=(
    timeout --signal=TERM --kill-after=30s "$case_timeout"
    "$source_root/tests/coccl-tests/build/$executable"
    -b "$bytes" -e "$bytes" -f 2
    -t 4 -g 1 -w 10 -n 20 -c 0
  )

  printf '%q ' "${command[@]}" >"$command_root/$stem.sh"
  printf '\n' >>"$command_root/$stem.sh"
  printf '[%s] %s confirmation %s/%s\n' \
    "$(date -Is)" "$mode" "$variant" "$stem"
  if [[ "$mode" == memory ]]; then
    "${command[@]}" >"$log" 2>&1 &
    local benchmark_pid=$!
    sample_memory "$benchmark_pid" "$sample_root/$stem.csv" &
    local sampler_pid=$!
    wait "$benchmark_pid"
    wait "$sampler_pid"
  else
    "${command[@]}" >"$log" 2>&1
  fi
  touch "$marker"

  if (( bytes == 8589934592 && cooldown_seconds > 0 )); then
    sleep "$cooldown_seconds"
  fi
}

if [[ "$mode" == performance ]]; then
  input="$result_root/comparison.csv"
  filter='NR > 1 && $10 == 1 && $11 == 0'
elif [[ "$mode" == memory ]]; then
  input="$result_root/memory-comparison.csv"
  filter='NR > 1 && $10 == 0'
else
  printf 'unknown confirmation mode: %s\n' "$mode" >&2
  exit 2
fi

while IFS=, read -r collective algorithm compressor depth bytes; do
  run_case original /data/home/scyb672/run/lxc/COCCL \
    "$collective" "$algorithm" "$compressor" "$depth" "$bytes" 1
  run_case migrate-copy "$root" \
    "$collective" "$algorithm" "$compressor" "$depth" "$bytes" 1
  run_case migrate-copy "$root" \
    "$collective" "$algorithm" "$compressor" "$depth" "$bytes" 2
  run_case original /data/home/scyb672/run/lxc/COCCL \
    "$collective" "$algorithm" "$compressor" "$depth" "$bytes" 2
done < <(awk -F, "$filter {
  key = \$1 FS \$2 FS \$3 FS \$4 FS \$5
  if (!seen[key]++) print \$1, \$2, \$3, \$4, \$5
}" OFS=, "$input")

python3 "$script_dir/m0_parse_confirmation.py" "$result_root"
