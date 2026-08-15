#!/usr/bin/env bash
set -euo pipefail

variant=${1:?usage: m0_matrix.sh VARIANT SOURCE_ROOT smoke|full|memory}
source_root=${2:?usage: m0_matrix.sh VARIANT SOURCE_ROOT smoke|full|memory}
mode=${3:-full}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

source "$script_dir/m0_env.sh" "$source_root"

result_root=${M0_RESULT_ROOT:-/data/home/scyb672/run/lxc/COCCL-migrate/results/M0}
performance_runs=${M0_PERFORMANCE_RUNS:-2}
memory_runs=${M0_MEMORY_RUNS:-2}
case_timeout=${M0_CASE_TIMEOUT:-1800}
cooldown_seconds=${M0_8G_COOLDOWN_SECONDS:-2}
tests="$source_root/tests/coccl-tests/build"
command_root="$result_root/commands/current/$variant"
debug_root="$result_root/debug/current/$variant"
compressors=(sdp4bit cuzfp)
pipeline_depths=(1 2 4 8)

if [[ "$mode" == memory ]]; then
  raw_root="$result_root/raw/current-memory/$variant"
  sample_root="$result_root/memory-samples/current/$variant"
else
  raw_root="$result_root/raw/current/$variant"
  sample_root=
fi

mkdir -p "$raw_root" "$command_root" "$debug_root"
[[ -z "$sample_root" ]] || mkdir -p "$sample_root"
export NCCL_DEBUG_FILE="$debug_root/nccl.%h.%p.log"

set_compressor_env() {
  local collective=$1
  local compressor=$2

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
  local collective=$1
  local algorithm=$2
  local compressor=$3
  local depth=$4
  local executable=$5
  local bytes=$6
  local run_id=$7

  local stem="${collective}__${algorithm}__${compressor}__d${depth}__b${bytes}__r${run_id}"
  local log="$raw_root/$stem.log"
  local marker="$raw_root/$stem.ok"
  local command_file="$command_root/$stem.sh"
  local command=(
    timeout --signal=TERM --kill-after=30s "$case_timeout"
    "$tests/$executable"
    -b "$bytes" -e "$bytes" -f 2
    -t 4 -g 1 -w 10 -n 20 -c 0
  )

  if [[ -f "$marker" && ${M0_FORCE:-0} != 1 ]]; then
    return
  fi

  if [[ "$compressor" == native ]]; then
    export NCCL_ENABLE_COMPRESS=0
  else
    set_compressor_env "$collective" "$compressor"
  fi
  export NCCL_PIPELINE_DEPTH="$depth"

  {
    printf 'NCCL_ENABLE_COMPRESS=%q ' "$NCCL_ENABLE_COMPRESS"
    printf 'NCCL_PIPELINE_DEPTH=%q ' "$NCCL_PIPELINE_DEPTH"
    printf '%q ' "${command[@]}"
    printf '\n'
  } >"$command_file"

  printf '[%s] %s\n' "$(date -Is)" "$stem"
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

run_performance_range() {
  local collective=$1
  local algorithm=$2
  local compressor=$3
  local depth=$4
  local executable=$5
  local begin_bytes=$6
  local end_bytes=$7
  local bytes
  local run_id

  for ((bytes = begin_bytes; bytes <= end_bytes; bytes *= 2)); do
    for ((run_id = 1; run_id <= performance_runs; run_id++)); do
      run_case "$collective" "$algorithm" "$compressor" "$depth" \
        "$executable" "$bytes" "$run_id"
    done
  done
}

run_full_matrix() {
  local compressor
  local depth
  local standard_begin=1048576
  local standard_end=8589934592
  local allreduce_begin=4096
  local oneshot_end=33554432

  run_performance_range alltoall native native 1 alltoall_p2p_perf \
    "$standard_begin" "$standard_end"
  for compressor in "${compressors[@]}"; do
    for depth in "${pipeline_depths[@]}"; do
      run_performance_range alltoall pipeline "$compressor" "$depth" \
        alltoall_comp_overlap_perf "$standard_begin" "$standard_end"
    done
  done

  run_performance_range allgather native native 1 all_gather_perf \
    "$standard_begin" "$standard_end"
  for compressor in "${compressors[@]}"; do
    for depth in "${pipeline_depths[@]}"; do
      run_performance_range allgather pipeline "$compressor" "$depth" \
        all_gather_comp_overlap_perf "$standard_begin" "$standard_end"
    done
  done

  run_performance_range reducescatter native native 1 reduce_scatter_perf \
    "$standard_begin" "$standard_end"
  for compressor in "${compressors[@]}"; do
    for depth in "${pipeline_depths[@]}"; do
      run_performance_range reducescatter oneshot "$compressor" "$depth" \
        reduce_scatter_comp_oneshot_overlap_perf \
        "$standard_begin" "$standard_end"
    done
  done

  run_performance_range allreduce native native 1 all_reduce_perf \
    "$allreduce_begin" "$standard_end"
  for compressor in "${compressors[@]}"; do
    run_performance_range allreduce oneshot "$compressor" 1 \
      all_reduce_comp_oneshot_perf "$allreduce_begin" "$oneshot_end"
    for depth in "${pipeline_depths[@]}"; do
      run_performance_range allreduce twoshot "$compressor" "$depth" \
        all_reduce_comp_twoshot_overlap_perf \
        "$standard_begin" "$standard_end"
    done
  done
}

run_memory_matrix() {
  local bytes
  local run_id
  for bytes in 67108864 1073741824 8589934592; do
    for ((run_id = 1; run_id <= memory_runs; run_id++)); do
      run_case alltoall native native 1 alltoall_p2p_perf "$bytes" "$run_id"
      run_case allgather native native 1 all_gather_perf "$bytes" "$run_id"
      run_case reducescatter native native 1 reduce_scatter_perf "$bytes" "$run_id"
      run_case allreduce native native 1 all_reduce_perf "$bytes" "$run_id"

      run_case alltoall pipeline sdp4bit 4 alltoall_comp_overlap_perf "$bytes" "$run_id"
      run_case allgather pipeline sdp4bit 4 all_gather_comp_overlap_perf "$bytes" "$run_id"
      run_case reducescatter oneshot sdp4bit 4 \
        reduce_scatter_comp_oneshot_overlap_perf "$bytes" "$run_id"
      run_case allreduce twoshot sdp4bit 4 \
        all_reduce_comp_twoshot_overlap_perf "$bytes" "$run_id"
    done
  done
}

case "$mode" in
  smoke)
    run_case alltoall native native 1 alltoall_p2p_perf 1048576 1
    run_case alltoall pipeline sdp4bit 1 alltoall_comp_overlap_perf 1048576 1
    ;;
  full)
    run_full_matrix
    ;;
  memory)
    run_memory_matrix
    ;;
  *)
    printf 'unknown mode: %s\n' "$mode" >&2
    exit 2
    ;;
esac
