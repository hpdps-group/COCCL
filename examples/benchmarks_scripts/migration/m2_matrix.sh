#!/usr/bin/env bash
set -euo pipefail

source_root=${1:?usage: m2_matrix.sh SOURCE_ROOT smoke|performance|memory}
mode=${2:-performance}
result_root=${M2_RESULT_ROOT:-/data/home/scyb672/run/lxc/COCCL-migrate/results/M2}
build_root=${M2_BUILD_ROOT:-$source_root/build}
tests=${M2_TESTS_DIR:-$source_root/tests/coccl-tests/build}
raw_root="$result_root/raw/$mode"
command_root="$result_root/commands/$mode"
sample_root="$result_root/memory-samples"
config_root="$result_root/runtime-configs"
timeout_seconds=${M2_CASE_TIMEOUT:-3600}

export CUDA_HOME=${CUDA_HOME:-/data/apps/cuda/12.4}
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$build_root/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export NCCL_DEBUG=WARN
export NCCL_BUFFSIZE=16777216
export NCCL_PIPELINE_DEPTH=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

mkdir -p "$raw_root" "$command_root" "$config_root"
[[ "$mode" != memory ]] || mkdir -p "$sample_root"

materialize_config() {
  local compressor=$1
  local template="$source_root/examples/benchmarks_scripts/configs/m2_${compressor}_explicit.toml"
  local output="$config_root/${compressor}.toml"
  sed "s|^library_path = .*|library_path = \"$build_root/obj/device/compress/libcompress\"|" \
    "$template" >"$output"
  printf '%s\n' "$output"
}

set_case_environment() {
  local compressor=$1
  export NCCL_ENABLE_COMPRESS=0
  unset COCCL_CONFIG_FILE
  if [[ "$compressor" == native ]]; then
    export COCCL_ENABLE=0
  else
    export COCCL_ENABLE=1
    export COCCL_CONFIG_FILE
    COCCL_CONFIG_FILE=$(materialize_config "$compressor")
  fi
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
  local executable=$4
  local bytes=$5
  local stem="${collective}__${algorithm}__${compressor}__b${bytes}"
  local log="$raw_root/$stem.log"
  local marker="$raw_root/$stem.ok"
  local command_file="$command_root/$stem.sh"
  local command=(
    timeout --signal=TERM --kill-after=30s "$timeout_seconds"
    "$tests/$executable"
    -b "$bytes" -e "$bytes" -f 2
    -t 4 -g 1 -w 20 -n 30 -c 0
  )

  if [[ -f "$marker" && ${M2_FORCE:-0} != 1 ]]; then
    return
  fi

  set_case_environment "$compressor"
  {
    printf 'COCCL_ENABLE=%q NCCL_ENABLE_COMPRESS=0 ' "$COCCL_ENABLE"
    if [[ "$compressor" != native ]]; then
      printf 'COCCL_CONFIG_FILE=%q ' "$COCCL_CONFIG_FILE"
    fi
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
  if (( bytes == 8589934592 )); then
    sleep 2
  fi
}

run_native() {
  local bytes=$1
  run_case alltoall native native alltoall_p2p_perf "$bytes"
  run_case allgather native native all_gather_perf "$bytes"
  run_case reducescatter native native reduce_scatter_perf "$bytes"
  run_case allreduce native native all_reduce_perf "$bytes"
}

run_compressed_alltoall() {
  local bytes=$1
  run_case alltoall explicit sdp4bit alltoall_comp_overlap_perf "$bytes"
  run_case alltoall explicit zfp alltoall_comp_overlap_perf "$bytes"
}

case "$mode" in
  smoke)
    run_case alltoall native native alltoall_p2p_perf 67108864
    run_compressed_alltoall 67108864
    ;;
  performance)
    for bytes in 67108864 536870912 1073741824 8589934592; do
      run_native "$bytes"
      run_compressed_alltoall "$bytes"
    done
    ;;
  memory)
    for bytes in 67108864 1073741824 8589934592; do
      run_native "$bytes"
      run_compressed_alltoall "$bytes"
    done
    ;;
  *)
    printf 'unknown mode: %s\n' "$mode" >&2
    exit 2
    ;;
esac
