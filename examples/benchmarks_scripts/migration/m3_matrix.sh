#!/usr/bin/env bash
set -euo pipefail

source_root=${1:?usage: m3_matrix.sh SOURCE_ROOT smoke|performance|memory|lifecycle}
mode=${2:-performance}
result_root=${M3_RESULT_ROOT:-/data/home/scyb672/run/lxc/COCCL-migrate/results/M3}
build_root=${M3_BUILD_ROOT:-$source_root/build}
tests=${M3_TESTS_DIR:-$source_root/tests/coccl-tests/build}
raw_root="$result_root/raw/$mode"
command_root="$result_root/commands/$mode"
sample_root="$result_root/memory-samples"
config_root="$result_root/runtime-configs"
timeout_seconds=${M3_CASE_TIMEOUT:-3600}

export CUDA_HOME=${CUDA_HOME:-/data/apps/cuda/12.4}
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$build_root/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
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
  local compressor=$1
  local executable=$2
  local bytes=$3
  local algorithm=native
  [[ "$compressor" == native ]] || algorithm=explicit
  local stem="alltoall__${algorithm}__${compressor}__b${bytes}"
  local log="$raw_root/$stem.log"
  local marker="$raw_root/$stem.ok"
  local command_file="$command_root/$stem.sh"
  local command=(
    timeout --signal=TERM --kill-after=30s "$timeout_seconds"
    "$tests/$executable"
    -b "$bytes" -e "$bytes" -f 2
    -t 4 -g 1 -w 20 -n 30 -c 0
  )

  if [[ -f "$marker" && ${M3_FORCE:-0} != 1 ]]; then
    return
  fi

  set_case_environment "$compressor"
  export NCCL_DEBUG=WARN
  unset NCCL_DEBUG_SUBSYS
  if [[ "$mode" == lifecycle ]]; then
    export NCCL_DEBUG=INFO
    export NCCL_DEBUG_SUBSYS=INIT
  fi

  {
    printf 'COCCL_ENABLE=%q NCCL_ENABLE_COMPRESS=0 NCCL_DEBUG=%q ' \
      "$COCCL_ENABLE" "$NCCL_DEBUG"
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

run_size() {
  local bytes=$1
  run_case native alltoall_p2p_perf "$bytes"
  run_case sdp4bit alltoall_comp_overlap_perf "$bytes"
  run_case zfp alltoall_comp_overlap_perf "$bytes"
}

case "$mode" in
  smoke)
    run_size 67108864
    ;;
  performance)
    for bytes in 67108864 536870912 1073741824 8589934592; do
      run_size "$bytes"
    done
    ;;
  memory)
    for bytes in 67108864 1073741824 8589934592; do
      run_size "$bytes"
    done
    ;;
  lifecycle)
    run_case sdp4bit alltoall_comp_overlap_perf 67108864
    ;;
  *)
    printf 'unknown mode: %s\n' "$mode" >&2
    exit 2
    ;;
esac
