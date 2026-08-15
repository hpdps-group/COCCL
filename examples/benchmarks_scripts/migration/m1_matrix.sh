#!/usr/bin/env bash
set -euo pipefail

variant=${1:?usage: m1_matrix.sh original|migrate SOURCE_ROOT performance|memory}
source_root=${2:?usage: m1_matrix.sh original|migrate SOURCE_ROOT performance|memory}
mode=${3:-performance}

result_root=${M1_RESULT_ROOT:-/data/home/scyb672/run/lxc/COCCL-migrate/results/M1}
tests="$source_root/tests/coccl-tests/build"
raw_root="$result_root/raw/$mode/$variant"
command_root="$result_root/commands/$mode/$variant"
sample_root="$result_root/memory-samples/$variant"
timeout_seconds=${M1_CASE_TIMEOUT:-3600}
read -r -a compressors <<<"${M1_COMPRESSORS:-native sdp4bit zfp}"

export CUDA_HOME=${CUDA_HOME:-/data/apps/cuda/12.4}
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$source_root/build/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export NCCL_DEBUG=WARN
export NCCL_BUFFSIZE=16777216
export NCCL_PIPELINE_DEPTH=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

mkdir -p "$raw_root" "$command_root"
[[ "$mode" != memory ]] || mkdir -p "$sample_root"

set_case_environment() {
  local compressor=$1

  unset COCCL_ENABLE COCCL_CONFIG_FILE
  if [[ "$compressor" == native ]]; then
    export NCCL_ENABLE_COMPRESS=0
    return
  fi

  export NCCL_ENABLE_COMPRESS=1
  if [[ "$variant" == original ]]; then
    local original_name=$compressor
    [[ "$compressor" != zfp ]] || original_name=cuzfp
    export NCCL_COMPRESSORS=sdp4bit,cuzfp
    export NCCL_COMPRESSORS_CONFIG_PATH="$source_root/examples/benchmarks_scripts/configs"
    export NCCL_COMPRESSORS_LIB_PATH="$source_root/build/obj/device/compress/libcompress"
    export NCCL_ENABLE_ALLTOALL_COMPRESS=1
    export NCCL_ALLTOALL_COMPRESSORS="$original_name"
  else
    export COCCL_ENABLE=1
    export COCCL_CONFIG_FILE="$source_root/examples/benchmarks_scripts/configs/m1_${compressor}.toml"
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
  local bytes=$2
  local executable=alltoall_comp_overlap_perf
  [[ "$compressor" != native ]] || executable=alltoall_p2p_perf

  local stem="alltoall__${compressor}__b${bytes}"
  local log="$raw_root/$stem.log"
  local marker="$raw_root/$stem.ok"
  local command_file="$command_root/$stem.sh"
  local command=(
    timeout --signal=TERM --kill-after=30s "$timeout_seconds"
    "$tests/$executable"
    -b "$bytes" -e "$bytes" -f 2
    -t 4 -g 1 -w 20 -n 30 -c 0
  )

  if [[ -f "$marker" && ${M1_FORCE:-0} != 1 ]]; then
    return
  fi

  set_case_environment "$compressor"
  {
    printf 'NCCL_ENABLE_COMPRESS=%q ' "$NCCL_ENABLE_COMPRESS"
    printf 'NCCL_PIPELINE_DEPTH=1 '
    if [[ "$variant" == original && "$compressor" != native ]]; then
      printf 'NCCL_ALLTOALL_COMPRESSORS=%q ' "$NCCL_ALLTOALL_COMPRESSORS"
    elif [[ "$variant" == migrate && "$compressor" != native ]]; then
      printf 'COCCL_ENABLE=1 COCCL_CONFIG_FILE=%q ' "$COCCL_CONFIG_FILE"
    fi
    printf '%q ' "${command[@]}"
    printf '\n'
  } >"$command_file"

  printf '[%s] %s/%s\n' "$(date -Is)" "$variant" "$stem"
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

case "$mode" in
  performance)
    for compressor in "${compressors[@]}"; do
      for ((bytes = 1048576; bytes <= 8589934592; bytes *= 2)); do
        run_case "$compressor" "$bytes"
      done
    done
    ;;
  memory)
    for compressor in "${compressors[@]}"; do
      for bytes in 67108864 1073741824 8589934592; do
        run_case "$compressor" "$bytes"
      done
    done
    ;;
  *)
    printf 'unknown mode: %s\n' "$mode" >&2
    exit 2
    ;;
esac
