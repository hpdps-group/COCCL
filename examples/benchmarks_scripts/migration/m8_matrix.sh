#!/usr/bin/env bash
set -euo pipefail

source_root=${1:?usage: m8_matrix.sh SOURCE_ROOT smoke|performance|layout|memory}
mode=${2:-performance}
result_root=${M8_RESULT_ROOT:-/data/home/scyb672/run/lxc/COCCL-migrate/results/M8}
build_root=${M8_BUILD_ROOT:-$source_root/build}
tests=${M8_TESTS_DIR:-$source_root/tests/coccl-tests/build}
raw_root="$result_root/raw/$mode"
command_root="$result_root/commands/$mode"
config_root="$result_root/runtime-configs"
sample_root="$result_root/memory-samples"
timeout_seconds=${M8_CASE_TIMEOUT:-3600}

export CUDA_HOME=${M8_CUDA_HOME:-/data/apps/cuda/12.4}
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$build_root/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export NCCL_BUFFSIZE=16777216
export NCCL_CUMEM_ENABLE=1
export NCCL_ENABLE_COMPRESS=0
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

mkdir -p "$raw_root" "$command_root" "$config_root"
[[ "$mode" != memory ]] || mkdir -p "$sample_root"

materialize_config() {
  local compressor=$1
  local depth=$2
  local template="$source_root/examples/benchmarks_scripts/configs/m1_${compressor}.toml"
  local output="$config_root/${compressor}-d${depth}.toml"
  sed "s|^library_path = .*|library_path = \"$build_root/obj/device/compress/libcompress\"|" \
    "$template" >"$output"
  printf '\n[pipeline]\ndepth = %s\n' "$depth" >>"$output"
  printf '%s\n' "$output"
}

set_case_environment() {
  local compressor=$1
  local depth=$2
  unset COCCL_CONFIG_FILE
  if [[ "$compressor" == native ]]; then
    export COCCL_ENABLE=0
  else
    export COCCL_ENABLE=1
    COCCL_CONFIG_FILE=$(materialize_config "$compressor" "$depth")
    export COCCL_CONFIG_FILE
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
  local algorithm=$1
  local compressor=$2
  local depth=$3
  local bytes=$4
  local executable=all_reduce_perf
  case "$algorithm" in
    native) executable=all_reduce_perf ;;
    oneshot) executable=all_reduce_comp_oneshot_perf ;;
    twoshot) executable=all_reduce_comp_twoshot_perf ;;
    *) printf 'unknown algorithm: %s\n' "$algorithm" >&2; exit 2 ;;
  esac
  local stem="allreduce__${algorithm}__${compressor}__d${depth}__b${bytes}"
  local log="$raw_root/$stem.log"
  local marker="$raw_root/$stem.ok"
  local oom_marker="$raw_root/$stem.oom"
  local command_file="$command_root/$stem.sh"
  local warmup=20
  local iterations=30
  if [[ "$mode" == smoke ]]; then
    warmup=2
    iterations=5
  fi
  local command=(
    timeout --signal=TERM --kill-after=30s "$timeout_seconds"
    "$tests/$executable" -b "$bytes" -e "$bytes" -f 2
    -t 4 -g 1 -w "$warmup" -n "$iterations" -c 0
  )

  if [[ -n ${M8_ONLY_STEM:-} && "$stem" != "$M8_ONLY_STEM" ]]; then
    return
  fi
  if [[ (-f "$marker" || -f "$oom_marker") && ${M8_FORCE:-0} != 1 ]]; then
    return
  fi
  if [[ ${M8_FORCE:-0} == 1 ]]; then
    rm -f "$marker" "$oom_marker"
  fi

  set_case_environment "$compressor" "$depth"
  export NCCL_DEBUG=WARN
  unset NCCL_DEBUG_SUBSYS
  if [[ "$mode" == memory ]]; then
    export NCCL_DEBUG=INFO
    export NCCL_DEBUG_SUBSYS=INIT
  fi

  {
    printf 'COCCL_ENABLE=%q NCCL_ENABLE_COMPRESS=0 NCCL_CUMEM_ENABLE=1 ' \
      "$COCCL_ENABLE"
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
    local benchmark_status=0
    if wait "$benchmark_pid"; then
      benchmark_status=0
    else
      benchmark_status=$?
    fi
    wait "$sampler_pid"
    if (( benchmark_status != 0 )); then
      if grep -q "Cuda failure 2 'out of memory'" "$log"; then
        touch "$oom_marker"
        if (( bytes == 8589934592 )); then sleep 2; fi
        return
      fi
      return "$benchmark_status"
    fi
  else
    "${command[@]}" >"$log" 2>&1
  fi
  rm -f "$oom_marker"
  touch "$marker"
  if (( bytes == 8589934592 )); then sleep 2; fi
}

run_layout() {
  local bytes=$1
  local depth=$2
  local stem="layout__b${bytes}__d${depth}"
  local log="$raw_root/$stem.log"
  local marker="$raw_root/$stem.ok"
  if [[ -n ${M8_ONLY_STEM:-} && "$stem" != "$M8_ONLY_STEM" ]]; then
    return
  fi
  [[ -f "$marker" && ${M8_FORCE:-0} != 1 ]] && return
  printf '[%s] %s\n' "$(date -Is)" "$stem"
  timeout --signal=TERM --kill-after=30s "$timeout_seconds" \
    "$tests/coccl_m5_pipeline_layout_test" \
    --benchmark "$bytes" "$depth" >"$log" 2>&1
  touch "$marker"
  if (( bytes == 8589934592 )); then sleep 2; fi
}

case "$mode" in
  smoke)
    run_case native native 1 67108864
    for compressor in sdp4bit zfp; do
      for ((bytes=4096; bytes<=33554432; bytes*=2)); do
        run_case oneshot "$compressor" 1 "$bytes"
      done
      for depth in 1 2 4 8; do
        run_case twoshot "$compressor" "$depth" 67108864
      done
    done
    ;;
  performance)
    for ((bytes=4096; bytes<=8589934592; bytes*=2)); do
      run_case native native 1 "$bytes"
    done
    for compressor in sdp4bit zfp; do
      for ((bytes=4096; bytes<=33554432; bytes*=2)); do
        run_case oneshot "$compressor" 1 "$bytes"
      done
      for ((bytes=1048576; bytes<=8589934592; bytes*=2)); do
        for depth in 1 2 4 8; do
          run_case twoshot "$compressor" "$depth" "$bytes"
        done
      done
    done
    ;;
  layout)
    for ((bytes=1048576; bytes<=8589934592; bytes*=2)); do
      for depth in 1 2 4 8; do
        run_layout "$bytes" "$depth"
      done
    done
    ;;
  memory)
    for compressor in sdp4bit zfp; do
      run_case oneshot "$compressor" 1 33554432
    done
    for bytes in 67108864 1073741824 8589934592; do
      for compressor in sdp4bit zfp; do
        for depth in 1 2 4 8; do
          run_case twoshot "$compressor" "$depth" "$bytes"
        done
      done
    done
    ;;
  *)
    printf 'unknown mode: %s\n' "$mode" >&2
    exit 2
    ;;
esac
