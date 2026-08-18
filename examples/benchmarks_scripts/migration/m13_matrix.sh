#!/usr/bin/env bash
set -euo pipefail

source_root=${1:?usage: m13_matrix.sh SOURCE_ROOT PROFILE MODE}
profile=${2:?usage: m13_matrix.sh SOURCE_ROOT PROFILE MODE}
mode=${3:-performance}
result_root=${M13_RESULT_ROOT:-/data/home/scyb672/run/lxc/COCCL-migrate/results/M13}
build_root=${M13_BUILD_ROOT:-/data/home/scyb672/run/lxc/COCCL-migrate/build}
nccl_lib_root=${M13_NCCL_LIB_ROOT:-$build_root/lib}
plugin_root=${M13_PLUGIN_ROOT:-$build_root/obj/device/compress/libcompress}
tests=${M13_TESTS_DIR:-/data/home/scyb672/run/lxc/COCCL-migrate/tests/coccl-tests/build}
case_root="$result_root/$profile"
raw_root="$case_root/raw/$mode"
command_root="$case_root/commands/$mode"
config_root="$case_root/runtime-configs"
sample_root="$case_root/memory-samples"
timeout_seconds=${M13_CASE_TIMEOUT:-3600}

case "$profile:$mode" in
  native:smoke|native:performance|native:memory|\
  fallback:smoke|fallback:performance|fallback:memory|\
  auto-sdp4bit:smoke|auto-sdp4bit:deadlock|\
  sdp4bit:smoke|sdp4bit:performance|sdp4bit:memory|sdp4bit:deadlock|\
  zfp:smoke|zfp:performance|zfp:memory|zfp:deadlock|\
  zfp-raw:smoke|zfp-raw:performance|zfp-raw:memory|zfp-raw:deadlock) ;;
  *) printf 'unsupported profile/mode: %s/%s\n' "$profile" "$mode" >&2; exit 2 ;;
esac

export CUDA_HOME=${M13_CUDA_HOME:-/data/apps/cuda/12.4}
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$nccl_lib_root:$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export NCCL_BUFFSIZE=16777216
export NCCL_CUMEM_ENABLE=1
export NCCL_ENABLE_COMPRESS=0
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

mkdir -p "$raw_root" "$command_root" "$config_root"
[[ "$mode" != memory ]] || mkdir -p "$sample_root"

materialize_config() {
  local template_profile=$profile
  [[ "$template_profile" != auto-sdp4bit ]] || template_profile=sdp4bit
  [[ "$template_profile" != fallback ]] || template_profile=sdp4bit
  local template="$source_root/examples/benchmarks_scripts/configs/m13_${template_profile//-/_}.toml"
  local output="$config_root/$profile.toml"
  sed "s|^library_path = .*|library_path = \"$plugin_root\"|" \
    "$template" >"$output"
  if [[ "$profile" == fallback ]]; then
    sed -i 's/compression_threshold_bytes = 0/compression_threshold_bytes = 17179869184/' "$output"
  fi
  printf '%s\n' "$output"
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

executable_for() {
  case "$profile" in
    native|fallback|auto-sdp4bit) printf 'sendrecv_perf\n' ;;
    *) printf 'sendrecv_comp_perf\n' ;;
  esac
}

configure_runtime() {
  unset COCCL_ENABLE COCCL_CONFIG_FILE
  if [[ "$profile" != native ]]; then
    export COCCL_ENABLE=1
    export COCCL_CONFIG_FILE
    COCCL_CONFIG_FILE=$(materialize_config)
  fi
}

run_case() {
  local pattern=$1
  local bytes=$2
  local executable
  executable=$(executable_for)
  local stem="sendrecv__${pattern}__${profile}__b${bytes}"
  local log="$raw_root/$stem.log"
  local marker="$raw_root/$stem.ok"
  local command_file="$command_root/$stem.sh"
  local warmup=20
  local iterations=30
  if [[ "$mode" == smoke || "$mode" == deadlock ]]; then
    warmup=2
    iterations=5
  fi
  local command=(
    timeout --signal=TERM --kill-after=30s "$timeout_seconds"
    "$tests/$executable" -b "$bytes" -e "$bytes" -f 2
    -t 4 -g 1 -w "$warmup" -n "$iterations" -c 0 -d float
  )

  if [[ -n ${M13_ONLY_STEM:-} && "$stem" != "$M13_ONLY_STEM" ]]; then
    return
  fi
  if [[ -f "$marker" && ${M13_FORCE:-0} != 1 ]]; then
    return
  fi
  [[ ${M13_FORCE:-0} != 1 ]] || rm -f "$marker"

  configure_runtime
  unset COCCL_SENDRECV_SAME_STREAM_BIDIRECTIONAL
  unset COCCL_SENDRECV_BATCH_STRESS
  case "$pattern" in
    ring) ;;
    bidirectional-same-stream)
      export COCCL_SENDRECV_SAME_STREAM_BIDIRECTIONAL=1 ;;
    bidirectional-multistream)
      export COCCL_SENDRECV_BATCH_STRESS=1 ;;
  esac
  export NCCL_DEBUG=WARN
  unset NCCL_DEBUG_SUBSYS
  if [[ "$mode" == memory ]]; then
    export NCCL_DEBUG=INFO
    export NCCL_DEBUG_SUBSYS=INIT
  fi
  {
    printf 'COCCL_ENABLE=%q ' "${COCCL_ENABLE:-0}"
    printf 'COCCL_CONFIG_FILE=%q ' "${COCCL_CONFIG_FILE:-}"
    printf 'COCCL_SENDRECV_SAME_STREAM_BIDIRECTIONAL=%q ' \
      "${COCCL_SENDRECV_SAME_STREAM_BIDIRECTIONAL:-0}"
    printf 'COCCL_SENDRECV_BATCH_STRESS=%q ' \
      "${COCCL_SENDRECV_BATCH_STRESS:-0}"
    printf '%q ' "${command[@]}"
    printf '\n'
  } >"$command_file"

  printf '[%s] %s\n' "$(date -Is)" "$stem"
  if [[ "$mode" == memory ]]; then
    "${command[@]}" >"$log" 2>&1 &
    local benchmark_pid=$!
    sample_memory "$benchmark_pid" "$sample_root/$stem.csv" &
    local sampler_pid=$!
    local status=0
    wait "$benchmark_pid" || status=$?
    wait "$sampler_pid"
    (( status == 0 )) || return "$status"
  else
    "${command[@]}" >"$log" 2>&1
  fi
  grep -q '^#  Rank  3 ' "$log"
  grep -Eq '^[[:space:]]+[0-9]+[[:space:]]+[0-9]+' "$log"
  touch "$marker"
  (( bytes != 8589934592 )) || sleep 2
}

case "$mode" in
  smoke)
    run_case ring 1048576
    ;;
  deadlock)
    run_case ring 67108864
    run_case bidirectional-same-stream 67108864
    run_case bidirectional-multistream 67108864
    ;;
  performance|memory)
    for bytes in 1048576 67108864 536870912 1073741824 8589934592; do
      run_case ring "$bytes"
    done
    ;;
esac
