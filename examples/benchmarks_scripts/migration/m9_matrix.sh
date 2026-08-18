#!/usr/bin/env bash
set -euo pipefail

source_root=${1:?usage: m9_matrix.sh SOURCE_ROOT native|sdp4bit|zfp smoke|performance|memory}
compressor=${2:?usage: m9_matrix.sh SOURCE_ROOT native|sdp4bit|zfp smoke|performance|memory}
mode=${3:-performance}
result_root=${M9_RESULT_ROOT:-/data/home/scyb672/run/lxc/COCCL-migrate/results/M9}
build_root=${M9_BUILD_ROOT:-$source_root/build}
tests=${M9_TESTS_DIR:-$source_root/tests/coccl-tests/build}
case_root="$result_root/$compressor"
raw_root="$case_root/raw/$mode"
command_root="$case_root/commands/$mode"
config_root="$case_root/runtime-configs"
sample_root="$case_root/memory-samples"
timeout_seconds=${M9_CASE_TIMEOUT:-3600}

case "$compressor" in
  native|sdp4bit|zfp) ;;
  *) printf 'unknown compressor: %s\n' "$compressor" >&2; exit 2 ;;
esac

export CUDA_HOME=${M9_CUDA_HOME:-/data/apps/cuda/12.4}
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$build_root/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export NCCL_BUFFSIZE=16777216
export NCCL_CUMEM_ENABLE=1
export NCCL_ENABLE_COMPRESS=0
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

mkdir -p "$raw_root" "$command_root" "$config_root"
[[ "$mode" != memory ]] || mkdir -p "$sample_root"

materialize_config() {
  local depth=$1
  local profile=$2
  local template="$source_root/examples/benchmarks_scripts/configs/m1_${compressor}.toml"
  if [[ "$profile" == subadd ]]; then
    template="$source_root/examples/benchmarks_scripts/configs/m9_sdp4bit_subadd.toml"
  fi
  local output="$config_root/${compressor}-${profile}-d${depth}.toml"
  local replacements=(
    -e "s|^library_path = .*|library_path = \"$build_root/obj/device/compress/libcompress\"|"
  )
  if [[ "$profile" == 8bit ]]; then
    replacements+=(
      -e 's/quantBits = 4/quantBits = 8/g'
      -e 's/inQuantBits = 4/inQuantBits = 8/g'
      -e 's/outQuantBits = 4/outQuantBits = 8/g'
    )
  fi
  sed "${replacements[@]}" "$template" >"$output"
  printf '\n[pipeline]\ndepth = %s\n' "$depth" >>"$output"
  printf '%s\n' "$output"
}

set_case_environment() {
  local depth=$1
  local profile=$2
  unset COCCL_CONFIG_FILE
  if [[ "$compressor" == native ]]; then
    export COCCL_ENABLE=0
  else
    export COCCL_ENABLE=1
    COCCL_CONFIG_FILE=$(materialize_config "$depth" "$profile")
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

executable_for() {
  local collective=$1
  local algorithm=$2
  case "$collective:$algorithm" in
    alltoall:native) printf 'alltoall_p2p_perf\n' ;;
    alltoall:pipeline) printf 'alltoall_comp_perf\n' ;;
    allgather:native) printf 'all_gather_perf\n' ;;
    allgather:pipeline) printf 'all_gather_comp_perf\n' ;;
    reducescatter:native) printf 'reduce_scatter_perf\n' ;;
    reducescatter:oneshot) printf 'reduce_scatter_comp_oneshot_perf\n' ;;
    allreduce:native) printf 'all_reduce_perf\n' ;;
    allreduce:oneshot) printf 'all_reduce_comp_oneshot_perf\n' ;;
    allreduce:twoshot) printf 'all_reduce_comp_twoshot_perf\n' ;;
    *) printf 'unsupported case: %s/%s\n' "$collective" "$algorithm" >&2; return 2 ;;
  esac
}

run_case() {
  local collective=$1
  local algorithm=$2
  local depth=$3
  local bytes=$4
  local datatype=${5:-float}
  local profile=${6:-default}
  local executable
  executable=$(executable_for "$collective" "$algorithm")
  local stem="${collective}__${algorithm}__${compressor}__d${depth}__b${bytes}"
  [[ "$datatype" == float ]] || stem+="__dtype${datatype}"
  [[ "$profile" == default ]] || stem+="__profile${profile}"
  local log="$raw_root/$stem.log"
  local marker="$raw_root/$stem.ok"
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
    -t 4 -g 1 -w "$warmup" -n "$iterations" -c 0 -d "$datatype"
  )

  if [[ -n ${M9_ONLY_STEM:-} && "$stem" != "$M9_ONLY_STEM" ]]; then
    return
  fi
  if [[ -f "$marker" && ${M9_FORCE:-0} != 1 ]]; then
    return
  fi
  [[ ${M9_FORCE:-0} != 1 ]] || rm -f "$marker"

  set_case_environment "$depth" "$profile"
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
    local status=0
    wait "$benchmark_pid" || status=$?
    wait "$sampler_pid"
    (( status == 0 )) || return "$status"
  else
    "${command[@]}" >"$log" 2>&1
  fi
  touch "$marker"
  (( bytes != 8589934592 )) || sleep 2
}

run_standard_range() {
  local collective=$1
  local algorithm=$2
  for ((bytes=1048576; bytes<=8589934592; bytes*=2)); do
    if [[ "$algorithm" == native ]]; then
      run_case "$collective" "$algorithm" 1 "$bytes"
    else
      for depth in 1 2 4 8; do
        run_case "$collective" "$algorithm" "$depth" "$bytes"
      done
    fi
  done
}

case "$mode" in
  smoke)
    if [[ "$compressor" == native ]]; then
      run_case alltoall native 1 67108864
      run_case allgather native 1 67108864
      run_case reducescatter native 1 67108864
      run_case allreduce native 1 67108864
    else
      run_case alltoall pipeline 4 67108864
      run_case allgather pipeline 4 67108864
      run_case reducescatter oneshot 4 67108864
      run_case allreduce oneshot 1 33554432
      run_case allreduce twoshot 4 67108864
      if [[ "$compressor" == sdp4bit ]]; then
        for datatype in half bfloat16; do
          run_case reducescatter oneshot 4 67108864 "$datatype"
          run_case allreduce twoshot 4 67108864 "$datatype"
        done
        run_case alltoall pipeline 4 67108864 float 8bit
        run_case reducescatter oneshot 4 67108864 float 8bit
        run_case allreduce twoshot 4 67108864 float 8bit
        for datatype in float half bfloat16; do
          run_case allgather pipeline 1 67108864 "$datatype" subadd
        done
      fi
    fi
    ;;
  performance)
    if [[ "$compressor" == native ]]; then
      run_standard_range alltoall native
      run_standard_range allgather native
      run_standard_range reducescatter native
      for ((bytes=4096; bytes<=8589934592; bytes*=2)); do
        run_case allreduce native 1 "$bytes"
      done
    else
      run_standard_range alltoall pipeline
      run_standard_range allgather pipeline
      run_standard_range reducescatter oneshot
      for ((bytes=4096; bytes<=33554432; bytes*=2)); do
        run_case allreduce oneshot 1 "$bytes"
      done
      run_standard_range allreduce twoshot
    fi
    ;;
  memory)
    [[ "$compressor" != native ]] || exit 0
    for bytes in 67108864 1073741824 8589934592; do
      run_case alltoall pipeline 4 "$bytes"
      run_case allgather pipeline 4 "$bytes"
      run_case reducescatter oneshot 4 "$bytes"
      run_case allreduce twoshot 4 "$bytes"
    done
    if [[ "$compressor" == sdp4bit ]]; then
      run_case allgather pipeline 1 67108864 float subadd
    fi
    ;;
  *) printf 'unknown mode: %s\n' "$mode" >&2; exit 2 ;;
esac
