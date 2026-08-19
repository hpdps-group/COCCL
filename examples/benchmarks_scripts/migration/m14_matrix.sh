#!/usr/bin/env bash
set -euo pipefail

source_root=${1:?usage: m14_matrix.sh SOURCE_ROOT MODE}
mode=${2:-completion}
result_root=${M14_RESULT_ROOT:-/data/home/scyb672/run/lxc/COCCL-migrate/results/M14}
build_root=${M14_BUILD_ROOT:-/data/home/scyb672/run/lxc/COCCL-migrate/build}
nccl_lib_root=${M14_NCCL_LIB_ROOT:-$build_root/lib}
plugin_root=${M14_PLUGIN_ROOT:-$build_root/obj/device/compress/libcompress}
tests=${M14_TESTS_DIR:-/data/home/scyb672/run/lxc/COCCL-migrate/tests/coccl-tests/build}
raw_root="$result_root/raw/$mode"
command_root="$result_root/commands/$mode"
config_root="$result_root/runtime-configs"
sample_root="$result_root/memory-samples"
timeout_seconds=${M14_CASE_TIMEOUT:-7200}
warmup=${M14_WARMUP:-20}
iterations=${M14_ITERATIONS:-30}

case "$mode" in
  smoke|completion|sweep|endpoint|memory|fixed) ;;
  *) printf 'unsupported mode: %s\n' "$mode" >&2; exit 2 ;;
esac

export CUDA_HOME=${M14_CUDA_HOME:-/data/apps/cuda/12.4}
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$nccl_lib_root:$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export NCCL_BUFFSIZE=16777216
export NCCL_CUMEM_ENABLE=1
export NCCL_ENABLE_COMPRESS=0
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

mkdir -p "$raw_root" "$command_root" "$config_root"
[[ "$mode" != memory ]] || mkdir -p "$sample_root"

materialize_config() {
  local profile=$1
  local depth=$2
  local template
  case "$profile" in
    dietgpu) template="$source_root/examples/benchmarks_scripts/configs/m14_dietgpu.toml" ;;
    sdp4bit|zfp) template="$source_root/examples/benchmarks_scripts/configs/m1_$profile.toml" ;;
  esac
  local output="$config_root/${profile}-d${depth}.toml"
  sed "s|^library_path = .*|library_path = \"$plugin_root\"|" \
    "$template" >"$output"
  if [[ "$profile" == dietgpu ]]; then
    sed -i "s/probBits = [0-9][0-9]*/probBits = ${M14_PROB_BITS:-10}/" \
      "$output"
  fi
  printf '\n[pipeline]\ndepth = %s\n' "$depth" >>"$output"
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
  case "$1" in
    sendrecv) printf 'sendrecv_comp_perf\n' ;;
    alltoall) printf 'alltoall_comp_perf\n' ;;
    allgather) printf 'all_gather_comp_perf\n' ;;
    reducescatter-oneshot) printf 'reduce_scatter_comp_oneshot_perf\n' ;;
    reducescatter-twoshot) printf 'reduce_scatter_comp_twoshot_perf\n' ;;
    allreduce-oneshot) printf 'all_reduce_comp_oneshot_perf\n' ;;
    allreduce-twoshot) printf 'all_reduce_comp_twoshot_perf\n' ;;
    allreduce-tripleshot) printf 'all_reduce_comp_tripleshot_perf\n' ;;
  esac
}

run_case() {
  local profile=$1
  local recipe=$2
  local depth=$3
  local bytes=$4
  local executable
  executable=$(executable_for "$recipe")
  local stem="${recipe}__${profile}__d${depth}__b${bytes}"
  local log="$raw_root/$stem.log"
  local marker="$raw_root/$stem.ok"
  local command_file="$command_root/$stem.sh"
  local command=(
    timeout --signal=TERM --kill-after=30s "$timeout_seconds"
    "$tests/$executable" -b "$bytes" -e "$bytes" -f 2
    -t 4 -g 1 -w "$warmup" -n "$iterations" -c 0 -d float
  )

  if [[ -n ${M14_ONLY_STEM:-} && "$stem" != "$M14_ONLY_STEM" ]]; then
    return
  fi
  if [[ -f "$marker" && ${M14_FORCE:-0} != 1 ]]; then
    return
  fi
  [[ ${M14_FORCE:-0} != 1 ]] || rm -f "$marker"

  export COCCL_ENABLE=1
  export COCCL_CONFIG_FILE
  COCCL_CONFIG_FILE=$(materialize_config "$profile" "$depth")
  export NCCL_DEBUG=WARN
  unset NCCL_DEBUG_SUBSYS
  if [[ "$mode" == memory ]]; then
    export NCCL_DEBUG=INFO
    export NCCL_DEBUG_SUBSYS=INIT
  fi
  {
    printf 'COCCL_ENABLE=1 COCCL_CONFIG_FILE=%q ' "$COCCL_CONFIG_FILE"
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

recipes=(
  sendrecv alltoall allgather
  reducescatter-oneshot reducescatter-twoshot
  allreduce-oneshot allreduce-twoshot allreduce-tripleshot
)

case "$mode" in
  smoke)
    for recipe in "${recipes[@]}"; do
      run_case dietgpu "$recipe" 1 1048576
    done
    ;;
  completion)
    for recipe in "${recipes[@]}"; do
      bytes=67108864
      [[ "$recipe" != allreduce-oneshot ]] || bytes=33554432
      for depth in 1 2 4 8; do
        run_case dietgpu "$recipe" "$depth" "$bytes"
      done
    done
    ;;
  sweep)
    for recipe in sendrecv alltoall allgather reducescatter-oneshot \
                  allreduce-twoshot; do
      for depth in 1 2 4 8; do
        for ((bytes=1048576; bytes<=8589934592; bytes*=2)); do
          run_case dietgpu "$recipe" "$depth" "$bytes"
        done
      done
    done
    for ((bytes=4096; bytes<=33554432; bytes*=2)); do
      run_case dietgpu allreduce-oneshot 1 "$bytes"
    done
    ;;
  endpoint)
    for recipe in "${recipes[@]}"; do
      run_case dietgpu "$recipe" 1 1048576
      bytes=8589934592
      [[ "$recipe" != allreduce-oneshot ]] || bytes=33554432
      run_case dietgpu "$recipe" 1 "$bytes"
    done
    ;;
  memory)
    for recipe in "${recipes[@]}"; do
      bytes=67108864
      [[ "$recipe" != allreduce-oneshot ]] || bytes=33554432
      run_case dietgpu "$recipe" 4 "$bytes"
    done
    ;;
  fixed)
    run_case sdp4bit alltoall 4 67108864
    run_case zfp allgather 4 67108864
    ;;
esac
