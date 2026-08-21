#!/usr/bin/env bash
set -euo pipefail

cuda_root=${1:-/data/apps/cuda/12.4}
coccl_root=${2:-/data/home/scyb672/run/lxc/COCCL-migrate}
gpus=${3:-4}
warmup=${COCCL_BENCH_WARMUP:-20}
iterations=${COCCL_BENCH_ITERATIONS:-30}
tests="$coccl_root/tests/coccl-tests/build"
plugin_root="$coccl_root/build/obj/coccl-extend/compressor_plugin/libcompress"
config_root=${COCCL_BENCH_CONFIG_ROOT:-/data/home/scyb672/run/codex-tmp/coccl-benchmark-configs/single-node}

if type module >/dev/null 2>&1; then
  module unload cuda >/dev/null 2>&1 || true
  module load cuda/12.4
fi

mkdir -p "$config_root"
export CUDA_HOME="$cuda_root"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$coccl_root/build/lib:$plugin_root:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_BUFFSIZE=16777216

config_for() {
  local compressor=$1
  local depth=$2
  local output="$config_root/${compressor}-d${depth}.toml"
  sed "s|^library_path = .*|library_path = \"$plugin_root\"|" \
    "$coccl_root/examples/benchmarks_scripts/configs/m16_${compressor}.toml" \
    >"$output"
  printf '\n[pipeline]\ndepth = %s\n' "$depth" >>"$output"
  printf '\n[autotune]\nenabled = false\n' >>"$output"
  printf '%s\n' "$output"
}

run_native() {
  local title=$1 executable=$2 begin=$3 end=$4
  echo "========== $title native =========="
  COCCL_ENABLE=0 "$tests/$executable" \
    -b "$begin" -e "$end" -f 2 -t "$gpus" -g 1 \
    -w "$warmup" -n "$iterations" -c 0
}

run_compressed() {
  local title=$1 executable=$2 compressor=$3 depth=$4 begin=$5 end=$6
  echo "========== $title $compressor depth=$depth =========="
  COCCL_ENABLE=1 COCCL_CONFIG_FILE=$(config_for "$compressor" "$depth") \
    "$tests/$executable" \
    -b "$begin" -e "$end" -f 2 -t "$gpus" -g 1 \
    -w "$warmup" -n "$iterations" -c 0
}

run_native alltoall alltoall_p2p_perf 1MB 8G
run_native allgather all_gather_perf 1MB 8G
run_native reducescatter reduce_scatter_perf 1MB 8G
run_native allreduce all_reduce_perf 4KB 8G

for compressor in sdp4bit zfp; do
  for depth in 1 2 4 8; do
    run_compressed alltoall alltoall_comp_perf "$compressor" "$depth" 1MB 8G
    run_compressed allgather all_gather_comp_perf "$compressor" "$depth" 1MB 8G
    run_compressed reducescatter reduce_scatter_comp_oneshot_perf \
      "$compressor" "$depth" 1MB 8G
    run_compressed allreduce all_reduce_comp_twoshot_perf \
      "$compressor" "$depth" 1MB 8G
  done
  run_compressed allreduce-oneshot all_reduce_comp_oneshot_perf \
    "$compressor" 1 4KB 32M
done
