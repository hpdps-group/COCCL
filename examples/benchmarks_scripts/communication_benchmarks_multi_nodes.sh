#!/usr/bin/env bash
set -euo pipefail

cuda_root=${1:?usage: communication_benchmarks_multi_nodes.sh CUDA MPI COCCL GPUS_PER_NODE NNODES HOSTFILE}
mpi_root=${2:?usage: communication_benchmarks_multi_nodes.sh CUDA MPI COCCL GPUS_PER_NODE NNODES HOSTFILE}
coccl_root=${3:?usage: communication_benchmarks_multi_nodes.sh CUDA MPI COCCL GPUS_PER_NODE NNODES HOSTFILE}
gpus_per_node=${4:?usage: communication_benchmarks_multi_nodes.sh CUDA MPI COCCL GPUS_PER_NODE NNODES HOSTFILE}
nodes=${5:?usage: communication_benchmarks_multi_nodes.sh CUDA MPI COCCL GPUS_PER_NODE NNODES HOSTFILE}
hostfile=${6:?usage: communication_benchmarks_multi_nodes.sh CUDA MPI COCCL GPUS_PER_NODE NNODES HOSTFILE}
total_ranks=$((gpus_per_node * nodes))
warmup=${COCCL_BENCH_WARMUP:-20}
iterations=${COCCL_BENCH_ITERATIONS:-30}
tests="$coccl_root/tests/coccl-tests/build"
plugin_root="$coccl_root/build/obj/coccl-extend/compressor_plugin/libcompress"
config_root=${COCCL_BENCH_CONFIG_ROOT:-/data/home/scyb672/run/codex-tmp/coccl-benchmark-configs/multi-node}

mkdir -p "$config_root"
export CUDA_HOME="$cuda_root"
export PATH="$CUDA_HOME/bin:$mpi_root/bin:$PATH"
export LD_LIBRARY_PATH="$coccl_root/build/lib:$plugin_root:$CUDA_HOME/lib64:$mpi_root/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_BUFFSIZE=16777216

launcher=(
  "$mpi_root/bin/mpirun" -np "$total_ranks" --hostfile "$hostfile"
  --map-by "ppr:${gpus_per_node}:node" --bind-to none --mca btl '^openib'
)

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

run_test() {
  local title=$1 executable=$2 begin=$3 end=$4 enabled=$5 config=${6:-}
  local environment=(
    -x PATH -x LD_LIBRARY_PATH -x CUDA_VISIBLE_DEVICES
    -x NCCL_DEBUG -x NCCL_BUFFSIZE -x COCCL_ENABLE
  )
  export COCCL_ENABLE="$enabled"
  if [[ "$enabled" == 1 ]]; then
    export COCCL_CONFIG_FILE="$config"
    environment+=(-x COCCL_CONFIG_FILE)
  else
    unset COCCL_CONFIG_FILE
  fi
  echo "========== $title =========="
  "${launcher[@]}" "${environment[@]}" "$tests/$executable" \
    -b "$begin" -e "$end" -f 2 -t 1 -g 1 \
    -w "$warmup" -n "$iterations" -c 0
}

run_test 'alltoall native' alltoall_p2p_perf 1MB 8G 0
run_test 'allgather native' all_gather_perf 1MB 8G 0
run_test 'reducescatter native' reduce_scatter_perf 1MB 8G 0
run_test 'allreduce native' all_reduce_perf 4KB 8G 0

for compressor in sdp4bit zfp; do
  for depth in 1 2 4 8; do
    config=$(config_for "$compressor" "$depth")
    run_test "alltoall $compressor depth=$depth" alltoall_comp_perf 1MB 8G 1 "$config"
    run_test "allgather $compressor depth=$depth" all_gather_comp_perf 1MB 8G 1 "$config"
    run_test "reducescatter oneshot $compressor depth=$depth" \
      reduce_scatter_comp_oneshot_perf 1MB 8G 1 "$config"
    run_test "reducescatter twoshot $compressor depth=$depth" \
      reduce_scatter_comp_twoshot_perf 1MB 8G 1 "$config"
    run_test "allreduce twoshot $compressor depth=$depth" \
      all_reduce_comp_twoshot_perf 1MB 8G 1 "$config"
    run_test "allreduce tripleshot $compressor depth=$depth" \
      all_reduce_comp_tripleshot_perf 1MB 8G 1 "$config"
  done
  run_test "allreduce oneshot $compressor" all_reduce_comp_oneshot_perf \
    4KB 32M 1 "$(config_for "$compressor" 1)"
done
