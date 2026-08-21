#!/usr/bin/env bash
set -euo pipefail

mode=${1:-single}
case "$mode" in
  single|multi) ;;
  *) echo "usage: $0 <single|multi>" >&2; exit 2 ;;
esac

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
coccl_root=${COCCL_ROOT:-$(cd "$script_dir/../.." && pwd)}
cuda_home=${CUDA_HOME:-}
mpi_home=${MPI_HOME:-}
gpus_per_node=${GPUS_PER_NODE:-4}
nodes=${NNODES:-2}
hostfile=${HOSTFILE:-}
warmup=${COCCL_BENCH_WARMUP:-20}
iterations=${COCCL_BENCH_ITERATIONS:-30}
pause_seconds=${COCCL_BENCH_PAUSE_SECONDS:-2}
collective_begin=${COCCL_BENCH_BEGIN:-1MB}
collective_end=${COCCL_BENCH_END:-8G}
allreduce_begin=${COCCL_BENCH_ALLREDUCE_BEGIN:-4KB}
oneshot_begin=${COCCL_BENCH_ONESHOT_BEGIN:-4KB}
oneshot_end=${COCCL_BENCH_ONESHOT_END:-32M}
tests="$coccl_root/tests/coccl-tests/build"
plugin_root="$coccl_root/build/obj/coccl-extend/compressor_plugin/libcompress"
config_root=${COCCL_BENCH_CONFIG_ROOT:-$coccl_root/build/examples/benchmark-configs/$mode}
read -r -a compressors <<<"${COCCL_BENCH_COMPRESSORS:-sdp4bit zfp}"
read -r -a depths <<<"${COCCL_BENCH_DEPTHS:-1 2 4 8}"

: "${cuda_home:?Set CUDA_HOME before running the benchmark}"
: "${mpi_home:?Set MPI_HOME before running the benchmark}"
if [[ "$mode" == multi ]]; then
  : "${hostfile:?Set HOSTFILE for a multi-node benchmark}"
fi

mkdir -p "$config_root"
export CUDA_HOME="$cuda_home"
export NCCL_HOME="$coccl_root/build"
export PATH="$CUDA_HOME/bin:$MPI_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$NCCL_HOME/lib:$plugin_root:$CUDA_HOME/lib64:$MPI_HOME/lib:${LD_LIBRARY_PATH:-}"
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_BUFFSIZE=${NCCL_BUFFSIZE:-16777216}

config_for() {
  local compressor=$1 depth=$2
  local template="$script_dir/configs/m16_${compressor}.toml"
  local output="$config_root/${compressor}-d${depth}.toml"

  sed "s|^library_path = .*|library_path = \"$plugin_root\"|" \
    "$template" >"$output"
  printf '\n[pipeline]\ndepth = %s\n' "$depth" >>"$output"
  printf '\n[autotune]\nenabled = false\n' >>"$output"
  printf '%s\n' "$output"
}

run_case() {
  local title=$1 executable=$2 begin=$3 end=$4 enabled=$5 config=${6:-}
  local arguments=(
    -b "$begin" -e "$end" -f 2 -g 1
    -w "$warmup" -n "$iterations" -c 0
  )

  export COCCL_ENABLE="$enabled"
  if [[ "$enabled" == 1 ]]; then
    export COCCL_CONFIG_FILE="$config"
  else
    unset COCCL_CONFIG_FILE
  fi

  echo "========== $title =========="
  if [[ "$mode" == single ]]; then
    "$tests/$executable" "${arguments[@]}" -t "$gpus_per_node"
  else
    local total_ranks=$((nodes * gpus_per_node))
    local exports=(
      -x PATH -x LD_LIBRARY_PATH -x NCCL_DEBUG -x NCCL_BUFFSIZE
      -x COCCL_ENABLE
    )
    [[ "$enabled" == 1 ]] && exports+=(-x COCCL_CONFIG_FILE)
    [[ -n ${CUDA_VISIBLE_DEVICES:-} ]] && exports+=(-x CUDA_VISIBLE_DEVICES)
    [[ -n ${NCCL_IB_DISABLE:-} ]] && exports+=(-x NCCL_IB_DISABLE)
    [[ -n ${NCCL_IB_HCA:-} ]] && exports+=(-x NCCL_IB_HCA)
    [[ -n ${NCCL_SOCKET_IFNAME:-} ]] && exports+=(-x NCCL_SOCKET_IFNAME)
    [[ -n ${NCCL_LOCAL_REGISTER:-} ]] && exports+=(-x NCCL_LOCAL_REGISTER)

    "$mpi_home/bin/mpirun" -np "$total_ranks" --hostfile "$hostfile" \
      --map-by "ppr:${gpus_per_node}:node" --bind-to none --mca btl '^openib' \
      "${exports[@]}" "$tests/$executable" "${arguments[@]}" -t 1
  fi

  if [[ "$end" == 8G && "$pause_seconds" != 0 ]]; then
    sleep "$pause_seconds"
  fi
}

run_compressed() {
  local title=$1 executable=$2 compressor=$3 depth=$4 begin=$5 end=$6
  run_case "$title $compressor depth=$depth" "$executable" "$begin" "$end" \
    1 "$(config_for "$compressor" "$depth")"
}

echo "COCCL_ROOT=$coccl_root mode=$mode GPUs/node=$gpus_per_node nodes=$nodes"
echo "compressors=${compressors[*]} depths=${depths[*]} warmup=$warmup iterations=$iterations"

run_case "AllToAll native" alltoall_p2p_perf "$collective_begin" "$collective_end" 0
run_case "AllGather native" all_gather_perf "$collective_begin" "$collective_end" 0
run_case "ReduceScatter native" reduce_scatter_perf "$collective_begin" "$collective_end" 0
run_case "AllReduce native" all_reduce_perf "$allreduce_begin" "$collective_end" 0

for compressor in "${compressors[@]}"; do
  for depth in "${depths[@]}"; do
    run_compressed AllToAll alltoall_comp_perf "$compressor" "$depth" \
      "$collective_begin" "$collective_end"
    run_compressed AllGather all_gather_comp_perf "$compressor" "$depth" \
      "$collective_begin" "$collective_end"
    run_compressed "ReduceScatter OneShot" reduce_scatter_comp_oneshot_perf \
      "$compressor" "$depth" "$collective_begin" "$collective_end"
    run_compressed "AllReduce TwoShot" all_reduce_comp_twoshot_perf \
      "$compressor" "$depth" "$collective_begin" "$collective_end"

    if [[ "$mode" == multi ]]; then
      run_compressed "ReduceScatter TwoShot" reduce_scatter_comp_twoshot_perf \
        "$compressor" "$depth" "$collective_begin" "$collective_end"
      run_compressed "AllReduce TripleShot" all_reduce_comp_tripleshot_perf \
        "$compressor" "$depth" "$collective_begin" "$collective_end"
    fi
  done

  run_compressed "AllReduce OneShot" all_reduce_comp_oneshot_perf \
    "$compressor" 1 "$oneshot_begin" "$oneshot_end"
done
