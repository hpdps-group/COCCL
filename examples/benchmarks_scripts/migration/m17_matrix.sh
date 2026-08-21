#!/usr/bin/env bash
set -euo pipefail

source_root=${1:?usage: m17_matrix.sh SOURCE_ROOT environment|smoke|pack|native|performance|oracle}
mode=${2:?usage: m17_matrix.sh SOURCE_ROOT environment|smoke|pack|native|performance|oracle}
hostfile=${M17_HOSTFILE:?M17_HOSTFILE must name the two-node hostfile}
cuda_root=${M17_CUDA_HOME:-/data/apps/cuda/12.4}
mpi_root=${M17_MPI_HOME:-/data/apps/openmpi/4.1.5_cuda12.8}
current_root=${M17_CURRENT_ROOT:-/data/home/scyb672/run/lxc/COCCL-migrate}
baseline_root=${M17_BASELINE_ROOT:-/data/home/scyb672/run/lxc/COCCL}
temp_root=${M17_TEMP_ROOT:-/data/home/scyb672/run/codex-tmp/coccl-m17-hierarchical}
result_root=${M17_RESULT_ROOT:-$current_root/results/M17}
timeout_seconds=${M17_CASE_TIMEOUT:-1800}
cooldown_seconds=${M17_8G_COOLDOWN_SECONDS:-2}

export CUDA_HOME="$cuda_root"
export PATH="$CUDA_HOME/bin:$mpi_root/bin:$PATH"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_BUFFSIZE=16777216
export NCCL_CUMEM_ENABLE=1

config_root="$result_root/runtime-configs"
mkdir -p "$result_root/raw" "$config_root"

runtime_environment=(
  CUDA_HOME PATH LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES
  NCCL_DEBUG NCCL_BUFFSIZE NCCL_CUMEM_ENABLE NCCL_SOCKET_IFNAME
  COCCL_ENABLE COCCL_CONFIG_FILE
  NCCL_ENABLE_COMPRESS NCCL_COMPRESSORS NCCL_COMPRESSORS_LIB_PATH
  NCCL_COMPRESSORS_CONFIG_PATH NCCL_COMPRESS_ENABLE_THRESHOLD
  NCCL_PIPELINE_DEPTH NCCL_ENABLE_CHECK
  NCCL_ENABLE_ALLREDUCE_COMPRESS NCCL_ALLREDUCE_COMPRESSORS
  NCCL_ALLREDUCE_INTER_COMPRESSORS
  NCCL_ENABLE_REDUCESCATTER_COMPRESS NCCL_REDUCESCATTER_COMPRESSORS
  NCCL_REDUCESCATTER_INTER_COMPRESSORS
)

launcher=(
  "$mpi_root/bin/mpirun" -np 8 --hostfile "$hostfile"
  --map-by ppr:4:node --bind-to none --mca btl '^openib'
)

materialize_current_config() {
  local compressor=$1
  local depth=$2
  local template="$source_root/examples/benchmarks_scripts/configs/m16_${compressor}.toml"
  local output="$config_root/${compressor}-hierarchical-d${depth}.toml"
  sed "s|^library_path = .*|library_path = \"$current_root/build/obj/coccl-extend/compressor_plugin/libcompress\"|" \
    "$template" >"$output"
  printf '\n[pipeline]\ndepth = %s\n' "$depth" >>"$output"
  printf '\n[autotune]\nenabled = false\n' >>"$output"
  printf '%s\n' "$output"
}

clear_runtime() {
  unset COCCL_ENABLE COCCL_CONFIG_FILE NCCL_ENABLE_COMPRESS NCCL_COMPRESSORS \
    NCCL_COMPRESSORS_LIB_PATH NCCL_COMPRESSORS_CONFIG_PATH \
    NCCL_COMPRESS_ENABLE_THRESHOLD NCCL_PIPELINE_DEPTH NCCL_ENABLE_CHECK \
    NCCL_ENABLE_ALLREDUCE_COMPRESS NCCL_ALLREDUCE_COMPRESSORS \
    NCCL_ALLREDUCE_INTER_COMPRESSORS \
    NCCL_ENABLE_REDUCESCATTER_COMPRESS NCCL_REDUCESCATTER_COMPRESSORS \
    NCCL_REDUCESCATTER_INTER_COMPRESSORS
}

configure_current() {
  local compressor=$1
  local depth=$2
  clear_runtime
  export COCCL_ENABLE=1
  export COCCL_CONFIG_FILE
  COCCL_CONFIG_FILE=$(materialize_current_config "$compressor" "$depth")
  export LD_LIBRARY_PATH="$current_root/build/lib:$current_root/build/obj/coccl-extend/compressor_plugin/libcompress:$cuda_root/lib64:$mpi_root/lib"
}

configure_baseline() {
  local operation=$1
  local compressor=$2
  local depth=$3
  local legacy_name=$compressor
  [[ "$compressor" != zfp ]] || legacy_name=cuzfp
  clear_runtime
  export LD_LIBRARY_PATH="$baseline_root/build/lib:$baseline_root/build/obj/device/compress/libcompress:$cuda_root/lib64:$mpi_root/lib"
  export NCCL_ENABLE_COMPRESS=1
  export NCCL_COMPRESSORS="$legacy_name"
  export NCCL_COMPRESSORS_LIB_PATH="$baseline_root/build/obj/device/compress/libcompress"
  export NCCL_COMPRESSORS_CONFIG_PATH="$temp_root/correctness/legacy-configs"
  export NCCL_COMPRESS_ENABLE_THRESHOLD=1048576
  export NCCL_PIPELINE_DEPTH="$depth"
  export NCCL_ENABLE_CHECK=0
  if [[ "$operation" == reducescatter ]]; then
    export NCCL_ENABLE_REDUCESCATTER_COMPRESS=1
    export NCCL_REDUCESCATTER_COMPRESSORS="$legacy_name"
    [[ "$compressor" != sdp4bit && "$compressor" != zfp ]] || \
      export NCCL_REDUCESCATTER_INTER_COMPRESSORS="$legacy_name"
  else
    export NCCL_ENABLE_ALLREDUCE_COMPRESS=1
    export NCCL_ALLREDUCE_COMPRESSORS="$legacy_name"
    [[ "$compressor" != sdp4bit ]] || \
      export NCCL_ALLREDUCE_INTER_COMPRESSORS="$legacy_name"
  fi
}

binary_path() {
  local implementation=$1
  local operation=$2
  local algorithm=$3
  if [[ "$implementation" == native ]]; then
    if [[ "$operation" == reducescatter ]]; then
      printf '%s\n' "$temp_root/current-perf/reduce_scatter_perf"
    else
      printf '%s\n' "$temp_root/current-perf/all_reduce_perf"
    fi
    return
  fi
  if [[ "$implementation" == baseline ]]; then
    if [[ "$operation" == reducescatter ]]; then
      printf '%s\n' "$temp_root/baseline-perf/reduce_scatter_comp_twoshot_tl_overlap_perf"
    else
      printf '%s\n' "$temp_root/baseline-perf/all_reduce_comp_tripleshot_tl_overlap_perf"
    fi
    return
  fi
  if [[ "$operation" == reducescatter ]]; then
    printf '%s\n' "$temp_root/current-perf/reduce_scatter_comp_${algorithm}_perf"
  else
    printf '%s\n' "$temp_root/current-perf/all_reduce_comp_${algorithm}_perf"
  fi
}

run_case() {
  local phase=$1
  local implementation=$2
  local operation=$3
  local algorithm=$4
  local compressor=$5
  local depth=$6
  local bytes=$7
  local stem="${phase}__${implementation}__${operation}__${algorithm}__${compressor}__d${depth}__b${bytes}"
  local raw_root="$result_root/raw/$phase"
  local log="$raw_root/$stem.log"
  local marker="$raw_root/$stem.ok"
  local status_file="$raw_root/$stem.status"
  local command_file="$raw_root/$stem.command"
  mkdir -p "$raw_root"
  if [[ -n ${M17_ONLY_IMPLEMENTATION:-} &&
        "$implementation" != "$M17_ONLY_IMPLEMENTATION" ]]; then
    return
  fi
  if [[ -n ${M17_ONLY_COMPRESSOR:-} &&
        "$compressor" != "$M17_ONLY_COMPRESSOR" ]]; then
    return
  fi
  if [[ -n ${M17_ONLY_STEM:-} && "$stem" != "$M17_ONLY_STEM" ]]; then
    return
  fi
  if [[ -f "$marker" && ${M17_FORCE:-0} != 1 ]]; then
    return
  fi
  rm -f "$marker" "$raw_root/$stem.failed"

  if [[ "$implementation" == current ]]; then
    configure_current "$compressor" "$depth"
  elif [[ "$implementation" == native ]]; then
    clear_runtime
    export COCCL_ENABLE=0
    export LD_LIBRARY_PATH="$current_root/build/lib:$cuda_root/lib64:$mpi_root/lib"
  else
    configure_baseline "$operation" "$compressor" "$depth"
  fi

  local case_launcher=("${launcher[@]}")
  local variable
  for variable in "${runtime_environment[@]}"; do
    [[ -v $variable ]] && case_launcher+=(-x "$variable")
  done
  local binary
  binary=$(binary_path "$implementation" "$operation" "$algorithm")
  local command=(
    timeout --signal=TERM --kill-after=30s "$timeout_seconds"
    "${case_launcher[@]}" "$binary"
    -b "$bytes" -e "$bytes" -f 2 -t 1 -g 1
    -d float -o sum -w 20 -n 30 -c 0
  )
  printf '%q ' "${command[@]}" >"$command_file"
  printf '\n' >>"$command_file"
  printf '[%s] %s\n' "$(date -Is)" "$stem"

  set +e
  "${command[@]}" >"$log" 2>&1
  local status=$?
  set -e
  printf '%s\n' "$status" >"$status_file"
  if grep -Eq '^[[:space:]]*[0-9]+[[:space:]]+' "$log"; then
    touch "$marker"
  else
    touch "$raw_root/$stem.failed"
    return 1
  fi
  if (( bytes == 8589934592 && cooldown_seconds > 0 )); then
    sleep "$cooldown_seconds"
  fi
}

run_pack() {
  local bytes=$1
  local depth=$2
  local pack_mode=$3
  local stem="pack__${pack_mode}__d${depth}__b${bytes}"
  local raw_root="$result_root/raw/pack"
  local log="$raw_root/$stem.log"
  local marker="$raw_root/$stem.ok"
  mkdir -p "$raw_root"
  if [[ -f "$marker" && ${M17_FORCE:-0} != 1 ]]; then
    return
  fi
  local command=(
    timeout 600 "$mpi_root/bin/mpirun" -np 1 --hostfile "$hostfile"
    --map-by ppr:1:node --bind-to none --mca btl '^openib'
    -x "LD_LIBRARY_PATH=$current_root/build/lib:$cuda_root/lib64"
    -x CUDA_VISIBLE_DEVICES
    "$temp_root/layout/coccl_m5_pipeline_layout_test"
    --benchmark "$bytes" "$depth" "$pack_mode"
  )
  printf '[%s] %s\n' "$(date -Is)" "$stem"
  "${command[@]}" >"$log" 2>&1
  grep -q '^bytes,chunks,depth,mode,time_us$' "$log"
  touch "$marker"
}

run_environment() {
  {
    printf 'M17 2x4 A800 environment\n'
    printf 'hostfile=%s\n' "$hostfile"
    printf 'current_commit=%s\n' "$(git -C "$source_root" rev-parse HEAD)"
    printf 'baseline_commit=%s\n' "$(git -C "$baseline_root" rev-parse HEAD)"
    "$mpi_root/bin/mpirun" -np 2 --hostfile "$hostfile" \
      --map-by ppr:1:node --bind-to none --mca btl '^openib' hostname
    "$mpi_root/bin/mpirun" -np 2 --hostfile "$hostfile" \
      --map-by ppr:1:node --bind-to none --mca btl '^openib' \
      nvidia-smi --query-gpu=name,memory.total,driver_version \
      --format=csv,noheader
    "$CUDA_HOME/bin/nvcc" --version
  } >"$result_root/environment.txt" 2>&1
}

run_performance() {
  local implementation operation algorithm compressor depth bytes
  for implementation in baseline current; do
    for operation in reducescatter allreduce; do
      [[ "$operation" == reducescatter ]] && algorithm=twoshot || algorithm=tripleshot
      for compressor in sdp4bit zfp; do
        for depth in 1 2 4 8; do
          for ((bytes=1048576; bytes<=8589934592; bytes*=2)); do
            run_case performance "$implementation" "$operation" \
              "$algorithm" "$compressor" "$depth" "$bytes"
          done
        done
      done
    done
  done
}

run_native() {
  local operation bytes
  for operation in reducescatter allreduce; do
    for ((bytes=1048576; bytes<=8589934592; bytes*=2)); do
      run_case performance native "$operation" native native 1 "$bytes"
    done
  done
}

run_oracle() {
  local compressor depth bytes algorithm
  local sizes=(4194304 33554432 67108864 536870912 1073741824 8589934592)
  for compressor in sdp4bit zfp; do
    for depth in 1 4; do
      for bytes in "${sizes[@]}"; do
        run_case oracle current reducescatter oneshot \
          "$compressor" "$depth" "$bytes"
        run_case oracle current allreduce twoshot \
          "$compressor" "$depth" "$bytes"
        if (( bytes <= 33554432 )); then
          run_case oracle current allreduce oneshot \
            "$compressor" "$depth" "$bytes"
        fi
      done
    done
  done
}

case "$mode" in
  environment)
    run_environment
    ;;
  smoke)
    run_case smoke baseline reducescatter twoshot sdp4bit 1 1048576
    run_case smoke baseline allreduce tripleshot zfp 1 1048576
    run_case smoke current reducescatter twoshot sdp4bit 1 1048576
    run_case smoke current allreduce tripleshot zfp 1 1048576
    ;;
  pack)
    for bytes in 536870912 1073741824 2147483648 4294967296 8589934592; do
      for depth in 1 2 4 8; do
        for pack_mode in plain-pack plain-unpack swizzle; do
          run_pack "$bytes" "$depth" "$pack_mode"
        done
      done
    done
    ;;
  native)
    run_native
    ;;
  performance)
    run_performance
    ;;
  oracle)
    run_oracle
    ;;
  *)
    printf 'unsupported M17 mode: %s\n' "$mode" >&2
    exit 2
    ;;
esac
