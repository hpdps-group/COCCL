#!/usr/bin/env bash
set -euo pipefail

source_root=${1:?usage: m16_matrix.sh SOURCE_ROOT IMPLEMENTATION SUITE}
implementation=${2:?usage: m16_matrix.sh SOURCE_ROOT IMPLEMENTATION SUITE}
suite=${3:?usage: m16_matrix.sh SOURCE_ROOT IMPLEMENTATION SUITE}

cuda_root=${M16_CUDA_HOME:-/data/apps/cuda/12.4}
mpi_root=${M16_MPI_HOME:-/data/apps/openmpi/4.1.5_cuda12.8}
current_root=${M16_CURRENT_ROOT:-/data/home/scyb672/run/lxc/COCCL-migrate}
baseline_root=${M16_BASELINE_ROOT:-/data/home/scyb672/run/lxc/COCCL}
temp_root=${M16_TEMP_ROOT:-/data/home/scyb672/run/codex-tmp/coccl-m16-correctness}
result_root=${M16_RESULT_ROOT:-$current_root/results/M16}
timeout_seconds=${M16_CASE_TIMEOUT:-300}

case "$implementation:$suite" in
  current:single|current:hierarchical|current:subadd|\
  baseline:single|baseline:hierarchical|baseline:subadd) ;;
  *) printf 'unsupported implementation/suite: %s/%s\n' \
       "$implementation" "$suite" >&2; exit 2 ;;
esac

if [[ "$suite" == hierarchical ]]; then
  ranks=${M16_RANKS:-8}
  topology=${M16_TOPOLOGY:-2x4}
else
  ranks=${M16_RANKS:-4}
  topology=${M16_TOPOLOGY:-single-node}
fi

export CUDA_HOME="$cuda_root"
export PATH="$CUDA_HOME/bin:$mpi_root/bin:$PATH"
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_BUFFSIZE=16777216
export NCCL_CUMEM_ENABLE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

raw_root="$result_root/raw/$implementation/$suite"
config_root="$result_root/runtime-configs"
mkdir -p "$raw_root" "$config_root"

launcher=("$mpi_root/bin/mpirun" -np "$ranks" --bind-to none)
if [[ -n ${M16_HOSTFILE:-} ]]; then
  launcher+=(--hostfile "$M16_HOSTFILE" --map-by ppr:4:node)
fi

runtime_environment=(
  CUDA_HOME PATH LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES
  NCCL_DEBUG NCCL_BUFFSIZE NCCL_CUMEM_ENABLE
  COCCL_ENABLE COCCL_CONFIG_FILE
  NCCL_ENABLE_COMPRESS NCCL_COMPRESSORS NCCL_COMPRESSORS_LIB_PATH
  NCCL_COMPRESSORS_CONFIG_PATH NCCL_COMPRESS_ENABLE_THRESHOLD
  NCCL_PIPELINE_DEPTH NCCL_ENABLE_CHECK
  NCCL_ENABLE_ALLTOALL_COMPRESS NCCL_ALLTOALL_COMPRESSORS
  NCCL_ENABLE_ALLGATHER_COMPRESS NCCL_ALLGATHER_COMPRESSORS
  NCCL_ENABLE_ALLREDUCE_COMPRESS NCCL_ALLREDUCE_COMPRESSORS
  NCCL_ENABLE_REDUCESCATTER_COMPRESS NCCL_REDUCESCATTER_COMPRESSORS
  NCCL_ENABLE_SENDRECV_COMPRESS NCCL_SENDRECV_COMPRESSORS
  NCCL_SENDRECV_BWD_COMPRESSORS NCCL_ALLGATHER_INTER_COMPRESSORS
  NCCL_ALLREDUCE_INTER_COMPRESSORS NCCL_REDUCESCATTER_INTER_COMPRESSORS
)

materialize_current_config() {
  local compressor=$1
  local depth=$2
  local template="$source_root/examples/benchmarks_scripts/configs/m16_${compressor}.toml"
  [[ "$suite" != subadd ]] || \
    template="$source_root/examples/benchmarks_scripts/configs/m16_sdp4bit_subadd.toml"
  local output="$config_root/${compressor}-${suite}-d${depth}.toml"
  sed "s|^library_path = .*|library_path = \"$current_root/build/obj/coccl-extend/compressor_plugin/libcompress\"|" \
    "$template" >"$output"
  printf '\n[pipeline]\ndepth = %s\n' "$depth" >>"$output"
  printf '\n[autotune]\nenabled = false\n' >>"$output"
  printf '%s\n' "$output"
}

configure_current() {
  local compressor=$1
  local depth=$2
  export COCCL_ENABLE=1
  export COCCL_CONFIG_FILE
  COCCL_CONFIG_FILE=$(materialize_current_config "$compressor" "$depth")
  export LD_LIBRARY_PATH="$current_root/build/lib:$current_root/build/obj/coccl-extend/compressor_plugin/libcompress:$cuda_root/lib64:$mpi_root/lib:${LD_LIBRARY_PATH:-}"
}

configure_baseline() {
  local compressor=$1
  local legacy_name=$compressor
  [[ "$compressor" != zfp ]] || legacy_name=cuzfp
  local config_root="$temp_root/legacy-configs"
  [[ "$suite" != subadd ]] || config_root="$temp_root/legacy-subadd-configs"

  unset COCCL_ENABLE COCCL_CONFIG_FILE
  export LD_LIBRARY_PATH="$baseline_root/build/lib:$baseline_root/build/obj/device/compress/libcompress:$cuda_root/lib64:$mpi_root/lib:${LD_LIBRARY_PATH:-}"
  export NCCL_ENABLE_COMPRESS=1
  export NCCL_COMPRESSORS="$legacy_name"
  export NCCL_COMPRESSORS_LIB_PATH="$baseline_root/build/obj/device/compress/libcompress"
  export NCCL_COMPRESSORS_CONFIG_PATH="$config_root"
  export NCCL_COMPRESS_ENABLE_THRESHOLD=1048576
  export NCCL_PIPELINE_DEPTH=1
  export NCCL_ENABLE_CHECK=0
  export NCCL_ENABLE_ALLTOALL_COMPRESS=1
  export NCCL_ALLTOALL_COMPRESSORS="$legacy_name"
  export NCCL_ENABLE_ALLGATHER_COMPRESS=1
  export NCCL_ALLGATHER_COMPRESSORS="$legacy_name"
  export NCCL_ENABLE_ALLREDUCE_COMPRESS=1
  export NCCL_ALLREDUCE_COMPRESSORS="$legacy_name"
  export NCCL_ENABLE_REDUCESCATTER_COMPRESS=1
  export NCCL_REDUCESCATTER_COMPRESSORS="$legacy_name"
  export NCCL_ENABLE_SENDRECV_COMPRESS=1
  export NCCL_SENDRECV_COMPRESSORS="$legacy_name"
  export NCCL_SENDRECV_BWD_COMPRESSORS="$legacy_name"
  unset NCCL_ALLGATHER_INTER_COMPRESSORS \
    NCCL_ALLREDUCE_INTER_COMPRESSORS NCCL_REDUCESCATTER_INTER_COMPRESSORS
  if [[ "$compressor" == sdp4bit ]]; then
    export NCCL_ALLGATHER_INTER_COMPRESSORS=sdp4bit
    export NCCL_ALLREDUCE_INTER_COMPRESSORS=sdp4bit
    export NCCL_REDUCESCATTER_INTER_COMPRESSORS=sdp4bit
  fi
}

run_case() {
  local compressor=$1
  local depth=$2
  local datatype=$3
  local stem="${topology}__${suite}__${compressor}__${datatype}__d${depth}"
  local log="$raw_root/$stem.log"
  local marker="$raw_root/$stem.ok"
  if [[ -n ${M16_ONLY_STEM:-} && "$stem" != "$M16_ONLY_STEM" ]]; then
    return
  fi
  if [[ -f "$marker" && ${M16_FORCE:-0} != 1 ]]; then
    return
  fi
  [[ ${M16_FORCE:-0} != 1 ]] || rm -f "$marker"

  local binary="$temp_root/$implementation/coccl_m16_correctness"
  if [[ "$implementation" == current ]]; then
    configure_current "$compressor" "$depth"
  else
    configure_baseline "$compressor"
  fi
  local case_launcher=("${launcher[@]}")
  local variable
  for variable in "${runtime_environment[@]}"; do
    [[ -v $variable ]] && case_launcher+=(-x "$variable")
  done
  local command=(
    timeout --signal=TERM --kill-after=30s "$timeout_seconds"
    "${case_launcher[@]}" "$binary"
    --suite "$suite" --compressor "$compressor" --datatype "$datatype"
    --topology "$topology" --depth "$depth"
  )
  printf '[%s] %s\n' "$(date -Is)" "$stem"
  "${command[@]}" >"$log" 2>&1
  grep -q '^COCCL_CORRECTNESS ' "$log"
  touch "$marker"
}

if [[ "$implementation" == baseline ]]; then
  compressors=(sdp4bit zfp)
  depths=(1)
else
  compressors=(sdp4bit zfp dietgpu)
  depths=(1 2 4 8)
fi
if [[ "$suite" == subadd ]]; then
  compressors=(sdp4bit)
fi

for compressor in "${compressors[@]}"; do
  for depth in "${depths[@]}"; do
    for datatype in float half bfloat16; do
      run_case "$compressor" "$depth" "$datatype"
    done
  done
done
