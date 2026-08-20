#!/usr/bin/env bash
set -euo pipefail

source_root=${1:?usage: m20_matrix.sh SOURCE_ROOT MODE}
mode=${2:?usage: m20_matrix.sh SOURCE_ROOT MODE}

cuda_root=${M20_CUDA_HOME:-/data/apps/cuda/12.8}
mpi_root=${M20_MPI_HOME:-/data/apps/openmpi/4.1.5_cuda12.8}
current_root=${M20_CURRENT_ROOT:-/data/home/scyb672/run/lxc/COCCL-migrate}
temp_root=${M20_TEMP_ROOT:-/data/home/scyb672/run/codex-tmp/coccl-m20-remainder-slices}
result_root=${M20_RESULT_ROOT:-$current_root/results/M20}
runtime_root=${M20_RUNTIME_ROOT:-/data/home/scyb672/run/codex-tmp/coccl-m19-scope-policy/runtime-build}
plugin_root=${M20_PLUGIN_ROOT:-/data/home/scyb672/run/codex-tmp/coccl-m19-scope-policy/plugins/libcompress}
test_root=${M20_TEST_ROOT:-$temp_root/gpu-tests}
timeout_seconds=${M20_CASE_TIMEOUT:-1800}

case "$mode" in
  correctness-single|correctness-two-node|performance-single|performance-two-node) ;;
  *) printf 'unsupported M20 mode: %s\n' "$mode" >&2; exit 2 ;;
esac

export CUDA_HOME="$cuda_root"
export PATH="$cuda_root/bin:$mpi_root/bin:/usr/local/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$runtime_root/lib:$plugin_root:$cuda_root/lib64:$mpi_root/lib"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_CUMEM_ENABLE=1
export NCCL_BUFFSIZE=16777216
export COCCL_ENABLE=1

config_root="$result_root/runtime-configs"
raw_root="$result_root/raw/$mode"
mkdir -p "$config_root" "$raw_root"

runtime_variables=(
  CUDA_HOME PATH LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES NCCL_DEBUG
  NCCL_CUMEM_ENABLE NCCL_BUFFSIZE COCCL_ENABLE COCCL_CONFIG_FILE
)

if [[ "$mode" == *two-node ]]; then
  hostfile=${M20_HOSTFILE:?M20_HOSTFILE must name the two-node hostfile}
  launcher=(
    "$mpi_root/bin/mpirun" -np 8 --hostfile "$hostfile"
    --map-by ppr:4:node --bind-to none --mca btl '^openib'
  )
  ranks=8
  topology=2x4
else
  launcher=("$mpi_root/bin/mpirun" -np 4 --bind-to none --mca btl '^openib')
  ranks=4
  topology=single-node
fi

write_policy() {
  local file=$1 operation=$2 scope=$3 compressor=$4 subadd=${5:-false}
  local pipeline_depth=${6:-1}
  {
    printf '\n[normal.%s.%s]\ncompressor = "%s"\n' \
      "$operation" "$scope" "$compressor"
    printf '\n[normal.%s.%s.config]\n' "$operation" "$scope"
    if [[ "$subadd" == true ]]; then
      printf 'groupCount = 128\nquantBits = 4\nquantType = "Symmetric"\n'
    else case "$compressor:$operation" in
      sdp4bit:all_to_all|sdp4bit:all_gather)
        printf 'groupCount = 2048\nquantBits = 4\nquantType = "Symmetric"\n'
        ;;
      sdp4bit:*)
        printf 'groupCount = 128\nquantBits = 4\nquantType = "Symmetric"\n'
        printf 'hadamard = false\npipelineSize = 1\n'
        ;;
      zfp:all_to_all) printf 'rate = 4\n' ;;
      zfp:*) printf 'rate = 8\n' ;;
      dietgpu:*) printf 'probBits = 10\n' ;;
    esac
    fi
    [[ "$subadd" != true ]] || \
      printf 'subAdd = true\npipelineSize = %s\n' "$pipeline_depth"
  } >>"$file"
}

materialize_config() {
  local compressor=$1 depth=$2 scope_case=$3
  local file="$config_root/${compressor}-${scope_case}-d${depth}.toml"
  {
    printf 'schema_version = 3\n\n[runtime]\nmode = "normal"\n'
    printf 'compression_threshold_bytes = 1099511627776\n'
    printf '\n[compressor_plugins]\ncompressors = ["%s"]\n' "$compressor"
    printf 'library_path = "%s"\n' "$plugin_root"
  } >"$file"

  case "$scope_case" in
    single)
      for operation in all_to_all all_gather reduce_scatter all_reduce sendrecv; do
        write_policy "$file" "$operation" default "$compressor"
      done
      ;;
    subadd)
      write_policy "$file" all_gather default "$compressor" true "$depth"
      ;;
    default|intra|inter)
      write_policy "$file" reduce_scatter "$scope_case" "$compressor"
      write_policy "$file" all_reduce "$scope_case" "$compressor"
      ;;
  esac
  {
    printf '\n[pipeline]\ndepth = %s\n' "$depth"
    printf '\n[autotune]\nenabled = false\n'
    printf 'reduce_scatter_algorithm = "%s"\n' \
      "$([[ "$topology" == 2x4 ]] && printf twoshot || printf oneshot)"
    printf 'all_reduce_algorithm = "%s"\n' \
      "$([[ "$topology" == 2x4 ]] && printf tripleshot || printf twoshot)"
  } >>"$file"
  printf '%s\n' "$file"
}

case_launcher() {
  CASE_LAUNCHER=("${launcher[@]}")
  local variable
  for variable in "${runtime_variables[@]}"; do
    [[ -v $variable ]] && CASE_LAUNCHER+=(-x "$variable")
  done
}

run_logged() {
  local stem=$1
  shift
  [[ -z ${M20_ONLY_STEM:-} || "$stem" == "$M20_ONLY_STEM" ]] || return 0
  local log="$raw_root/$stem.log"
  local marker="$raw_root/$stem.ok"
  if [[ -f "$marker" && ${M20_FORCE:-0} != 1 ]]; then return; fi
  rm -f "$marker" "$raw_root/$stem.failed"
  printf '[%s] %s\n' "$(date -Is)" "$stem"
  set +e
  timeout --signal=TERM --kill-after=30s "$timeout_seconds" \
    "$@" >"$log" 2>&1
  local status=$?
  set -e
  printf '%s\n' "$status" >"$raw_root/$stem.status"
  if (( status != 0 )); then
    touch "$raw_root/$stem.failed"
    return "$status"
  fi
  touch "$marker"
}

run_correctness() {
  local compressor=$1 datatype=$2 depth=$3 raw_count=$4
  local scope_case=${5:-$([[ "$topology" == 2x4 ]] && printf default || printf single)}
  export COCCL_CONFIG_FILE
  COCCL_CONFIG_FILE=$(materialize_config "$compressor" "$depth" "$scope_case")
  case_launcher
  local suite=$([[ "$topology" == 2x4 ]] && printf hierarchical || printf single)
  local stem="${topology}__${scope_case}__${compressor}__${datatype}__d${depth}__c${raw_count}"
  local command=(
    "${CASE_LAUNCHER[@]}" "$test_root/coccl_m16_correctness"
    --suite "$suite" --compressor "$compressor" --datatype "$datatype"
    --topology "$topology" --depth "$depth" --raw-chunk-elements "$raw_count"
  )
  [[ -z ${M20_OPERATION:-} ]] || command+=(--operation "$M20_OPERATION")
  [[ -z ${M20_ALGORITHM:-} ]] || command+=(--algorithm "$M20_ALGORITHM")
  run_logged "$stem" "${command[@]}"
}

run_subadd() {
  local depth=$1 raw_count=$2
  export COCCL_CONFIG_FILE
  COCCL_CONFIG_FILE=$(materialize_config sdp4bit "$depth" subadd)
  case_launcher
  run_logged "single-node__subadd__sdp4bit__float__d${depth}__c${raw_count}" \
    "${CASE_LAUNCHER[@]}" "$test_root/coccl_m16_correctness" \
    --suite subadd --compressor sdp4bit --datatype float \
    --topology single-node --depth "$depth" --raw-chunk-elements "$raw_count"
}

perf_binary() {
  case "$1:$2" in
    alltoall:compressed) printf '%s/alltoall_comp_perf\n' "$test_root" ;;
    allgather:compressed) printf '%s/all_gather_comp_perf\n' "$test_root" ;;
    reducescatter:compressed)
      if [[ "$topology" == 2x4 ]]; then
        printf '%s/reduce_scatter_comp_twoshot_perf\n' "$test_root"
      else
        printf '%s/reduce_scatter_comp_oneshot_perf\n' "$test_root"
      fi
      ;;
    allreduce:compressed)
      if [[ "$topology" == 2x4 ]]; then
        printf '%s/all_reduce_comp_tripleshot_perf\n' "$test_root"
      else
        printf '%s/all_reduce_comp_twoshot_perf\n' "$test_root"
      fi
      ;;
  esac
}

run_performance() {
  local compressor=$1 operation=$2 depth=$3 target_bytes=$4 remainder=$5 variant=$6
  local type_bytes=4
  local target_elements=$((target_bytes / ranks / type_bytes))
  local quotient=$(((target_elements - remainder) / depth))
  quotient=$((quotient - quotient % (16 / type_bytes)))
  local raw_count=$((depth * quotient + remainder))
  [[ "$variant" != control* ]] || raw_count=$((raw_count - remainder))
  local actual_bytes=$((ranks * type_bytes * raw_count))
  local effective_depth=$depth
  [[ "$variant" != depth1* ]] || effective_depth=1
  export COCCL_CONFIG_FILE
  COCCL_CONFIG_FILE=$(materialize_config "$compressor" "$effective_depth" \
    "$([[ "$topology" == 2x4 ]] && printf default || printf single)")
  case_launcher
  local binary
  binary=$(perf_binary "$operation" compressed)
  local stem="${topology}__${compressor}__${operation}__d${depth}__${variant}__c${raw_count}__b${actual_bytes}"
  run_logged "$stem" "${CASE_LAUNCHER[@]}" "$binary" \
    -b "$actual_bytes" -e "$actual_bytes" -f 2 -t 1 -g 1 \
    -d float -o sum -w 20 -n 30 -c 0
  (( target_bytes < 8589934592 )) || sleep 2
}

correctness_cases=(2:4097 4:4097 4:4099 8:4097 8:4103)

case "$mode" in
  correctness-single)
    for item in "${correctness_cases[@]}"; do
      IFS=: read -r depth raw_count <<<"$item"
      for datatype in float half bfloat16; do
        run_correctness sdp4bit "$datatype" "$depth" "$raw_count"
        run_correctness zfp "$datatype" "$depth" "$raw_count"
      done
      run_correctness dietgpu float "$depth" "$raw_count"
    done
    run_subadd 4 4097
    ;;
  correctness-two-node)
    for item in "${correctness_cases[@]}"; do
      IFS=: read -r depth raw_count <<<"$item"
      for datatype in float half bfloat16; do
        run_correctness sdp4bit "$datatype" "$depth" "$raw_count" default
        run_correctness zfp "$datatype" "$depth" "$raw_count" default
      done
    done
    run_correctness sdp4bit float 4 4097 intra
    run_correctness sdp4bit float 4 4097 inter
    run_correctness dietgpu float 4 4097 default
    ;;
  performance-single|performance-two-node)
    read -r -a compressors <<<"${M20_COMPRESSORS:-sdp4bit zfp}"
    read -r -a depths <<<"${M20_DEPTHS:-2 4 8}"
    read -r -a targets <<<"${M20_TARGETS:-33554432 100663296 536870912 1073741824 4294967296 8589934592}"
    if [[ "$topology" == 2x4 ]]; then
      operations=(reducescatter allreduce)
      targets=(100663296 536870912 1073741824)
    else
      operations=(alltoall allgather reducescatter allreduce)
    fi
    [[ -z ${M20_OPERATIONS:-} ]] || read -r -a operations <<<"$M20_OPERATIONS"
    for compressor in "${compressors[@]}"; do
      for operation in "${operations[@]}"; do
        for depth in "${depths[@]}"; do
          for target in "${targets[@]}"; do
            run_performance "$compressor" "$operation" "$depth" "$target" 1 remainder
            run_performance "$compressor" "$operation" "$depth" "$target" 1 control
            run_performance "$compressor" "$operation" "$depth" "$target" 1 depth1
          done
          for target in 33554432 536870912 8589934592; do
            [[ "$topology" != 2x4 ]] || continue
            run_performance "$compressor" "$operation" "$depth" "$target" $((depth - 1)) remainder-max
            run_performance "$compressor" "$operation" "$depth" "$target" $((depth - 1)) control-max
            run_performance "$compressor" "$operation" "$depth" "$target" $((depth - 1)) depth1-max
          done
        done
      done
    done
    ;;
esac
