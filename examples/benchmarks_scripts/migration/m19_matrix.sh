#!/usr/bin/env bash
set -euo pipefail

source_root=${1:?usage: m19_matrix.sh SOURCE_ROOT MODE}
mode=${2:?usage: m19_matrix.sh SOURCE_ROOT environment|smoke|correctness|performance|dietgpu}
hostfile=${M19_HOSTFILE:?M19_HOSTFILE must name the two-node hostfile}
cuda_root=${M19_CUDA_HOME:-/data/apps/cuda/12.4}
mpi_root=${M19_MPI_HOME:-/data/apps/openmpi/4.1.5_cuda12.8}
current_root=${M19_CURRENT_ROOT:-/data/home/scyb672/run/lxc/COCCL-migrate}
temp_root=${M19_TEMP_ROOT:-/data/home/scyb672/run/codex-tmp/coccl-m19-scope-policy}
result_root=${M19_RESULT_ROOT:-$current_root/results/M19}
runtime_root=${M19_RUNTIME_ROOT:-$temp_root/runtime-build}
plugin_root=${M19_PLUGIN_ROOT:-$temp_root/plugins/libcompress}
test_root=${M19_TEST_ROOT:-$temp_root/gpu-tests}
timeout_seconds=${M19_CASE_TIMEOUT:-1800}

case "$mode" in
  environment|smoke|correctness|performance|dietgpu) ;;
  *) printf 'unsupported M19 mode: %s\n' "$mode" >&2; exit 2 ;;
esac

export CUDA_HOME="$cuda_root"
export PATH="$cuda_root/bin:$mpi_root/bin:/usr/local/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$runtime_root/lib:$plugin_root:$cuda_root/lib64:$mpi_root/lib"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_CUMEM_ENABLE=1
export NCCL_BUFFSIZE=16777216

config_root="$result_root/runtime-configs"
raw_root="$result_root/raw"
memory_root="$result_root/memory-samples"
mkdir -p "$config_root" "$raw_root/$mode" "$memory_root"

launcher=(
  "$mpi_root/bin/mpirun" -np 8 --hostfile "$hostfile"
  --map-by ppr:4:node --bind-to none --mca btl '^openib'
)
runtime_variables=(
  CUDA_HOME PATH LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES
  NCCL_DEBUG NCCL_CUMEM_ENABLE NCCL_BUFFSIZE
  COCCL_ENABLE COCCL_CONFIG_FILE
)

write_scope() {
  local file=$1 operation=$2 scope=$3 spec=$4
  [[ -n "$spec" ]] || return
  {
    printf '\n[normal.%s.%s]\n' "$operation" "$scope"
    if [[ "$spec" == disabled ]]; then
      printf 'enabled = false\n'
      return
    fi
    IFS=: read -r codec first second <<<"$spec"
    printf 'compressor = "%s"\n' "$codec"
    printf '\n[normal.%s.%s.config]\n' "$operation" "$scope"
    case "$codec" in
      sdp4bit)
        printf 'quantBits = %s\ngroupCount = %s\n' "$first" "$second"
        printf 'quantType = "Symmetric"\nhadamard = false\npipelineSize = 1\n'
        ;;
      zfp) printf 'rate = %s\n' "$first" ;;
      dietgpu) printf 'probBits = %s\n' "$first" ;;
    esac
  } >>"$file"
}

case_scopes() {
  local case_name=$1
  default_scope= intra_scope= inter_scope=
  plugins='["sdp4bit"]'
  case "$case_name" in
    default-sdp) default_scope=sdp4bit:4:128 ;;
    default-zfp) default_scope=zfp:8; plugins='["zfp"]' ;;
    intra-sdp) intra_scope=sdp4bit:4:128 ;;
    inter-sdp) inter_scope=sdp4bit:4:128 ;;
    split-sdp)
      intra_scope=sdp4bit:8:1024
      inter_scope=sdp4bit:4:512
      ;;
    default-intra-override)
      default_scope=sdp4bit:4:128
      intra_scope=sdp4bit:8:1024
      ;;
    default-inter-override)
      default_scope=sdp4bit:8:1024
      inter_scope=sdp4bit:4:512
      ;;
    disable-intra)
      default_scope=sdp4bit:8:1024
      intra_scope=disabled
      inter_scope=sdp4bit:4:512
      ;;
    disable-inter)
      default_scope=sdp4bit:4:128
      intra_scope=sdp4bit:8:1024
      inter_scope=disabled
      ;;
    mixed-sdp-zfp)
      intra_scope=sdp4bit:8:1024
      inter_scope=zfp:8
      plugins='["sdp4bit", "zfp"]'
      ;;
    dietgpu-inter)
      default_scope=disabled
      intra_scope=disabled
      inter_scope=dietgpu:10
      plugins='["dietgpu"]'
      ;;
    *) printf 'unknown M19 config case: %s\n' "$case_name" >&2; exit 2 ;;
  esac
}

materialize_config() {
  local case_name=$1 depth=$2
  local file="$config_root/${case_name}-d${depth}.toml"
  case_scopes "$case_name"
  {
    printf 'schema_version = 3\n\n[runtime]\nmode = "normal"\n'
    printf 'compression_threshold_bytes = 1048576\n'
    printf '\n[compressor_plugins]\ncompressors = %s\n' "$plugins"
    printf 'library_path = "%s"\n' "$plugin_root"
  } >"$file"
  for operation in reduce_scatter all_reduce; do
    write_scope "$file" "$operation" default "$default_scope"
    write_scope "$file" "$operation" intra "$intra_scope"
    write_scope "$file" "$operation" inter "$inter_scope"
  done
  if [[ "$case_name" == dietgpu-inter ]]; then
    write_scope "$file" sendrecv default "$default_scope"
    write_scope "$file" sendrecv intra "$intra_scope"
    write_scope "$file" sendrecv inter "$inter_scope"
  fi
  {
    printf '\n[pipeline]\ndepth = %s\n' "$depth"
    printf '\n[autotune]\nenabled = false\n'
    printf 'reduce_scatter_algorithm = "twoshot"\n'
    printf 'all_reduce_algorithm = "tripleshot"\n'
  } >>"$file"
  printf '%s\n' "$file"
}

configure_compressed() {
  local case_name=$1 depth=$2
  export COCCL_ENABLE=1
  export COCCL_CONFIG_FILE
  COCCL_CONFIG_FILE=$(materialize_config "$case_name" "$depth")
}

configure_native() {
  export COCCL_ENABLE=0
  unset COCCL_CONFIG_FILE
}

case_launcher() {
  CASE_LAUNCHER=("${launcher[@]}")
  local variable
  for variable in "${runtime_variables[@]}"; do
    if [[ -v $variable ]]; then CASE_LAUNCHER+=(-x "$variable"); fi
  done
}

sample_memory() {
  local benchmark_pid=$1 output=$2
  : >"$output"
  local index=0
  local samples=()
  local pids=()
  while read -r host _; do
    local sample="${output}.${index}"
    samples+=("$sample")
    ssh -n "$host" "timeout 120 stdbuf -oL nvidia-smi \
      --query-gpu=index,memory.used --format=csv,noheader,nounits \
      --loop-ms=100 | stdbuf -oL sed 's/^/${host},/'" >"$sample" &
    pids+=("$!")
    index=$((index + 1))
  done <"$hostfile"
  while kill -0 "$benchmark_pid" 2>/dev/null; do sleep 0.1; done
  kill "${pids[@]}" 2>/dev/null || true
  local pid
  for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
  local sample
  for sample in "${samples[@]}"; do cat "$sample" >>"$output"; done
  rm -f "${samples[@]}"
}

run_logged() {
  local phase=$1 stem=$2
  shift 2
  local directory="$raw_root/$phase"
  local log="$directory/$stem.log"
  local marker="$directory/$stem.ok"
  local command_file="$directory/$stem.command"
  local memory_file="$memory_root/$stem.csv"
  mkdir -p "$directory"
  if [[ -n ${M19_ONLY_STEM:-} && "$stem" != "$M19_ONLY_STEM" ]]; then
    return
  fi
  if [[ -f "$marker" && ${M19_FORCE:-0} != 1 ]]; then return; fi
  rm -f "$marker" "$directory/$stem.failed"
  printf '%q ' "$@" >"$command_file"
  printf '\n' >>"$command_file"
  printf '[%s] %s\n' "$(date -Is)" "$stem"
  set +e
  "$@" >"$log" 2>&1 &
  local benchmark_pid=$!
  if [[ "$phase" == performance || "$phase" == dietgpu-performance ]]; then
    sample_memory "$benchmark_pid" "$memory_file"
  fi
  wait "$benchmark_pid"
  local status=$?
  set -e
  printf '%s\n' "$status" >"$directory/$stem.status"
  if (( status == 0 )); then
    touch "$marker"
  else
    touch "$directory/$stem.failed"
    return "$status"
  fi
}

run_correctness_case() {
  local case_name=$1 depth=$2 datatype=$3
  configure_compressed "$case_name" "$depth"
  case_launcher
  local stem="${case_name}__d${depth}__${datatype}"
  run_logged correctness "$stem" \
    timeout --signal=TERM --kill-after=30s "$timeout_seconds" \
    "${CASE_LAUNCHER[@]}" "$test_root/coccl_m16_correctness" \
    --suite hierarchical --compressor "$case_name" --datatype "$datatype" \
    --topology 2x4 --depth "$depth"
}

perf_binary() {
  local path=$1 operation=$2
  if [[ "$path" == native ]]; then
    [[ "$operation" == reducescatter ]] && \
      printf '%s/reduce_scatter_perf\n' "$test_root" || \
      printf '%s/all_reduce_perf\n' "$test_root"
  else
    [[ "$operation" == reducescatter ]] && \
      printf '%s/reduce_scatter_comp_twoshot_perf\n' "$test_root" || \
      printf '%s/all_reduce_comp_tripleshot_perf\n' "$test_root"
  fi
}

run_perf_case() {
  local path=$1 case_name=$2 operation=$3 depth=$4 bytes=$5
  if [[ "$path" == native ]]; then
    configure_native
  else
    configure_compressed "$case_name" "$depth"
  fi
  case_launcher
  local stem="${path}__${case_name}__${operation}__d${depth}__b${bytes}"
  local binary
  binary=$(perf_binary "$path" "$operation")
  run_logged performance "$stem" \
    timeout --signal=TERM --kill-after=30s "$timeout_seconds" \
    "${CASE_LAUNCHER[@]}" "$binary" \
    -b "$bytes" -e "$bytes" -f 2 -t 1 -g 1 \
    -d float -o sum -w 20 -n 30 -c 0
}

run_dietgpu_correctness() {
  local operation=$1 depth=$2
  configure_compressed dietgpu-inter "$depth"
  case_launcher
  run_logged dietgpu-correctness "dietgpu-inter__${operation}__d${depth}" \
    timeout --signal=TERM --kill-after=30s "$timeout_seconds" \
    "${CASE_LAUNCHER[@]}" "$test_root/coccl_dietgpu_integer_test" \
    --mode correctness --datatype int32 --operation "$operation" \
    --depth "$depth"
}

run_dietgpu_perf() {
  local path=$1 operation=$2 pattern=$3 depth=$4 bytes=$5
  if [[ "$path" == native ]]; then
    configure_native
  else
    configure_compressed dietgpu-inter "$depth"
  fi
  case_launcher
  local elements=$((bytes / 4))
  local stem="${path}__${operation}__${pattern}__d${depth}__b${bytes}"
  run_logged dietgpu-performance "$stem" \
    timeout --signal=TERM --kill-after=30s "$timeout_seconds" \
    "${CASE_LAUNCHER[@]}" "$test_root/coccl_dietgpu_integer_test" \
    --mode performance --datatype int32 --operation "$operation" \
    --path "$path" --pattern "$pattern" --elements "$elements" \
    --depth "$depth" --warmups 20 --iterations 30
}

run_codec_probe() {
  local operation=$1 pattern=$2 bytes=$3
  local raw_bytes frames
  if [[ "$operation" == sendrecv-cross-node ]]; then
    raw_bytes=$bytes
    frames=1
  else
    raw_bytes=$((bytes / 4))
    frames=2
  fi
  local frame_bytes=$((raw_bytes / frames))
  export COCCL_ENABLE=0
  case_launcher
  local stem="${operation}__${pattern}__b${bytes}"
  run_logged dietgpu-codec "$stem" \
    timeout --signal=TERM --kill-after=30s "$timeout_seconds" \
    "$mpi_root/bin/mpirun" -np 1 --hostfile "$hostfile" \
    --map-by ppr:1:node --bind-to none --mca btl '^openib' \
    -x PATH -x LD_LIBRARY_PATH -x CUDA_VISIBLE_DEVICES \
    "$test_root/coccl_m14_dietgpu_codec_test" \
    "$plugin_root/libdietgpu.so" --probe "$pattern" \
    "$frame_bytes" "$frames" 10
}

run_environment() {
  {
    printf 'M19 2x4 A800 environment\n'
    printf 'hostfile=%s\n' "$hostfile"
    printf 'commit=%s\n' "$(git -C "$source_root" rev-parse HEAD)"
    while read -r host _; do
      ssh -n "$host" hostname
      ssh -n "$host" nvidia-smi --query-gpu=name,memory.total,driver_version \
        --format=csv,noheader
    done <"$hostfile"
    "$cuda_root/bin/nvcc" --version
  } >"$result_root/environment.txt" 2>&1
}

case "$mode" in
  environment)
    run_environment
    ;;
  smoke)
    run_correctness_case default-sdp 1 float
    run_correctness_case split-sdp 4 float
    for operation in reducescatter-twoshot allreduce-tripleshot \
                     sendrecv-cross-node; do
      run_dietgpu_correctness "$operation" 1
    done
    ;;
  correctness)
    cases=(default-sdp default-zfp intra-sdp inter-sdp split-sdp
           default-intra-override default-inter-override disable-intra
           disable-inter mixed-sdp-zfp)
    for case_name in "${cases[@]}"; do
      for depth in 1 4; do
        for datatype in float half bfloat16; do
          run_correctness_case "$case_name" "$depth" "$datatype"
        done
      done
    done
    ;;
  performance)
    sizes=(67108864 536870912 1073741824)
    for operation in reducescatter allreduce; do
      for bytes in "${sizes[@]}"; do
        run_perf_case native native "$operation" 1 "$bytes"
      done
      for case_name in default-sdp default-zfp intra-sdp inter-sdp split-sdp; do
        for depth in 1 4; do
          for bytes in "${sizes[@]}"; do
            run_perf_case compressed "$case_name" "$operation" "$depth" "$bytes"
          done
        done
      done
    done
    ;;
  dietgpu)
    sizes=(67108864 536870912 1073741824)
    for operation in reducescatter-twoshot allreduce-tripleshot \
                     sendrecv-cross-node; do
      run_dietgpu_correctness "$operation" 1
      [[ "$operation" == sendrecv-cross-node ]] || \
        run_dietgpu_correctness "$operation" 4
    done
    for operation in reducescatter-twoshot allreduce-tripleshot; do
      for pattern in compressible random; do
        for depth in 1 4; do
          for bytes in "${sizes[@]}"; do
            run_dietgpu_perf native "$operation" "$pattern" "$depth" "$bytes"
            run_dietgpu_perf compressed "$operation" "$pattern" "$depth" "$bytes"
          done
        done
        for bytes in "${sizes[@]}"; do
          run_codec_probe "$operation" "$pattern" "$bytes"
        done
      done
    done
    for pattern in compressible random; do
      for bytes in "${sizes[@]}"; do
        run_dietgpu_perf native sendrecv-cross-node "$pattern" 1 "$bytes"
        run_dietgpu_perf compressed sendrecv-cross-node "$pattern" 1 "$bytes"
        run_codec_probe sendrecv-cross-node "$pattern" "$bytes"
      done
    done
    ;;
esac
