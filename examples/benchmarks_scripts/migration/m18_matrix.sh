#!/usr/bin/env bash
set -euo pipefail

source_root=${1:?usage: m18_matrix.sh SOURCE_ROOT profile|report}
mode=${2:?usage: m18_matrix.sh SOURCE_ROOT profile|report}
hostfile=${M18_HOSTFILE:?M18_HOSTFILE must name the two-node hostfile}
cuda_root=${M18_CUDA_HOME:-/data/apps/cuda/12.4}
mpi_root=${M18_MPI_HOME:-/data/apps/openmpi/4.1.5_cuda12.8}
current_root=${M18_CURRENT_ROOT:-/data/home/scyb672/run/lxc/COCCL-migrate}
m17_root=${M18_M17_RESULT_ROOT:-$current_root/results/M17}
temp_root=${M18_TEMP_ROOT:-/data/home/scyb672/run/codex-tmp/coccl-m18-autotune}
result_root=${M18_RESULT_ROOT:-$current_root/results/M18}

export CUDA_HOME="$cuda_root"
export PATH="$CUDA_HOME/bin:$mpi_root/bin:$PATH"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

mkdir -p "$result_root/raw/profile" "$result_root/runtime-configs" \
  "$result_root/model_snapshots"

materialize_config() {
  local compressor=$1
  local source="$m17_root/runtime-configs/${compressor}-hierarchical-d1.toml"
  local output="$result_root/runtime-configs/${compressor}-autotune-d1.toml"
  awk '/^\[autotune\]/{exit} {print}' "$source" >"$output"
  cat >>"$output" <<'EOF'

[autotune]
enabled = true
profile_min_bytes = 3145728
profile_max_bytes = 201326592
warmup = 3
iterations = 10
reduce_scatter_algorithm = "auto"
all_reduce_algorithm = "auto"
EOF
  printf '%s\n' "$output"
}

run_profile() {
  local compressor=$1
  local config
  config=$(materialize_config "$compressor")
  local log="$result_root/raw/profile/${compressor}.log"
  local status="$result_root/raw/profile/${compressor}.status"
  local command=(
    timeout --signal=TERM --kill-after=30s 1800
    "$mpi_root/bin/mpirun" -np 8 --hostfile "$hostfile"
    --map-by ppr:4:node --bind-to none --mca btl '^openib'
    -x "LD_LIBRARY_PATH=$current_root/build/lib:$current_root/build/obj/coccl-extend/compressor_plugin/libcompress:$cuda_root/lib64:$mpi_root/lib"
    -x CUDA_VISIBLE_DEVICES
    -x "COCCL_CONFIG_FILE=$config"
    -x COCCL_ENABLE=1
    -x NCCL_DEBUG=INFO
    -x NCCL_DEBUG_SUBSYS=TUNING
    -x NCCL_CUMEM_ENABLE=1
    "$temp_root/perf/all_reduce_perf"
    -b 4194304 -e 4194304 -f 2 -t 1 -g 1
    -d float -o sum -w 20 -n 30 -c 0
  )
  printf '[%s] M18 profile %s\n' "$(date -Is)" "$compressor"
  set +e
  "${command[@]}" >"$log" 2>&1
  local rc=$?
  set -e
  printf '%s\n' "$rc" >"$status"
  (( rc == 0 ))
}

case "$mode" in
  profile)
    run_profile sdp4bit
    run_profile zfp
    ;;
  report)
    python3 "$source_root/examples/benchmarks_scripts/migration/m18_report.py" \
      "$result_root" "$m17_root/candidate_oracle.csv" \
      "$temp_root/host/coccl_m18_model_selector"
    ;;
  *)
    printf 'unknown mode: %s\n' "$mode" >&2
    exit 2
    ;;
esac
