#!/usr/bin/env bash
set -euo pipefail

mode=${1:?usage: m23_matrix.sh <environment|preflight|smoke|run|report>}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source_root=${M23_SOURCE_ROOT:-$(cd "$script_dir/../../.." && pwd)}
current_root=${M23_CURRENT_ROOT:-/data/home/scyb672/run/lxc/COCCL-migrate}
qwen_root=${M23_QWEN_ROOT:-/data/home/scyb672/run/lxc/qwen-training}
original_root=${M23_ORIGINAL_ROOT:-/data/home/scyb672/run/lxc/COCCL}
temp_root=${M23_TEMP_ROOT:-/data/home/scyb672/run/codex-tmp/coccl-m23-qwen-training}
result_root=${M23_RESULT_ROOT:-$current_root/results/M23}
log_root=${M23_LOG_ROOT:-$qwen_root/logs/qwen-openwebmath-coccl-migrate-2node-1000iter-m23}
checkpoint_root=${M23_CHECKPOINT_ROOT:-$qwen_root/checkpoints/coccl-migrate-m23}
master_host=${M23_MASTER_HOST:-d1n41a11g02}
worker_host=${M23_WORKER_HOST:-d1n41a28g03}
master_addr=${M23_MASTER_ADDR:-10.252.14.32}
ssh_user=${M23_SSH_USER:-scyb672}
node_script="$script_dir/m23_qwen_node.sh"
plugin_root="$current_root/build/obj/coccl-extend/compressor_plugin/libcompress"
config_root="$result_root/runtime-configs"
coccl_commit=$(git -C "$source_root" rev-parse HEAD)

mkdir -p "$temp_root" "$result_root" "$config_root"

materialize_config() {
  local depth=$1
  local config="$config_root/m23-sdp4bit-d${depth}.toml"
  cat >"$config" <<EOF
schema_version = 3

[runtime]
mode = "training"
compression_threshold_bytes = 67108864

[compressor_plugins]
compressors = ["sdp4bit"]
library_path = "$plugin_root"

[training]
observation_iterations = 5
max_events = 65536

[training.classifier]
data_parallel_size = 2
tensor_parallel_size = 2
pipeline_parallel_size = 2
dp_strategy = "sdp"
sequence_parallel = true
context_parallel = false

[training.dp.all_gather.default]
compressor = "sdp4bit"

[training.dp.all_gather.default.config]
groupCount = 2048
quantBits = 8
quantType = "Symmetric"
subAdd = true
pipelineSize = $depth

[training.dp.reduce_scatter.default]
compressor = "sdp4bit"

[training.dp.reduce_scatter.default.config]
groupCount = 128
quantBits = 4
quantType = "Symmetric"
hadamard = true

[training.dp.reduce_scatter.intra]
compressor = "sdp4bit"

[training.dp.reduce_scatter.intra.config]
groupCount = 128
quantBits = 4
quantType = "Symmetric"
hadamard = true

[training.dp.reduce_scatter.inter]
compressor = "sdp4bit"

[training.dp.reduce_scatter.inter.config]
groupCount = 128
quantBits = 4
quantType = "Symmetric"
hadamard = true

[pipeline]
depth = $depth

[autotune]
enabled = false
reduce_scatter_algorithm = "oneshot"
all_reduce_algorithm = "oneshot"
EOF
  printf '%s\n' "$config"
}

case_port() {
  local model=$1 depth=$2 overlap=$3 backend=$4
  local base=29600
  [[ "$model" == qwen3 ]] && base=29700
  [[ "$overlap" == on ]] && base=$((base + 20))
  case "$backend" in
    current) base=$((base + 0)) ;;
    original) base=$((base + 40)) ;;
    native) base=$((base + 60)) ;;
  esac
  printf '%s\n' $((base + depth))
}

run_case() {
  local model=$1 depth=$2 overlap=$3 backend=$4 phase=$5
  local stem="${model}-d${depth}-${overlap}-${backend}"
  if [[ -n ${M23_ONLY_CASE:-} && "$stem" != "$M23_ONLY_CASE" ]]; then
    return
  fi

  local case_root output iterations warmup eval_interval eval_iters timeout_seconds debug
  if [[ "$phase" == formal ]]; then
    case_root="$log_root/$model/d${depth}-${overlap}"
    output="$checkpoint_root/$model/d${depth}-${overlap}"
    iterations=${M23_TRAIN_ITERS:-1000}
    warmup=${M23_WARMUP_ITERS:-10}
    eval_interval=${M23_EVAL_INTERVAL:-100}
    eval_iters=${M23_EVAL_ITERS:-10}
    timeout_seconds=${M23_CASE_TIMEOUT:-3600}
    debug=${M23_NCCL_DEBUG:-INFO}
  else
    case_root="$temp_root/$phase/$stem"
    output="$temp_root/$phase/checkpoints/$stem"
    iterations=${M23_SMOKE_ITERS:-12}
    warmup=${M23_SMOKE_WARMUP_ITERS:-4}
    eval_interval=1000
    eval_iters=1
    timeout_seconds=${M23_SMOKE_TIMEOUT:-1200}
    debug=${M23_SMOKE_NCCL_DEBUG:-INFO}
  fi
  local marker="$case_root/complete.ok"
  if [[ -f "$marker" && ${M23_FORCE:-0} != 1 ]]; then
    printf '[M23] skip completed %s\n' "$stem"
    return
  fi

  local config port
  config=$(materialize_config "$depth")
  port=$(case_port "$model" "$depth" "$overlap" "$backend")
  mkdir -p "$case_root/nccl" "$output"
  rm -f "$marker"
  rm -f "$case_root/nccl"/nccl.*.log

  launch_node() {
    local host=$1 node_rank=$2 log=$3
    local remote=(
      timeout --signal=TERM --kill-after=30 "${timeout_seconds}s"
      env
      M23_QWEN_ROOT="$qwen_root"
      M23_COCCL_ROOT="$current_root"
      M23_ORIGINAL_ROOT="$original_root"
      M23_MASTER_ADDR="$master_addr"
      M23_TRAIN_ITERS="$iterations"
      M23_WARMUP_ITERS="$warmup"
      M23_EVAL_INTERVAL="$eval_interval"
      M23_EVAL_ITERS="$eval_iters"
      M23_SAVE_INTERVAL=1000
      M23_NCCL_DEBUG="$debug"
      M23_NCCL_DEBUG_SUBSYS="${M23_NCCL_DEBUG_SUBSYS:-TUNING}"
      M23_NCCL_LOG_DIR="$case_root/nccl"
      M23_COCCL_COMMIT="$coccl_commit"
      bash "$node_script" "$model" "$node_rank" "$port" "$config"
      "$depth" "$overlap" "$backend" "$output" "$stem"
    )
    local command
    printf -v command '%q ' "${remote[@]}"
    ssh -o BatchMode=yes -o ConnectTimeout=15 "$ssh_user@$host" "$command" \
      >"$log" 2>&1
  }

  printf '[M23] start %s iterations=%s port=%s\n' "$stem" "$iterations" "$port"
  launch_node "$master_host" 0 "$case_root/node0.log" &
  local master_pid=$!
  launch_node "$worker_host" 1 "$case_root/node1.log" &
  local worker_pid=$!
  set +e
  wait "$master_pid"
  local master_status=$?
  wait "$worker_pid"
  local worker_status=$?
  set -e
  printf '%s\n' "$master_status" >"$case_root/node0.status"
  printf '%s\n' "$worker_status" >"$case_root/node1.status"
  if (( master_status != 0 || worker_status != 0 )); then
    printf '[M23] failed %s node0=%s node1=%s\n' \
      "$stem" "$master_status" "$worker_status" >&2
    return 1
  fi
  touch "$marker"
  printf '[M23] complete %s\n' "$stem"
}

case "$mode" in
  environment)
    printf 'source_commit=%s\n' "$coccl_commit"
    printf 'library=%s\n' "$(realpath "$current_root/build/lib/libnccl.so.2")"
    printf 'plugin=%s\n' "$(realpath "$plugin_root/libsdp4bit.so")"
    printf 'master=%s worker=%s\n' "$master_host" "$worker_host"
    ;;
  preflight)
    run_case qwen25 1 on native preflight
    if ! run_case qwen25 1 on original preflight; then
      printf '[M23] original overlap preflight failed; result retained for comparison\n'
    fi
    run_case qwen25 1 on current preflight
    ;;
  smoke)
    run_case qwen25 1 off current smoke
    run_case qwen3 4 on current smoke
    ;;
  run)
    for model in qwen25 qwen3; do
      for overlap in off on; do
        for depth in 1 2 4 8; do
          run_case "$model" "$depth" "$overlap" current formal
        done
      done
    done
    ;;
  report)
    python3 "$script_dir/m23_report.py" "$result_root" "$log_root" "$coccl_commit"
    ;;
  *)
    printf 'unknown M23 mode: %s\n' "$mode" >&2
    exit 2
    ;;
esac
