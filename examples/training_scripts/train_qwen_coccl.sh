#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/training_envs.sh"

model=${1:?usage: $0 <qwen25|qwen3> [nnodes] [gpus-per-node] [node-rank]}
nnodes=${2:-${NNODES:-2}}
gpus_per_node=${3:-${GPUS_PER_NODE:-4}}
node_rank=${4:-${NODE_RANK:-0}}
coccl_root=${COCCL_ROOT:-$(cd "$script_dir/../.." && pwd)}
coccl_config=${COCCL_CONFIG_FILE:-$coccl_root/src/coccl-extend/extensions/configs/training.toml}

: "${CUDA_HOME:?Set CUDA_HOME in training_envs.sh or the environment}"
: "${PAI_MEGATRON_ROOT:?Set PAI_MEGATRON_ROOT in training_envs.sh or the environment}"
: "${TRAIN_OUTPUT_ROOT:?Set TRAIN_OUTPUT_ROOT in training_envs.sh or the environment}"
: "${TRAIN_LOG_ROOT:?Set TRAIN_LOG_ROOT in training_envs.sh or the environment}"

tp_size=${TP_SIZE:-2}
pp_size=${PP_SIZE:-2}
cp_size=${CP_SIZE:-1}
micro_batch_size=${MICRO_BATCH_SIZE:-1}
world_size=$((nnodes * gpus_per_node))
model_parallel_size=$((tp_size * pp_size * cp_size))
if (( world_size % model_parallel_size != 0 )); then
  echo "world size $world_size is not divisible by TP*PP*CP=$model_parallel_size" >&2
  exit 2
fi
dp_size=$((world_size / model_parallel_size))
global_batch_size=${GLOBAL_BATCH_SIZE:-$dp_size}
sequence_length=${SEQUENCE_LENGTH:-512}
train_iterations=${TRAIN_ITERS:-100}
warmup_iterations=${WARMUP_ITERS:-10}
train_tokens=$((train_iterations * global_batch_size * sequence_length))
warmup_tokens=$((warmup_iterations * global_batch_size * sequence_length))
save_interval=${SAVE_INTERVAL:-$train_iterations}
save_checkpoints=${SAVE_CHECKPOINTS:-off}
dp_overlap=${DP_OVERLAP:-off}

case "$dp_overlap" in
  on) unset QWEN_DISABLE_DP_OVERLAP ;;
  off) export QWEN_DISABLE_DP_OVERLAP=1 ;;
  *) echo "DP_OVERLAP must be on or off" >&2; exit 2 ;;
esac

case "$save_checkpoints" in
  on) ;;
  off)
    torchrun() {
      "$CONDA_PREFIX/bin/torchrun" "$@" --save ""
    }
    export -f torchrun
    ;;
  *) echo "SAVE_CHECKPOINTS must be on or off" >&2; exit 2 ;;
esac

export WORLD_SIZE="$nnodes"
export KUBERNETES_CONTAINER_RESOURCE_GPU="$gpus_per_node"
export RANK="$node_rank"
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29501}
export QWEN_GLOBAL_BATCH_SIZE="$global_batch_size"
export QWEN_EVAL_INTERVAL=${EVAL_INTERVAL:-100}
export QWEN_EVAL_ITERS=${EVAL_ITERS:-10}
export QWEN_SAVE_INTERVAL="$save_interval"
export QWEN_TIMING_LOG_LEVEL=${QWEN_TIMING_LOG_LEVEL:-1}
export QWEN_TIMING_LOG_OPTION=${QWEN_TIMING_LOG_OPTION:-minmax}
export MP_DATASET_TYPE=idxmap
export MP_AC_LAYERS=${MP_AC_LAYERS:-1}
export MP_ATTENTION_BACKEND=${MP_ATTENTION_BACKEND:-unfused}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export UB_SKIPMC=${UB_SKIPMC:-1}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-true}

export NCCL_HOME="$coccl_root/build"
plugin_root="$coccl_root/build/obj/coccl-extend/compressor_plugin/libcompress"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$NCCL_HOME/lib:$plugin_root:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$NCCL_HOME/lib/libnccl.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_BUFFSIZE=${NCCL_BUFFSIZE:-33554432}
export COCCL_ENABLE=1
export COCCL_CONFIG_FILE="$coccl_config"

learning_rate=${LEARNING_RATE:-1e-5}
minimum_learning_rate=${MINIMUM_LEARNING_RATE:-1e-6}
output="$TRAIN_OUTPUT_ROOT/$model"
log_dir="$TRAIN_LOG_ROOT/$model"
mkdir -p "$output" "$log_dir"

case "$model" in
  qwen25)
    : "${QWEN25_DATA_PREFIX:?Set QWEN25_DATA_PREFIX in training_envs.sh or the environment}"
    : "${QWEN25_MODEL_PATH:?Set QWEN25_MODEL_PATH in training_envs.sh or the environment}"
    example="$PAI_MEGATRON_ROOT/examples/qwen2_5"
    command=(
      bash "$example/run_mcore_qwen.sh" dlc "${QWEN25_MODEL_SIZE:-0.5B}"
      "$micro_batch_size" "$global_batch_size"
      "$learning_rate" "$minimum_learning_rate"
      "$sequence_length" "$sequence_length" bf16
      "$tp_size" "$pp_size" "$cp_size"
      true true false false none false "$save_interval"
      "$QWEN25_DATA_PREFIX" "$QWEN25_DATA_PREFIX" "$QWEN25_MODEL_PATH"
      "$train_tokens" "$warmup_tokens" "$output"
    )
    ;;
  qwen3)
    : "${QWEN3_DATA_PREFIX:?Set QWEN3_DATA_PREFIX in training_envs.sh or the environment}"
    : "${QWEN3_MODEL_PATH:?Set QWEN3_MODEL_PATH in training_envs.sh or the environment}"
    example="$PAI_MEGATRON_ROOT/examples/qwen3"
    command=(
      bash "$example/run_mcore_qwen3.sh" dlc "${QWEN3_MODEL_SIZE:-0.6B}"
      "$micro_batch_size" "$global_batch_size"
      "$learning_rate" "$minimum_learning_rate"
      "$sequence_length" "$sequence_length" bf16
      "$tp_size" "$pp_size" "$cp_size"
      "${EXPERT_TENSOR_PARALLEL_SIZE:-1}" "${EXPERT_MODEL_PARALLEL_SIZE:-1}"
      true true false false none false "$save_interval"
      "$QWEN3_DATA_PREFIX" "$QWEN3_DATA_PREFIX" "$QWEN3_MODEL_PATH"
      "$train_tokens" "$warmup_tokens" "$output"
    )
    ;;
  *)
    echo "unknown model '$model'; use qwen25 or qwen3" >&2
    exit 2
    ;;
esac

echo "model=$model rank=$node_rank/$nnodes GPUs/node=$gpus_per_node TP=$tp_size PP=$pp_size CP=$cp_size DP=$dp_size"
echo "COCCL_CONFIG_FILE=$COCCL_CONFIG_FILE iterations=$train_iterations DP_OVERLAP=$dp_overlap SAVE_CHECKPOINTS=$save_checkpoints"
cd "$example"
"${command[@]}" 2>&1 | tee "$log_dir/node${node_rank}.log"
