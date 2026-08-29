#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  bash train_qwen3_coccl.sh <nnodes> <gpus_per_node> [node_rank]
      [master_addr] [master_port]

Example (run one command on each node):
  bash train_qwen3_coccl.sh 2 4 0 10.0.0.1
  bash train_qwen3_coccl.sh 2 4 1 10.0.0.1

Edit training_envs.sh before launching. Checkpoint saving is disabled.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if (( $# < 2 )); then
  usage
  exit 2
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/training_envs.sh"

nnodes=$1
gpus_per_node=$2
node_rank=${3:-${SLURM_NODEID:-0}}
master_addr=${4:-${MASTER_ADDR:-127.0.0.1}}
master_port=${5:-${MASTER_PORT:-29501}}

coccl_root=${COCCL_ROOT:-$(cd "$script_dir/../.." && pwd)}
coccl_config=${COCCL_CONFIG_FILE:-$script_dir/configs/training.toml}

: "${CUDA_HOME:?Set CUDA_HOME in training_envs.sh or the environment}"
: "${MEGATRON_ROOT:?Set MEGATRON_ROOT in training_envs.sh or the environment}"
: "${QWEN3_DATA_PREFIX:?Set QWEN3_DATA_PREFIX in training_envs.sh or the environment}"
: "${QWEN3_TOKENIZER_DIR:?Set QWEN3_TOKENIZER_DIR in training_envs.sh or the environment}"
: "${TRAIN_OUTPUT_ROOT:?Set TRAIN_OUTPUT_ROOT in training_envs.sh or the environment}"
: "${TRAIN_LOG_ROOT:?Set TRAIN_LOG_ROOT in training_envs.sh or the environment}"

tp_size=${TP_SIZE:-2}
pp_size=${PP_SIZE:-2}
world_size=$((nnodes * gpus_per_node))
model_parallel_size=$((tp_size * pp_size))
if (( world_size % model_parallel_size != 0 )); then
  echo "world size $world_size is not divisible by TP*PP=$model_parallel_size" >&2
  exit 2
fi
dp_size=$((world_size / model_parallel_size))

micro_batch_size=${MICRO_BATCH_SIZE:-1}
global_batch_size=${GLOBAL_BATCH_SIZE:-$dp_size}
sequence_length=${SEQUENCE_LENGTH:-512}
train_iters=${TRAIN_ITERS:-200}
warmup_iters=${WARMUP_ITERS:-2}
eval_interval=${EVAL_INTERVAL:-$train_iters}
eval_iters=${EVAL_ITERS:-10}
dp_overlap=${DP_OVERLAP:-off}
transformer_impl=${TRANSFORMER_IMPL:-auto}

case "$dp_overlap" in
  on) overlap_args=(--overlap-grad-reduce --overlap-param-gather) ;;
  off) overlap_args=() ;;
  *) echo "DP_OVERLAP must be on or off" >&2; exit 2 ;;
esac

if [[ "$transformer_impl" == auto ]]; then
  if python -c 'import transformer_engine' >/dev/null 2>&1; then
    transformer_impl=transformer_engine
  else
    transformer_impl=local
  fi
fi

output_dir="$TRAIN_OUTPUT_ROOT/qwen3-tp${tp_size}-pp${pp_size}-dp${dp_size}-${dp_overlap}"
log_dir="$TRAIN_LOG_ROOT/qwen3-tp${tp_size}-pp${pp_size}-dp${dp_size}-${dp_overlap}"
data_cache_dir=${QWEN3_DATA_CACHE_DIR:-$TRAIN_OUTPUT_ROOT/data-cache}
mkdir -p "$output_dir" "$log_dir" "$data_cache_dir"

export NCCL_HOME="$coccl_root/build"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$NCCL_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$NCCL_HOME/lib/libnccl.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
export COCCL_ENABLE=1
export COCCL_CONFIG_FILE="$coccl_config"
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export PYTHONPATH="$MEGATRON_ROOT:${PYTHONPATH:-}"

model_args=(
  --use-mcore-models
  --transformer-impl "$transformer_impl"
  --num-layers 28
  --hidden-size 1024
  --ffn-hidden-size 3072
  --num-attention-heads 16
  --group-query-attention
  --num-query-groups 8
  --kv-channels 128
  --seq-length "$sequence_length"
  --max-position-embeddings 32768
  --position-embedding-type rope
  --rotary-base 1000000
  --rotary-percent 1.0
  --normalization RMSNorm
  --swiglu
  --qk-layernorm
  --disable-bias-linear
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --init-method-std 0.02
  --vocab-size 151936
  --make-vocab-size-divisible-by 1
)

train_args=(
  --micro-batch-size "$micro_batch_size"
  --global-batch-size "$global_batch_size"
  --train-iters "$train_iters"
  --lr-decay-iters "$train_iters"
  --lr-warmup-iters "$warmup_iters"
  --lr 1.0e-5
  --min-lr 1.0e-6
  --lr-decay-style cosine
  --clip-grad 1.0
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.95
  --bf16
  --calculate-per-token-loss
  --use-distributed-optimizer
  "${overlap_args[@]}"
)

parallel_args=(
  --tensor-model-parallel-size "$tp_size"
  --pipeline-model-parallel-size "$pp_size"
  --context-parallel-size 1
)
if (( tp_size > 1 )); then
  parallel_args+=(--sequence-parallel)
fi

data_args=(
  --data-path "$QWEN3_DATA_PREFIX"
  --split 99,1,0
  --tokenizer-type HuggingFaceTokenizer
  --tokenizer-model "$QWEN3_TOKENIZER_DIR"
  --data-cache-path "$data_cache_dir"
  --num-workers 1
  --no-create-attention-mask-in-dataloader
)

log_args=(
  --log-interval 1
  --timing-log-level 2
  --timing-log-option minmax
  --log-throughput
  --eval-iters "$eval_iters"
  --eval-interval "$eval_interval"
  --ckpt-format torch_dist
  --tensorboard-dir "$output_dir/tensorboard"
  --distributed-timeout-minutes 30
)

load_args=()
if [[ -n "$QWEN3_LOAD_DIR" ]]; then
  load_args=(
    --load "$QWEN3_LOAD_DIR"
    --finetune
    --no-load-optim
    --no-load-rng
    --no-initialization
    --exit-on-missing-checkpoint
    --no-use-tokenizer-model-from-checkpoint-args
  )
fi

echo "Qwen3 COCCL: rank=$node_rank/$nnodes GPUs/node=$gpus_per_node TP=$tp_size PP=$pp_size DP=$dp_size overlap=$dp_overlap"
echo "COCCL_CONFIG_FILE=$COCCL_CONFIG_FILE train_iters=$train_iters output=$output_dir"

cd "$MEGATRON_ROOT"
torchrun \
  --nproc_per_node "$gpus_per_node" \
  --nnodes "$nnodes" \
  --node_rank "$node_rank" \
  --master_addr "$master_addr" \
  --master_port "$master_port" \
  pretrain_gpt.py \
  "${model_args[@]}" \
  "${train_args[@]}" \
  "${parallel_args[@]}" \
  "${data_args[@]}" \
  "${log_args[@]}" \
  "${load_args[@]}" \
  2>&1 | tee "$log_dir/node${node_rank}.log"
