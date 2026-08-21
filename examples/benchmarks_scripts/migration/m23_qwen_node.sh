#!/usr/bin/env bash
set -euo pipefail

model=${1:?model is required}
node_rank=${2:?node rank is required}
master_port=${3:?master port is required}
config=${4:?config path is required}
depth=${5:?pipeline depth is required}
overlap=${6:?DP overlap mode is required}
backend=${7:?backend is required}
output=${8:?output path is required}
case_name=${9:?case name is required}

qwen_root=${M23_QWEN_ROOT:-/data/home/scyb672/run/lxc/qwen-training}
coccl_root=${M23_COCCL_ROOT:-/data/home/scyb672/run/lxc/COCCL-migrate}
original_root=${M23_ORIGINAL_ROOT:-/data/home/scyb672/run/lxc/COCCL}
master_addr=${M23_MASTER_ADDR:-10.252.14.32}
nnodes=2
gpus_per_node=4
micro_batch_size=1
global_batch_size=2
sequence_length=512
train_iterations=${M23_TRAIN_ITERS:-1000}
warmup_iterations=${M23_WARMUP_ITERS:-10}
train_tokens=$((train_iterations * global_batch_size * sequence_length))
warmup_tokens=$((warmup_iterations * global_batch_size * sequence_length))
nccl_log_dir=${M23_NCCL_LOG_DIR:?NCCL log directory is required}

source "$qwen_root/env/load_qwen_env.sh"
source "$qwen_root/config/models.env"
mkdir -p "$output" "$nccl_log_dir"

export WORLD_SIZE=$nnodes
export KUBERNETES_CONTAINER_RESOURCE_GPU=$gpus_per_node
export RANK=$node_rank
export MASTER_ADDR=$master_addr
export MASTER_PORT=$master_port
export QWEN_GLOBAL_BATCH_SIZE=$global_batch_size
export QWEN_EVAL_INTERVAL=${M23_EVAL_INTERVAL:-100}
export QWEN_EVAL_ITERS=${M23_EVAL_ITERS:-10}
export QWEN_SAVE_INTERVAL=${M23_SAVE_INTERVAL:-1000}
export QWEN_TIMING_LOG_LEVEL=1
export QWEN_TIMING_LOG_OPTION=minmax
export MP_DATASET_TYPE=idxmap
export MP_AC_LAYERS=1
export MP_ATTENTION_BACKEND=unfused
export CUDA_DEVICE_MAX_CONNECTIONS=1
export UB_SKIPMC=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=bond0
export NCCL_DEBUG=${M23_NCCL_DEBUG:-WARN}
export NCCL_DEBUG_SUBSYS=${M23_NCCL_DEBUG_SUBSYS:-TUNING}
export NCCL_DEBUG_FILE="$nccl_log_dir/nccl.%h.%p.log"

if [[ "$overlap" == off ]]; then
  export QWEN_DISABLE_DP_OVERLAP=1
  ddp_bucket_size=default
  torchrun() {
    "$CONDA_PREFIX/bin/torchrun" "$@" --save ""
  }
else
  unset QWEN_DISABLE_DP_OVERLAP
  export M23_DDP_BUCKET_SIZE=${M23_DDP_BUCKET_SIZE:-1073741824}
  ddp_bucket_size=$M23_DDP_BUCKET_SIZE
  torchrun() {
    "$CONDA_PREFIX/bin/torchrun" "$@" \
      --ddp-bucket-size "$M23_DDP_BUCKET_SIZE" --save ""
  }
fi
export -f torchrun

library=system
case "$backend" in
  current)
    export COCCL_ENABLE=1
    export COCCL_CONFIG_FILE=$config
    export NCCL_HOME="$coccl_root/build"
    library="$NCCL_HOME/lib/libnccl.so.2"
    export LD_LIBRARY_PATH="$NCCL_HOME/lib:${LD_LIBRARY_PATH:-}"
    export LD_PRELOAD="$library${LD_PRELOAD:+:$LD_PRELOAD}"
    ;;
  original)
    export NCCL_HOME="$original_root/build"
    library="$NCCL_HOME/lib/libnccl.so.2"
    export LD_LIBRARY_PATH="$NCCL_HOME/lib:${LD_LIBRARY_PATH:-}"
    export LD_PRELOAD="$library${LD_PRELOAD:+:$LD_PRELOAD}"
    export NCCL_ENABLE_COMPRESS=1
    export NCCL_COMPRESSORS=sdp4bit
    export NCCL_COMPRESSORS_CONFIG_PATH="$original_root/examples/training_scripts/configs_training"
    export NCCL_COMPRESSORS_LIB_PATH="$original_root/build/obj/device/compress/libcompress"
    export NCCL_ENABLE_ALLGATHER_COMPRESS=1
    export NCCL_ALLGATHER_COMPRESSORS=sdp4bit
    export NCCL_ENABLE_REDUCESCATTER_COMPRESS=1
    export NCCL_REDUCESCATTER_COMPRESSORS=sdp4bit
    export NCCL_REDUCESCATTER_INTER_COMPRESSORS=sdp4bit
    export NCCL_ENABLE_ALLREDUCE_COMPRESS=0
    export NCCL_ENABLE_SENDRECV_COMPRESS=0
    export NCCL_COMPRESS_ENABLE_THRESHOLD=67108864
    ;;
  native)
    unset LD_PRELOAD COCCL_CONFIG_FILE
    export COCCL_ENABLE=0
    export NCCL_ENABLE_COMPRESS=0
    ;;
  *)
    printf 'unknown backend: %s\n' "$backend" >&2
    exit 2
    ;;
esac

printf 'COCCL_M23_RUN case=%s model=%s rank=%s backend=%s depth=%s overlap=%s ddp_bucket_size=%s iterations=%s config=%s commit=%s library=%s\n' \
  "$case_name" "$model" "$node_rank" "$backend" "$depth" "$overlap" \
  "$ddp_bucket_size" "$train_iterations" "$config" \
  "${M23_COCCL_COMMIT:-unknown}" \
  "$(realpath -m "$library")"

case "$model" in
  qwen25)
    data="$qwen_root/data/mmap/open-web-math/smoke/qwen25/train_text_document"
    load="$qwen_root/models/mcore/qwen2.5-0.5b-base-tp1-pp1"
    example="$qwen_root/framework/Pai-Megatron-Patch/examples/qwen2_5"
    cd "$example"
    exec bash "$example/run_mcore_qwen.sh" dlc 0.5B "$micro_batch_size" "$global_batch_size" \
      1e-5 1e-6 "$sequence_length" "$sequence_length" bf16 \
      2 2 1 true true false false none false "$QWEN_SAVE_INTERVAL" \
      "$data" "$data" "$load" "$train_tokens" "$warmup_tokens" "$output"
    ;;
  qwen3)
    data="$qwen_root/data/mmap/open-web-math/smoke/qwen3/train_text_document"
    load="$qwen_root/models/mcore/qwen3-0.6b-base-tp1-pp1"
    example="$qwen_root/framework/Pai-Megatron-Patch/examples/qwen3"
    cd "$example"
    exec bash "$example/run_mcore_qwen3.sh" dlc 0.6B "$micro_batch_size" "$global_batch_size" \
      1e-5 1e-6 "$sequence_length" "$sequence_length" bf16 \
      2 2 1 1 1 true true false false none false "$QWEN_SAVE_INTERVAL" \
      "$data" "$data" "$load" "$train_tokens" "$warmup_tokens" "$output"
    ;;
  *)
    printf 'unknown model: %s\n' "$model" >&2
    exit 2
    ;;
esac
