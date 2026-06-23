#!/bin/bash

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_PRELOAD=/lib/x86_64-linux-gnu/libtinfo.so.6:$LD_PRELOAD

which python || { echo "Python not found"; exit 1; }

export PROJECT_ROOT=${PROJECT_ROOT:-/path/to/TACO-COCCL}
export COCCL_PATH=${COCCL_PATH:-$PROJECT_ROOT/coccl-acc}
export MEGATRON_PATH=${MEGATRON_PATH:-$PROJECT_ROOT/Megatron}

export DATA_PATH=${DATA_PATH:-/path/to/dataset}
export CKPT_PATH=${CKPT_PATH:-/path/to/qwen-ckpt}
export LOG_PATH=${LOG_PATH:-$PROJECT_ROOT/logs}

export NCCL_HOME=$COCCL_PATH/build
export LD_LIBRARY_PATH=$NCCL_HOME/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=$NCCL_HOME/lib/libnccl.so:$LD_PRELOAD

export NCCL_DEBUG=INFO
export NCCL_DEBUG_FILE=$LOG_PATH/nccl.log
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=^lo,docker,veth

export NCCL_ENABLE_COMPRESS=1
export NCCL_COMPRESSORS=taco
export NCCL_ENABLE_ALLREDUCE_COMPRESS=1
export NCCL_ALLREDUCE_COMPRESSORS=taco
export NCCL_ENABLE_ALLGATHER_COMPRESS=0
export NCCL_ALLGATHER_COMPRESSORS=taco
export NCCL_ENABLE_REDUCESCATTER_COMPRESS=0
export NCCL_REDUCESCATTER_COMPRESSORS=taco
export NCCL_REDUCESCATTER_INTER_COMPRESSORS=taco
export NCCL_ENABLE_SENDRECV_COMPRESS=0
export NCCL_SENDRECV_COMPRESSORS=taco
export NCCL_SENDRECV_BWD_COMPRESSORS=taco
export NCCL_COMPRESSORS_CONFIG_PATH=$COCCL_PATH/src/device/compress/configs
export NCCL_COMPRESSORS_LIB_PATH=$COCCL_PATH/build/obj/device/compress/libcompress

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6001}

cd $MEGATRON_PATH/examples/qwen2_5 || exit

unset MP_DATASET_TYPE

bash run_mcore_qwen.sh \
    dsw \
    7B \
    $GPUS_PER_NODE \
    64 \
    3e-4 \
    3e-5 \
    512 \
    512 \
    bf16 \
    2 \
    2 \
    1 \
    true \
    true \
    true \
    false \
    false \
    false \
    100000 \
    $DATA_PATH \
    $DATA_PATH \
    $CKPT_PATH \
    3276800 \
    32768 \
    $LOG_PATH \
    2>&1 | tee $LOG_PATH/train.log