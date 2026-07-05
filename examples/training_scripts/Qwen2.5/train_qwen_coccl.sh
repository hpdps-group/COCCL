#!/bin/bash

#module load nccl/2.18.3-cuda12.1

# SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# source activate paimegatron

source training_envs.sh

export PATH=$CUDA_PATH/bin:$PATH
export CUDACXX=$CUDA_PATH/bin/nvcc
export CUDA_HOME=$CUDA_PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$LD_LIBRARY_PATH
export CUDNN_INCLUDE_DIR=$CUDNN_PATH/include
export CUDNN_LIB_DIR=$CUDNN_PATH/lib

export CC=$GCC_PATH/bin/gcc
export CXX=$GCC_PATH/bin/g++

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

NNODES=${1:-1}
GPUS_PER_NODE=${2:-4}
export WORLD_SIZE=$NNODES
export KUBERNETES_CONTAINER_RESOURCE_GPU=$GPUS_PER_NODE
export RANK=0
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29501}
BATCH_JOB_ID=${BATCH_JOB_ID:-single_node}
export NCCL_HOME=$COCCL_PATH/build
export LD_LIBRARY_PATH=$NCCL_HOME/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$NCCL_HOME/lib:$LIBRARY_PATH
export C_INCLUDE_PATH=$NCCL_HOME/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$NCCL_HOME/include:$CPLUS_INCLUDE_PATH


export NCCL_DEBUG=INFO
export NCCL_DEBUG_FILE=$NCCL_LOG_DIR/ncclcomp.%h.%p
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_3:1,mlx5_4:1
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=13

export NCCL_ENABLE_COMPRESS=1
export NCCL_COMPRESSORS=sdp4bit,tahquant
export NCCL_ENABLE_ALLREDUCE_COMPRESS=1
export NCCL_ALLREDUCE_COMPRESSORS=tahquant
export NCCL_ENABLE_ALLGATHER_COMPRESS=1
export NCCL_ALLGATHER_COMPRESSORS=sdp4bit
export NCCL_ENABLE_REDUCESCATTER_COMPRESS=1
export NCCL_REDUCESCATTER_COMPRESSORS=sdp4bit
export NCCL_REDUCESCATTER_INTER_COMPRESSORS=sdp4bit
export NCCL_ENABLE_SENDRECV_COMPRESS=1
export NCCL_SENDRECV_COMPRESSORS=tahquant
export NCCL_SENDRECV_BWD_COMPRESSORS=tahquant
export NCCL_COMPRESS_ENABLE_THRESHOLD=10485760 # 10MB
export NCCL_COMPRESSORS_CONFIG_PATH=$COCCL_PATH/examples/training_scripts/configs_training
export NCCL_COMPRESSORS_LIB_PATH=$COCCL_PATH/build/obj/device/compress/libcompress
# export NCCL_ALGO=Tree
export NCCL_LOCAL_REGISTER=0
export NCCL_CHECKS_DISABLE=1
export NCCL_ENABLE_CHECK=0
export NCCL_BUFFSIZE=33554432

# 4. pretrain
cd $QWEN_EXAMPLE_PATH
unset MP_DATASET_TYPE

bash run_mcore_qwen.sh \
  dlc \
  3B \
  1 \
  64 \
  3e-4 \
  3e-5 \
  2048 \
  2048 \
  bf16 \
  2 \
  1 \
  1 \
  true \
  true \
  true \
  false \
  false \
  false \
  100000 \
  $DATASET_PATH \
  $DATASET_PATH \
  $MODEL_PATH \
  655360 \
  131072 \
  $OUTPUT_PATH \
  2>&1 | tee $LOG_DIR/qwen2.5-3B_distributed_Rank${RANK}-coccl-tp2pp2dp2-${BATCH_JOB_ID}.log

