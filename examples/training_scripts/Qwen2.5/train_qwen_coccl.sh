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

export COCCL_ENABLE=1
: "${COCCL_CONFIG_FILE:?set COCCL_CONFIG_FILE to a training config matching this job topology}"
export COCCL_CONFIG_FILE
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
