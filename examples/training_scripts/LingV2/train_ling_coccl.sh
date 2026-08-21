#!/bin/bash

# module unload cuda

# module load miniconda/24.9.2
# module load cuda/12.4
# module load cudnn/9.11.0.98_cuda12
# module load nvshmem/3.2.5_cuda12.4_gdrcopy2.4.4

# SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source training_envs.sh

export PATH=$CUDA_PATH/bin:$PATH
export CUDACXX=$CUDA_PATH/bin/nvcc
export CUDA_HOME=$CUDA_PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$LD_LIBRARY_PATH
export CUDNN_INCLUDE_DIR=$CUDNN_PATH/include
export CUDNN_LIB_DIR=$CUDNN_PATH/lib
export PYTHONNOUSERSITE=1


export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$MEGATRON_CORE_PATH:$PYTHONPATH

NNODES=${1:-1}
GPUS_PER_NODE=${2:-8}
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

export NCCL_DEBUG=WARN
export NCCL_DEBUG_FILE=$NCCL_LOG_DIR/ncclcomp.%h.%p

export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_3:1,mlx5_4:1
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=13

export COCCL_ENABLE=1
: "${COCCL_CONFIG_FILE:?set COCCL_CONFIG_FILE to a training config matching this job topology}"
export COCCL_CONFIG_FILE


export NCCL_CHECKS_DISABLE=1
export NCCL_ENABLE_CHECK=0
export NCCL_BUFFSIZE=16777216

cd $LING_PRETRAIN_PATH

bash run_pretrain_4k.sh 1 1 8 200 $LOG_DIR/ling_distributed_Rank${RANK}_coccl_tp1pp1ep8_${BATCH_JOB_ID}.log
