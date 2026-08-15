#!/usr/bin/env bash

m0_source_root=${1:?source root required}
m0_result_root=${M0_RESULT_ROOT:-/data/home/scyb672/run/lxc/COCCL-migrate/results/M0}

if type module >/dev/null 2>&1; then
  module unload cuda >/dev/null 2>&1 || true
  module load cuda/12.4
fi

export CUDA_HOME=${CUDA_HOME:-/data/apps/cuda/12.4}
export CUDACXX="$CUDA_HOME/bin/nvcc"
export NVHPC_CUDA_HOME="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

export NCCL_HOME="$m0_source_root/build"
export LD_LIBRARY_PATH="$NCCL_HOME/lib:$LD_LIBRARY_PATH"
export C_INCLUDE_PATH="$NCCL_HOME/include:${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="$NCCL_HOME/include:${CPLUS_INCLUDE_PATH:-}"

export NCCL_COMPRESSORS_CONFIG_PATH="$m0_source_root/examples/benchmarks_scripts/configs"
export NCCL_COMPRESSORS_LIB_PATH="$m0_source_root/build/obj/device/compress/libcompress"

export NCCL_DEBUG=WARN
export NCCL_BUFFSIZE=16777216
export NCCL_ENABLE_COMPRESS=1
export NCCL_COMPRESSORS=sdp4bit,cuzfp
export NCCL_ENABLE_ALLTOALL_COMPRESS=1
export NCCL_ENABLE_ALLREDUCE_COMPRESS=1
export NCCL_ENABLE_ALLGATHER_COMPRESS=1
export NCCL_ENABLE_REDUCESCATTER_COMPRESS=1
export NCCL_LOCAL_REGISTER=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

unset NCCL_ENABLE_AUTOTUNE COCCL_AUTOTUNE NCCL_COCCL_AUTOTUNE
