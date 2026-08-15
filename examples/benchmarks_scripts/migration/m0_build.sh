#!/usr/bin/env bash
set -euo pipefail

source_root=${1:?usage: m0_build.sh SOURCE_ROOT [NVCC_GENCODE]}
cuda_root=${CUDA_HOME:-/data/apps/cuda/12.4}
gencode=${2:-'-gencode=arch=compute_80,code=sm_80'}

if type module >/dev/null 2>&1; then
  module unload cuda >/dev/null 2>&1 || true
  module load cuda/12.4
  module load cmake/3.31.12
fi

export CUDA_HOME="$cuda_root"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export NVHPC_CUDA_HOME="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
if [[ -x /data/apps/cmake/3.31.12/bin/cmake ]]; then
  export PATH="/data/apps/cmake/3.31.12/bin:$PATH"
fi
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

make -C "$source_root" -j8 src.build \
  NVCC_GENCODE="$gencode" \
  SUBDIRS='sdp4bit/ taco/ tahquant/ zfp/'

plugin_dir="$source_root/build/obj/device/compress/libcompress"
test -f "$plugin_dir/libsdp4bit.so"
test -f "$plugin_dir/libtaco.so"
test -f "$plugin_dir/libtahquant.so"
test -f "$plugin_dir/libcuzfp.so"

export NCCL_HOME="$source_root/build"
export LD_LIBRARY_PATH="$NCCL_HOME/lib:$LD_LIBRARY_PATH"
export C_INCLUDE_PATH="$NCCL_HOME/include:${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="$NCCL_HOME/include:${CPLUS_INCLUDE_PATH:-}"
make -C "$source_root/tests/coccl-tests" -j8 \
  CUDA_HOME="$CUDA_HOME" \
  NCCL_HOME="$NCCL_HOME" \
  NVCC_GENCODE="$gencode"
