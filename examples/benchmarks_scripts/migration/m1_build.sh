#!/usr/bin/env bash
set -euo pipefail

source_root=${1:?usage: m1_build.sh SOURCE_ROOT}
cuda_root=${CUDA_HOME:-/data/apps/cuda/12.4}
gencode=${NVCC_GENCODE:-'-gencode=arch=compute_80,code=sm_80'}

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
  SUBDIRS='sdp4bit/ zfp/'

plugin_dir="$source_root/build/obj/device/compress/libcompress"
test -f "$plugin_dir/libsdp4bit.so"
test -f "$plugin_dir/libzfp.so"

export NCCL_HOME="$source_root/build"
export LD_LIBRARY_PATH="$NCCL_HOME/lib:$LD_LIBRARY_PATH"
make -C "$source_root/tests/coccl-tests/src" -j8 \
  BUILDDIR="$source_root/tests/coccl-tests/build" \
  CUDA_HOME="$CUDA_HOME" NCCL_HOME="$NCCL_HOME" \
  NVCC_GENCODE="$gencode" \
  "$source_root/tests/coccl-tests/build/alltoall_p2p_perf" \
  "$source_root/tests/coccl-tests/build/alltoall_comp_overlap_perf"

make -C "$source_root/tests/coccl-tests/src" m1-host-tests \
  BUILDDIR=/data/home/scyb672/run/codex-tmp/coccl-m1-host \
  CUDA_HOME="$CUDA_HOME" NCCL_HOME="$NCCL_HOME"
make -C "$source_root/tests/coccl-tests/src" m1-plugin-load-test \
  BUILDDIR=/data/home/scyb672/run/codex-tmp/coccl-m1-host \
  CUDA_HOME="$CUDA_HOME" NCCL_HOME="$NCCL_HOME"
