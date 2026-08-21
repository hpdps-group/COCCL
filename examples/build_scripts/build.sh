#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
coccl_root=${COCCL_ROOT:-$(cd "$script_dir/../.." && pwd)}
cuda_home=${CUDA_HOME:-${1:-}}
mpi_home=${MPI_HOME:-${2:-}}
cuda_arch=${CUDA_ARCH:-80}
build_jobs=${BUILD_JOBS:-$(nproc)}
nvcc_gencode=${NVCC_GENCODE:-"-gencode=arch=compute_${cuda_arch},code=sm_${cuda_arch}"}

: "${cuda_home:?Set CUDA_HOME or pass the CUDA path as the first argument}"
: "${mpi_home:?Set MPI_HOME or pass the MPI path as the second argument}"

export CUDA_HOME="$cuda_home"
export MPI_HOME="$mpi_home"
export NCCL_HOME="$coccl_root/build"
export PATH="$CUDA_HOME/bin:$MPI_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$NCCL_HOME/lib:$CUDA_HOME/lib64:$MPI_HOME/lib:${LD_LIBRARY_PATH:-}"

echo "Building COCCL in $coccl_root"
echo "CUDA_HOME=$CUDA_HOME MPI_HOME=$MPI_HOME CUDA_ARCH=$cuda_arch"

make -C "$coccl_root" -j "$build_jobs" src.build \
  CUDA_HOME="$CUDA_HOME" NVCC_GENCODE="$nvcc_gencode"

make -C "$coccl_root" -f src/coccl-extend/Makefile coccl-config-check \
  BUILDDIR="$NCCL_HOME" NCCLDIR="$coccl_root" \
  COCCL_ROOT=src/coccl-extend CUDA_HOME="$CUDA_HOME"

make -C "$coccl_root/tests/coccl-tests" -j "$build_jobs" \
  MPI=1 MPI_HOME="$MPI_HOME" CUDA_HOME="$CUDA_HOME" NCCL_HOME="$NCCL_HOME" \
  NVCC_GENCODE="$nvcc_gencode"

echo "COCCL library: $NCCL_HOME/lib/libnccl.so.2"
echo "Configuration checker: $NCCL_HOME/bin/coccl-config-check"
echo "Communication tests: $coccl_root/tests/coccl-tests/build"
