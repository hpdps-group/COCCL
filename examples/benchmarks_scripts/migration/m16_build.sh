#!/usr/bin/env bash
set -euo pipefail

source_root=${1:?usage: m16_build.sh SOURCE_ROOT}
cuda_root=${M16_CUDA_HOME:-/data/apps/cuda/12.4}
mpi_root=${M16_MPI_HOME:-/data/apps/openmpi/4.1.5_cuda12.8}
current_root=${M16_CURRENT_ROOT:-/data/home/scyb672/run/lxc/COCCL-migrate}
baseline_root=${M16_BASELINE_ROOT:-/data/home/scyb672/run/lxc/COCCL}
temp_root=${M16_TEMP_ROOT:-/data/home/scyb672/run/codex-tmp/coccl-m16-correctness}
result_root=${M16_RESULT_ROOT:-$current_root/results/M16}
gencode=${NVCC_GENCODE:-'-gencode=arch=compute_80,code=sm_80'}

current_binary="$temp_root/current/coccl_m16_correctness"
baseline_binary="$temp_root/baseline/coccl_m16_correctness"
legacy_configs="$temp_root/legacy-configs"
legacy_subadd_configs="$temp_root/legacy-subadd-configs"

export CUDA_HOME="$cuda_root"
export PATH="$CUDA_HOME/bin:$mpi_root/bin:$PATH"
mkdir -p "$(dirname "$current_binary")" "$(dirname "$baseline_binary")" \
  "$legacy_configs" "$legacy_subadd_configs" "$result_root"

make -C "$source_root/tests/coccl-tests/src" m16-correctness-build \
  BUILDDIR="$(dirname "$current_binary")" \
  NCCL_HOME="$current_root/build" CUDA_HOME="$CUDA_HOME" \
  MPI=1 MPI_HOME="$mpi_root" NVCC_GENCODE="$gencode" DEBUG=0 -j8

"$CUDA_HOME/bin/nvcc" -ccbin g++ $gencode -std=c++11 -O3 -g \
  -DCOCCL_M16_LEGACY_API \
  -I"$baseline_root/build/include" -I"$mpi_root/include" \
  "$source_root/tests/coccl-tests/src/coccl_m16_correctness.cu" \
  -L"$baseline_root/build/lib" -L"$mpi_root/lib" \
  -lnccl -lmpi -lcudart -lrt -o "$baseline_binary"

cp -a "$baseline_root/examples/benchmarks_scripts/configs/." "$legacy_configs/"
cp -a "$baseline_root/examples/benchmarks_scripts/configs/." \
  "$legacy_subadd_configs/"
cp "$legacy_configs/sdp4bit/sdp4bit_AG.config" \
  "$legacy_configs/sdp4bit/sdp4bit_SR.config"
cp "$legacy_configs/sdp4bit/sdp4bit_AG.config" \
  "$legacy_configs/sdp4bit/sdp4bit_SR_BWD.config"
cp "$baseline_root/examples/training_scripts/configs_training/sdp4bit/sdp4bit_AG.config" \
  "$legacy_subadd_configs/sdp4bit/sdp4bit_AG.config"

{
  printf 'M16 correctness build manifest\n'
  printf 'NVCC_GENCODE=%s\n' "$gencode"
  printf 'MPI_HOME=%s\n' "$mpi_root"
  printf 'current_binary=%s\n' "$current_binary"
  printf 'baseline_binary=%s\n' "$baseline_binary"
  printf 'elements_per_rank=4096\n'
} >"$result_root/build-manifest.txt"
