#!/usr/bin/env bash
set -euo pipefail

source_root=${1:?usage: m17_build.sh SOURCE_ROOT}
cuda_root=${M17_CUDA_HOME:-/data/apps/cuda/12.4}
mpi_root=${M17_MPI_HOME:-/data/apps/openmpi/4.1.5_cuda12.8}
current_root=${M17_CURRENT_ROOT:-/data/home/scyb672/run/lxc/COCCL-migrate}
baseline_root=${M17_BASELINE_ROOT:-/data/home/scyb672/run/lxc/COCCL}
temp_root=${M17_TEMP_ROOT:-/data/home/scyb672/run/codex-tmp/coccl-m17-hierarchical}
result_root=${M17_RESULT_ROOT:-$current_root/results/M17}
gencode=${NVCC_GENCODE:-'-gencode=arch=compute_80,code=sm_80'}

current_perf="$temp_root/current-perf"
baseline_perf="$temp_root/baseline-perf"
layout_build="$temp_root/layout"
correctness_root="$temp_root/correctness"

export CUDA_HOME="$cuda_root"
export PATH="$CUDA_HOME/bin:$mpi_root/bin:$PATH"
mkdir -p "$current_perf" "$baseline_perf" "$layout_build" "$result_root"

M16_CUDA_HOME="$cuda_root" M16_MPI_HOME="$mpi_root" \
M16_CURRENT_ROOT="$current_root" M16_BASELINE_ROOT="$baseline_root" \
M16_TEMP_ROOT="$correctness_root" M16_RESULT_ROOT="$result_root" \
NVCC_GENCODE="$gencode" \
  bash "$source_root/examples/benchmarks_scripts/migration/m16_build.sh" \
    "$source_root"

# Make does not track DEBUG or NVCC_GENCODE changes in existing CUDA objects.
# Rebuild this performance-critical plugin once with the M17 release flags.
make -B -C "$source_root/src/coccl-extend/extensions/compressor_plugin/sdp4bit" \
  BUILDDIR="$current_root/build" \
  SUBOBJDIR="$current_root/build/obj/coccl-extend/compressor_plugin" \
  COCCL_ROOT="$source_root/src/coccl-extend" \
  NCCLDIR="$source_root" CUDA_HOME="$CUDA_HOME" \
  NVHPC_CUDA_HOME="$CUDA_HOME" NVCC_GENCODE="$gencode" DEBUG=0 -j8

# The migrated fused DRC path does not support Hadamard. M17 therefore runs
# both implementations with the same non-Hadamard hierarchical encoding.
sed -i 's/^hadamard: 1$/hadamard: 0/' \
  "$correctness_root/legacy-configs/sdp4bit/sdp4bit_RS_Inter.config" \
  "$correctness_root/legacy-configs/sdp4bit/sdp4bit_AR_Inter.config"

current_targets=(
  "$current_perf/reduce_scatter_perf"
  "$current_perf/reduce_scatter_comp_oneshot_perf"
  "$current_perf/reduce_scatter_comp_twoshot_perf"
  "$current_perf/all_reduce_perf"
  "$current_perf/all_reduce_comp_oneshot_perf"
  "$current_perf/all_reduce_comp_twoshot_perf"
  "$current_perf/all_reduce_comp_tripleshot_perf"
)
make -C "$source_root/tests/coccl-tests/src" "${current_targets[@]}" \
  BUILDDIR="$current_perf" NCCL_HOME="$current_root/build" \
  CUDA_HOME="$CUDA_HOME" MPI=1 MPI_HOME="$mpi_root" \
  NVCC_GENCODE="$gencode" DEBUG=0 -j8

baseline_targets=(
  "$baseline_perf/reduce_scatter_comp_twoshot_tl_overlap_perf"
  "$baseline_perf/all_reduce_comp_tripleshot_tl_overlap_perf"
)
make -C "$baseline_root/tests/coccl-tests/src" "${baseline_targets[@]}" \
  BUILDDIR="$baseline_perf" NCCL_HOME="$baseline_root/build" \
  CUDA_HOME="$CUDA_HOME" MPI=1 MPI_HOME="$mpi_root" \
  NVCC_GENCODE="$gencode" DEBUG=0 -j8

make -C "$source_root/tests/coccl-tests/src" \
  "$layout_build/coccl_m5_pipeline_layout_test" \
  BUILDDIR="$layout_build" NCCL_HOME="$current_root/build" \
  CUDA_HOME="$CUDA_HOME" NVCC_GENCODE="$gencode" DEBUG=0 -j8

{
  printf 'M17 hierarchical build manifest\n'
  printf 'current_commit=%s\n' "$(git -C "$source_root" rev-parse HEAD)"
  printf 'baseline_commit=%s\n' "$(git -C "$baseline_root" rev-parse HEAD)"
  printf 'NVCC_GENCODE=%s\n' "$gencode"
  printf 'DEBUG=0\n'
  printf 'MPI_HOME=%s\n' "$mpi_root"
  printf 'sdp4bit_initial_source_hierarchical_hadamard=1\n'
  printf 'sdp4bit_m17_runtime_hierarchical_hadamard=0\n'
  printf 'current_perf=%s\n' "$current_perf"
  printf 'baseline_perf=%s\n' "$baseline_perf"
  printf 'correctness_root=%s\n' "$correctness_root"
  printf 'layout_binary=%s\n' "$layout_build/coccl_m5_pipeline_layout_test"
} >"$result_root/build-manifest.txt"
