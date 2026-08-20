#!/usr/bin/env bash
set -euo pipefail

source_root=${1:?usage: m20_build.sh SOURCE_ROOT}
cuda_root=${M20_CUDA_HOME:-/data/apps/cuda/12.8}
mpi_root=${M20_MPI_HOME:-/data/apps/openmpi/4.1.5_cuda12.8}
current_root=${M20_CURRENT_ROOT:-/data/home/scyb672/run/lxc/COCCL-migrate}
temp_root=${M20_TEMP_ROOT:-/data/home/scyb672/run/codex-tmp/coccl-m20-remainder-slices}
result_root=${M20_RESULT_ROOT:-$current_root/results/M20}
runtime_root=${M20_RUNTIME_ROOT:-/data/home/scyb672/run/codex-tmp/coccl-m19-scope-policy/runtime-build}
plugin_build=${M20_PLUGIN_BUILD:-/data/home/scyb672/run/codex-tmp/coccl-m19-scope-policy/plugins}
plugin_root=${M20_PLUGIN_ROOT:-$plugin_build/libcompress}
test_root=${M20_TEST_ROOT:-$temp_root/gpu-tests}
host_root=${M20_HOST_ROOT:-$temp_root/tests}
gencode=${NVCC_GENCODE:-'-gencode=arch=compute_80,code=sm_80'}

export CUDA_HOME="$cuda_root"
export PATH="$cuda_root/bin:$mpi_root/bin:/data/apps/cmake/3.31.12/bin:$PATH"
mkdir -p "$test_root" "$host_root" "$result_root"
test -f "$runtime_root/obj/device/manifest"
test -d "$plugin_root"

changed_host_objects=(
  "$runtime_root/obj/misc/coccl_pipeline_execute.o"
  "$runtime_root/obj/misc/coccl_pipeline_plan.o"
  "$runtime_root/obj/misc/coccl_pipeline_stage.o"
  "$runtime_root/obj/misc/coccl_frame_exchange.o"
  "$runtime_root/obj/misc/compress.o"
)
layout_object="$runtime_root/obj/device/pipeline/coccl_pipeline_layout.cu.o"

if [[ ${M20_REBUILD_RELEASE_CORE:-0} == 1 ]]; then
  mapfile -t host_objects < <(
    find "$runtime_root/obj" -type f -name '*.o' \
      ! -path '*/device/*' | sort
  )
  make -B -C "$source_root/src" "${host_objects[@]}" \
    BUILDDIR="$runtime_root" CUDA_HOME="$cuda_root" DEBUG=0 -j8
  make -B -C "$source_root/src/device" \
    "$runtime_root/obj/device/manifest" \
    BUILDDIR="$runtime_root" CUDA_HOME="$cuda_root" \
    NVCC_GENCODE="$gencode" DEBUG=0 -j8
else
  make -C "$source_root/src" "${changed_host_objects[@]}" \
    BUILDDIR="$runtime_root" CUDA_HOME="$cuda_root" DEBUG=0 -j4
  make -C "$source_root/src/device" "$layout_object" \
    BUILDDIR="$runtime_root" CUDA_HOME="$cuda_root" \
    NVCC_GENCODE="$gencode" DEBUG=0
fi

read -r -a device_objects <"$runtime_root/obj/device/manifest"
mapfile -t host_objects < <(
  find "$runtime_root/obj" -type f -name '*.o' \
    ! -path '*/device/*' | sort
)
lib_target="$runtime_root/lib/libnccl.so.2.21.5"
g++ -std=c++17 -fPIC -shared -Wl,--no-as-needed \
  -Wl,-soname,libnccl.so.2 -o "$lib_target" \
  "${host_objects[@]}" "${device_objects[@]}" \
  -L"$cuda_root/lib64" -lcudart_static -lpthread -lrt -ldl
ln -sfn libnccl.so.2.21.5 "$runtime_root/lib/libnccl.so.2"
ln -sfn libnccl.so.2 "$runtime_root/lib/libnccl.so"
ar crs "$runtime_root/lib/libnccl_static.a" \
  "${host_objects[@]}" "${device_objects[@]}"

for plugin in sdp4bit dietgpu; do
  make -C "$source_root/src/device/compress/$plugin" \
    BUILDDIR="$runtime_root" \
    SUBOBJDIR="$plugin_build" \
    NCCLDIR="$source_root" CUDA_HOME="$cuda_root" \
    NVHPC_CUDA_HOME="$cuda_root" NVCC_GENCODE="$gencode" DEBUG=0 -j8
done

gpu_targets=(
  "$test_root/coccl_m16_correctness"
  "$test_root/coccl_m5_pipeline_layout_test"
  "$test_root/alltoall_perf"
  "$test_root/alltoall_comp_perf"
  "$test_root/all_gather_perf"
  "$test_root/all_gather_comp_perf"
  "$test_root/reduce_scatter_perf"
  "$test_root/reduce_scatter_comp_oneshot_perf"
  "$test_root/reduce_scatter_comp_twoshot_perf"
  "$test_root/all_reduce_perf"
  "$test_root/all_reduce_comp_twoshot_perf"
  "$test_root/all_reduce_comp_tripleshot_perf"
)
make -C "$source_root/tests/coccl-tests/src" "${gpu_targets[@]}" \
  BUILDDIR="$test_root" NCCL_HOME="$runtime_root" CUDA_HOME="$cuda_root" \
  MPI=1 MPI_HOME="$mpi_root" NVCC_GENCODE="$gencode" DEBUG=0 -j8

make -C "$source_root/tests/coccl-tests/src" m20-host-tests \
  BUILDDIR="$host_root" NCCL_HOME="$runtime_root" CUDA_HOME="$cuda_root" \
  NVCC_GENCODE="$gencode" DEBUG=0

{
  printf 'M20 remainder build manifest\n'
  printf 'commit=%s\n' "$(git -C "$source_root" rev-parse HEAD)"
  printf 'runtime_root=%s\n' "$runtime_root"
  printf 'plugin_root=%s\n' "$plugin_root"
  printf 'test_root=%s\n' "$test_root"
  printf 'NVCC_GENCODE=%s\n' "$gencode"
  printf 'release_core_rebuilt=%s\n' "${M20_REBUILD_RELEASE_CORE:-0}"
} >"$result_root/build-manifest.txt"
