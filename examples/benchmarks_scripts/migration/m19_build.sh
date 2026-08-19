#!/usr/bin/env bash
set -euo pipefail

source_root=${1:?usage: m19_build.sh SOURCE_ROOT}
cuda_root=${M19_CUDA_HOME:-/data/apps/cuda/12.4}
mpi_root=${M19_MPI_HOME:-/data/apps/openmpi/4.1.5_cuda12.8}
current_root=${M19_CURRENT_ROOT:-/data/home/scyb672/run/lxc/COCCL-migrate}
temp_root=${M19_TEMP_ROOT:-/data/home/scyb672/run/codex-tmp/coccl-m19-scope-policy}
result_root=${M19_RESULT_ROOT:-$current_root/results/M19}
runtime_root=${M19_RUNTIME_ROOT:-$temp_root/runtime-build}
plugin_build=${M19_PLUGIN_BUILD:-$temp_root/plugins}
plugin_root="$plugin_build/libcompress"
test_root=${M19_TEST_ROOT:-$temp_root/gpu-tests}
host_root=${M19_HOST_ROOT:-$temp_root/tests}
gencode=${NVCC_GENCODE:-'-gencode=arch=compute_80,code=sm_80'}

export CUDA_HOME="$cuda_root"
export NVHPC_CUDA_HOME="$cuda_root"
export PATH="$cuda_root/bin:$mpi_root/bin:/data/apps/cmake/3.31.12/bin:$PATH"
mkdir -p "$runtime_root" "$plugin_root" "$test_root" "$host_root" \
  "$result_root"

test -f "$runtime_root/obj/device/manifest"

host_sources=(
  collectives.cc init.cc
  misc/coccl_allgather.cc misc/coccl_allreduce.cc
  misc/coccl_alltoall.cc misc/coccl_compressor_size.cc
  misc/coccl_config.cc misc/coccl_config_format.cc
  misc/coccl_config_log.cc misc/coccl_explicit.cc
  misc/coccl_frame_exchange.cc misc/coccl_group.cc
  misc/coccl_hierarchical_reduction.cc misc/coccl_operation.cc
  misc/coccl_pipeline_execute.cc misc/coccl_pipeline_plan.cc
  misc/coccl_pipeline_stage.cc misc/coccl_prepared_call.cc
  misc/coccl_reducescatter.cc misc/coccl_runtime.cc
  misc/coccl_sendrecv.cc misc/compress.cc misc/nccl_comp_wrapper.cc
  misc/tuning/coccl_autotune_cost_model.cc
  misc/tuning/coccl_autotune_profiler.cc
  misc/tuning/coccl_autotune_selector.cc
)
host_objects=()
for source in "${host_sources[@]}"; do
  host_objects+=("$runtime_root/obj/${source%.cc}.o")
done
make -C "$source_root/src" "${host_objects[@]}" \
  BUILDDIR="$runtime_root" CUDA_HOME="$cuda_root" DEBUG=0 -j8

read -r -a device_objects <"$runtime_root/obj/device/manifest"
mapfile -t all_host_objects < <(
  find "$runtime_root/obj" -type f -name '*.o' ! -path '*/device/*' | sort
)
lib_target="$runtime_root/lib/libnccl.so.2.21.5"
g++ -std=c++17 -fPIC -shared -Wl,--no-as-needed \
  -Wl,-soname,libnccl.so.2 -o "$lib_target" \
  "${all_host_objects[@]}" "${device_objects[@]}" \
  -L"$cuda_root/lib64" -lcudart_static -lpthread -lrt -ldl
ln -sfn libnccl.so.2.21.5 "$runtime_root/lib/libnccl.so.2"
ln -sfn libnccl.so.2 "$runtime_root/lib/libnccl.so"
rm -f "$runtime_root/lib/libnccl_static.a"
ar cr "$runtime_root/lib/libnccl_static.a" \
  "${all_host_objects[@]}" "${device_objects[@]}"

for plugin in sdp4bit taco tahquant zfp dietgpu; do
  make -C "$source_root/src/device/compress/$plugin" \
    BUILDDIR="$runtime_root" SUBOBJDIR="$plugin_build" \
    NCCLDIR="$source_root" CUDA_HOME="$cuda_root" \
    NVHPC_CUDA_HOME="$cuda_root" NVCC_GENCODE="$gencode" DEBUG=0 -j8
done

gpu_targets=(
  "$test_root/coccl_m16_correctness"
  "$test_root/coccl_dietgpu_integer_test"
  "$test_root/coccl_m14_dietgpu_codec_test"
  "$test_root/reduce_scatter_perf"
  "$test_root/reduce_scatter_comp_twoshot_perf"
  "$test_root/all_reduce_perf"
  "$test_root/all_reduce_comp_tripleshot_perf"
)
make -C "$source_root/tests/coccl-tests/src" "${gpu_targets[@]}" \
  BUILDDIR="$test_root" NCCL_HOME="$runtime_root" CUDA_HOME="$cuda_root" \
  MPI=1 MPI_HOME="$mpi_root" NVCC_GENCODE="$gencode" DEBUG=0 -j8

host_suites=(
  m1-host-tests m2-host-tests
  m5-host-tests m6-host-tests m7-host-tests m8-host-tests
  m9-host-tests m10-host-tests m11-host-tests m12-host-tests
  m13-host-tests m14-host-tests m15-host-tests
  m18-host-tests m19-host-tests
)
make -C "$source_root/tests/coccl-tests/src" "${host_suites[@]}" \
  BUILDDIR="$host_root" NCCL_HOME="$runtime_root" CUDA_HOME="$cuda_root" \
  M1_PLUGIN_DIR="$plugin_root"
mkdir -p "$temp_root/config-tests"
"$host_root/coccl_m19_scope_config_test" "$temp_root/config-tests"
"$host_root/coccl_m19_hierarchical_graph_test"
"$host_root/coccl_m19_codec_flow_test"
"$host_root/coccl_m19_selector_scope_test"

for plugin in dietgpu sdp4bit taco tahquant zfp; do
  "$host_root/coccl_m1_plugin_load_test" "$plugin_root" "$plugin"
done

{
  printf 'M19 scope-policy build manifest\n'
  printf 'commit=%s\n' "$(git -C "$source_root" rev-parse HEAD)"
  printf 'runtime_root=%s\n' "$runtime_root"
  printf 'plugin_root=%s\n' "$plugin_root"
  printf 'test_root=%s\n' "$test_root"
  printf 'NVCC_GENCODE=%s\n' "$gencode"
  printf 'MPI_HOME=%s\n' "$mpi_root"
  printf 'host_tests=M19 PASS\n'
  printf 'plugins=dietgpu,sdp4bit,taco,tahquant,zfp\n'
} >"$result_root/build-manifest.txt"
