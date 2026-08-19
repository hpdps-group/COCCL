#!/usr/bin/env bash
set -euo pipefail

source_root=${1:?usage: m14_build.sh SOURCE_ROOT}
cuda_root=${M14_CUDA_HOME:-/data/apps/cuda/12.4}
gencode=${NVCC_GENCODE:-'-gencode=arch=compute_80,code=sm_80'}
build_root=${M14_BUILD_ROOT:-/data/home/scyb672/run/lxc/COCCL-migrate/build}
test_build=${M14_TESTS_DIR:-/data/home/scyb672/run/lxc/COCCL-migrate/tests/coccl-tests/build}
host_test_build=${M14_HOST_TEST_DIR:-/data/home/scyb672/run/codex-tmp/coccl-m14-host}
result_root=${M14_RESULT_ROOT:-/data/home/scyb672/run/lxc/COCCL-migrate/results/M14}

if type module >/dev/null 2>&1; then
  module unload cuda >/dev/null 2>&1 || true
  module load cuda/12.4
fi
export CUDA_HOME="$cuda_root"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$build_root/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

manifest="$build_root/obj/device/manifest"
test -f "$manifest"
mkdir -p "$host_test_build" "$result_root"

objects=(
  "$build_root/obj/misc/coccl_frame_exchange.o"
  "$build_root/obj/misc/coccl_pipeline_plan.o"
  "$build_root/obj/misc/coccl_pipeline_stage.o"
  "$build_root/obj/misc/coccl_pipeline_execute.o"
  "$build_root/obj/misc/compress.o"
)
make -C "$source_root/src" "${objects[@]}" \
  BUILDDIR="$build_root" CUDA_HOME="$CUDA_HOME" \
  NVCC_GENCODE="$gencode" DEBUG=0 -j8
make -C "$source_root/src/device/compress/dietgpu" \
  BUILDDIR="$build_root" \
  SUBOBJDIR="$build_root/obj/device/compress" \
  NCCLDIR="$source_root" CUDA_HOME="$CUDA_HOME" \
  NVCC_GENCODE="$gencode" -j8

read -r -a device_objects <"$manifest"
mapfile -t host_objects < <(
  find "$build_root/obj" -type f -name '*.o' ! -path '*/device/*' | sort
)
lib_target="$build_root/lib/libnccl.so.2.21.5"
g++ -std=c++17 -fPIC -shared -Wl,--no-as-needed \
  -Wl,-soname,libnccl.so.2 -o "$lib_target" \
  "${host_objects[@]}" "${device_objects[@]}" \
  -L"$CUDA_HOME/lib64" -lcudart_static -lpthread -lrt -ldl
ln -sfn libnccl.so.2.21.5 "$build_root/lib/libnccl.so.2"
ln -sfn libnccl.so.2 "$build_root/lib/libnccl.so"
rm -f "$build_root/lib/libnccl_static.a"
ar cr "$build_root/lib/libnccl_static.a" \
  "${host_objects[@]}" "${device_objects[@]}"

test_targets=(
  sendrecv_comp alltoall_comp all_gather_comp
  reduce_scatter_comp_oneshot reduce_scatter_comp_twoshot
  all_reduce_comp_oneshot all_reduce_comp_twoshot
  all_reduce_comp_tripleshot
)
target_paths=()
for target in "${test_targets[@]}"; do
  target_paths+=("$test_build/${target}_perf")
done
make -C "$source_root/tests/coccl-tests/src" \
  "${target_paths[@]}" m14-codec-build \
  BUILDDIR="$test_build" NCCL_HOME="$build_root" CUDA_HOME="$CUDA_HOME" \
  NVCC_GENCODE="$gencode" -j8

host_targets=(
  m1-host-tests m1-plugin-load-test m2-host-tests m3-host-tests
  m4-host-tests m5-host-tests m6-host-tests m7-host-tests
  m8-host-tests m9-host-tests m10-host-tests m11-host-tests
  m12-host-tests m13-host-tests m14-host-tests
)
make -C "$source_root/tests/coccl-tests/src" \
  "${host_targets[@]}" \
  BUILDDIR="$host_test_build" NCCL_HOME="$build_root" \
  CUDA_HOME="$CUDA_HOME" NVCC_GENCODE="$gencode"

{
  printf 'M14 dietGPU framed pipeline build manifest\n'
  printf 'dietGPU_commit=a4d70a14066d2c3e5fe1849b3723e4cd423eee7e\n'
  printf 'glog_commit=6434410145ee40e7ffe32f97ab54d24d25f0459d\n'
  printf 'NVCC_GENCODE=%s\n' "$gencode"
  printf 'host_objects=%s\n' "${objects[*]}"
  printf 'gpu_tests=%s codec\n' "${test_targets[*]}"
  printf 'host_tests=M1-M14 PASS\n'
} >"$result_root/build-manifest.txt"
