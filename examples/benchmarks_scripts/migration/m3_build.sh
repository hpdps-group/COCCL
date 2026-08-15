#!/usr/bin/env bash
set -euo pipefail

source_root=${1:?usage: m3_build.sh SOURCE_ROOT}
cuda_root=${CUDA_HOME:-/data/apps/cuda/12.4}
gencode=${NVCC_GENCODE:-'-gencode=arch=compute_80,code=sm_80'}
build_root=${M3_BUILD_ROOT:-$source_root/build}
test_build=${M3_TESTS_DIR:-$source_root/tests/coccl-tests/build}
host_test_build=${M3_HOST_TEST_DIR:-/data/home/scyb672/run/codex-tmp/coccl-m3-host}

if type module >/dev/null 2>&1; then
  module unload cuda >/dev/null 2>&1 || true
  module load cuda/12.4
fi

export CUDA_HOME="$cuda_root"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$build_root/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

manifest="$build_root/obj/device/manifest"
test -f "$manifest"

make -C "$source_root/src" "$build_root/include/nccl.h" \
  BUILDDIR="$build_root" NVCC_GENCODE="$gencode"

host_objects=(
  "$build_root/obj/init.o"
  "$build_root/obj/misc/compress.o"
  "$build_root/obj/misc/nccl_comp_wrapper.o"
  "$build_root/obj/misc/coccl_buffer_management.o"
  "$build_root/obj/misc/coccl_buffer_legacy.o"
)
make -C "$source_root/src" "${host_objects[@]}" \
  BUILDDIR="$build_root" NVCC_GENCODE="$gencode"

mapfile -t all_host_objects < <(
  find "$build_root/obj" -type f -name '*.o' ! -path '*/device/*' | sort
)
read -r -a device_objects <"$manifest"
lib_target="$build_root/lib/libnccl.so.2.21.5"
g++ -std=c++17 -fPIC -shared -Wl,--no-as-needed \
  -Wl,-soname,libnccl.so.2 -o "$lib_target" \
  "${all_host_objects[@]}" "${device_objects[@]}" \
  -L"$CUDA_HOME/lib64" -lcudart_static -lpthread -lrt -ldl
ln -sfn libnccl.so.2.21.5 "$build_root/lib/libnccl.so.2"
ln -sfn libnccl.so.2 "$build_root/lib/libnccl.so"
ar cr "$build_root/lib/libnccl_static.a" \
  "${all_host_objects[@]}" "${device_objects[@]}"

plugin_dir="$build_root/obj/device/compress/libcompress"
test -f "$plugin_dir/libsdp4bit.so"
test -f "$plugin_dir/libzfp.so"

export NCCL_HOME="$build_root"
make -C "$source_root/tests/coccl-tests/src" -j8 \
  BUILDDIR="$test_build" CUDA_HOME="$CUDA_HOME" NCCL_HOME="$NCCL_HOME" \
  NVCC_GENCODE="$gencode" \
  "$test_build/alltoall_p2p_perf" \
  "$test_build/alltoall_comp_overlap_perf"

make -C "$source_root/tests/coccl-tests/src" m1-host-tests \
  BUILDDIR="$host_test_build" CUDA_HOME="$CUDA_HOME" NCCL_HOME="$NCCL_HOME"
make -C "$source_root/tests/coccl-tests/src" m1-plugin-load-test \
  BUILDDIR="$host_test_build" CUDA_HOME="$CUDA_HOME" NCCL_HOME="$NCCL_HOME"
make -C "$source_root/tests/coccl-tests/src" m2-host-tests \
  BUILDDIR="$host_test_build" CUDA_HOME="$CUDA_HOME" NCCL_HOME="$NCCL_HOME"
make -C "$source_root/tests/coccl-tests/src" m3-host-tests \
  BUILDDIR="$host_test_build" CUDA_HOME="$CUDA_HOME" NCCL_HOME="$NCCL_HOME"

nm -D --defined-only "$lib_target" | rg -q ' pcocclAllToAllComp$'
