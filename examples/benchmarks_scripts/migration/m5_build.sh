#!/usr/bin/env bash
set -euo pipefail

source_root=${1:?usage: m5_build.sh SOURCE_ROOT}
cuda_root=${CUDA_HOME:-/data/apps/cuda/12.4}
gencode=${NVCC_GENCODE:-'-gencode=arch=compute_80,code=sm_80'}
build_root=${M5_BUILD_ROOT:-$source_root/build}
test_build=${M5_TESTS_DIR:-$source_root/tests/coccl-tests/build}
host_test_build=${M5_HOST_TEST_DIR:-/data/home/scyb672/run/codex-tmp/coccl-m5-host}

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
  BUILDDIR="$build_root" NVCC_GENCODE="$gencode" DEBUG=0

host_objects=(
  "$build_root/obj/init.o"
  "$build_root/obj/misc/coccl_buffer_management.o"
  "$build_root/obj/misc/coccl_buffer_legacy.o"
  "$build_root/obj/misc/coccl_buffer_vmm.o"
  "$build_root/obj/misc/coccl_runtime.o"
  "$build_root/obj/misc/nccl_comp_wrapper.o"
  "$build_root/obj/misc/coccl_alltoall.o"
  "$build_root/obj/misc/coccl_pipeline_plan.o"
  "$build_root/obj/misc/coccl_pipeline_stage.o"
  "$build_root/obj/misc/coccl_pipeline_execute.o"
)
make -C "$source_root/src" "${host_objects[@]}" \
  BUILDDIR="$build_root" NVCC_GENCODE="$gencode" DEBUG=0

layout_object="$build_root/obj/device/pipeline/coccl_pipeline_layout.cu.o"
make -C "$source_root/src/device" "$layout_object" \
  BUILDDIR="$build_root" NVCC_GENCODE="$gencode" DEBUG=0

read -r -a current_device_objects <"$manifest"
device_objects=()
for object in "${current_device_objects[@]}"; do
  [[ "$object" == */device_glue.o || \
     "$object" == */pipeline/coccl_pipeline_layout.cu.o ]] || \
    device_objects+=("$object")
done
device_glue="$build_root/obj/device/device_glue.o"
"$CUDA_HOME/bin/nvcc" $gencode --compiler-options \
  '-fPIC -fvisibility=hidden' -dlink "${device_objects[@]}" \
  -o "$device_glue"
printf '%s ' "${device_objects[@]}" "$layout_object" "$device_glue" >"$manifest"
printf '\n' >>"$manifest"
device_objects+=("$layout_object" "$device_glue")

mapfile -t all_host_objects < <(
  find "$build_root/obj" -type f -name '*.o' ! -path '*/device/*' | sort
)
lib_target="$build_root/lib/libnccl.so.2.21.5"
g++ -std=c++17 -fPIC -shared -Wl,--no-as-needed \
  -Wl,-soname,libnccl.so.2 -o "$lib_target" \
  "${all_host_objects[@]}" "${device_objects[@]}" \
  -L"$CUDA_HOME/lib64" -lcudart_static -lpthread -lrt -ldl
ln -sfn libnccl.so.2.21.5 "$build_root/lib/libnccl.so.2"
ln -sfn libnccl.so.2 "$build_root/lib/libnccl.so"
rm -f "$build_root/lib/libnccl_static.a"
ar cr "$build_root/lib/libnccl_static.a" \
  "${all_host_objects[@]}" "${device_objects[@]}"

plugin_dir="$build_root/obj/device/compress/libcompress"
test -f "$plugin_dir/libsdp4bit.so"
test -f "$plugin_dir/libzfp.so"

export NCCL_HOME="$build_root"
make -C "$source_root/tests/coccl-tests/src" -j4 \
  BUILDDIR="$test_build" CUDA_HOME="$CUDA_HOME" NCCL_HOME="$NCCL_HOME" \
  NVCC_GENCODE="$gencode" DEBUG=0 \
  "$test_build/alltoall_p2p_perf" \
  "$test_build/alltoall_comp_perf" \
  "$test_build/coccl_m5_pipeline_layout_test"

for target in m1-host-tests m1-plugin-load-test m2-host-tests \
              m3-host-tests m4-host-tests m5-host-tests; do
  make -C "$source_root/tests/coccl-tests/src" "$target" \
    BUILDDIR="$host_test_build" CUDA_HOME="$CUDA_HOME" NCCL_HOME="$NCCL_HOME" \
    DEBUG=0
done

nm -D --defined-only "$lib_target" | grep -E ' pcocclAllToAllComp$' >/dev/null
if nm -D --defined-only "$lib_target" | \
    grep -E ' pncclAllToAllComp$| pncclAlltoAllCompOverlap$' >/dev/null; then
  echo "retired AllToAll Comp symbol remains" >&2
  exit 1
fi
