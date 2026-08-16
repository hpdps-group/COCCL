#!/usr/bin/env bash
set -euo pipefail

source_root=${1:?usage: m8_build.sh SOURCE_ROOT}
cuda_root=${M8_CUDA_HOME:-/data/apps/cuda/12.4}
gencode=${NVCC_GENCODE:-'-gencode=arch=compute_80,code=sm_80'}
build_root=${M8_BUILD_ROOT:-$source_root/build}
test_build=${M8_TESTS_DIR:-$source_root/tests/coccl-tests/build}
host_test_build=${M8_HOST_TEST_DIR:-/data/home/scyb672/run/codex-tmp/coccl-m8-host}
zfp_objdir=${M8_ZFP_OBJDIR:-/data/home/scyb672/run/codex-tmp/coccl-m8-zfp}

if type module >/dev/null 2>&1; then
  module unload cuda >/dev/null 2>&1 || true
  module load cuda/12.4
  module load cmake/3.31.12
fi

export CUDA_HOME="$cuda_root"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
if [[ -x /data/apps/cmake/3.31.12/bin/cmake ]]; then
  export PATH="/data/apps/cmake/3.31.12/bin:$PATH"
fi
export LD_LIBRARY_PATH="$build_root/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

manifest="$build_root/obj/device/manifest"
test -f "$manifest"

make -C "$source_root/src" "$build_root/include/nccl.h" \
  BUILDDIR="$build_root" NVCC_GENCODE="$gencode" DEBUG=0

host_objects=(
  "$build_root/obj/collectives.o"
  "$build_root/obj/init.o"
  "$build_root/obj/misc/coccl_buffer_management.o"
  "$build_root/obj/misc/coccl_buffer_legacy.o"
  "$build_root/obj/misc/coccl_buffer_vmm.o"
  "$build_root/obj/misc/compress.o"
  "$build_root/obj/misc/coccl_group.o"
  "$build_root/obj/misc/coccl_runtime.o"
  "$build_root/obj/misc/coccl_explicit.o"
  "$build_root/obj/misc/nccl_comp_wrapper.o"
  "$build_root/obj/misc/coccl_alltoall.o"
  "$build_root/obj/misc/coccl_allgather.o"
  "$build_root/obj/misc/coccl_reducescatter.o"
  "$build_root/obj/misc/coccl_allreduce.o"
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
make -C "$source_root/src/device/compress/zfp" zfp.build \
  BUILDDIR="$build_root" SUBOBJDIR="$zfp_objdir" \
  CUDA_HOME="$CUDA_HOME" NVCC_GENCODE="$gencode"
cp "$zfp_objdir/libcompress/libzfp.so" "$plugin_dir/libzfp.so"
test -f "$plugin_dir/libzfp.so"

export NCCL_HOME="$build_root"
make -C "$source_root/tests/coccl-tests/src" -j4 \
  BUILDDIR="$test_build" CUDA_HOME="$CUDA_HOME" NCCL_HOME="$NCCL_HOME" \
  NVCC_GENCODE="$gencode" DEBUG=0 \
  "$test_build/all_reduce_perf" \
  "$test_build/all_reduce_comp_oneshot_perf" \
  "$test_build/all_reduce_comp_twoshot_perf" \
  "$test_build/all_reduce_comp_tripleshot_perf" \
  "$test_build/coccl_m5_pipeline_layout_test"

for target in m1-host-tests m1-plugin-load-test m2-host-tests \
              m3-host-tests m4-host-tests m5-host-tests m6-host-tests \
              m7-host-tests m8-host-tests; do
  make -C "$source_root/tests/coccl-tests/src" "$target" \
    BUILDDIR="$host_test_build" CUDA_HOME="$CUDA_HOME" NCCL_HOME="$NCCL_HOME" \
    DEBUG=0
done

for symbol in OneShot TwoShot TripleShot; do
  nm -D --defined-only "$lib_target" |
    grep -E " pcocclAllReduceComp${symbol}$" >/dev/null
done
if nm -D --defined-only "$lib_target" |
    grep -E ' p?ncclAllReduceComp(TwoShotOverlap|TripleShotTLOverlap|Ring|Optim)$' >/dev/null; then
  echo "retired AllReduce symbol is still exported" >&2
  exit 1
fi
if grep -En 'ncclAllReduceComp(TwoShotOverlap|TripleShotTLOverlap|Ring|Optim)' \
    "$source_root/src/misc/nccl_comp_wrapper.cc" \
    "$source_root/src/nccl.h.in" >/dev/null; then
  echo "retired AllReduce source is still present" >&2
  exit 1
fi
