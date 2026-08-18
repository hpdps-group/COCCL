#!/usr/bin/env bash
set -euo pipefail

source_root=${1:?usage: m9_build.sh SOURCE_ROOT}
cuda_root=${M9_CUDA_HOME:-/data/apps/cuda/12.4}
gencode=${NVCC_GENCODE:-'-gencode=arch=compute_80,code=sm_80'}
build_root=${M9_BUILD_ROOT:-$source_root/build}
test_build=${M9_TESTS_DIR:-$source_root/tests/coccl-tests/build}
host_test_build=${M9_HOST_TEST_DIR:-/data/home/scyb672/run/codex-tmp/coccl-m9-host}
plugin_work=${M9_PLUGIN_WORK:-/data/home/scyb672/run/codex-tmp/coccl-m9-build}
result_root=${M9_RESULT_ROOT:-/data/home/scyb672/run/lxc/COCCL-migrate/results/M9}

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
mkdir -p "$plugin_work" "$result_root"

make -C "$source_root/src" "$build_root/include/nccl.h" \
  BUILDDIR="$build_root" CUDA_HOME="$CUDA_HOME" \
  NVCC_GENCODE="$gencode" DEBUG=0
make -C "$source_root/src" \
  "$build_root/obj/misc/compress.o" "$build_root/obj/init.o" \
  BUILDDIR="$build_root" CUDA_HOME="$CUDA_HOME" \
  NVCC_GENCODE="$gencode" DEBUG=0 -j8

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

make -C "$source_root/src/device/compress/sdp4bit" -j8 \
  BUILDDIR="$build_root" SUBOBJDIR="$plugin_work/sdp4bit" \
  CUDA_HOME="$CUDA_HOME" NVCC_GENCODE="$gencode"
make -C "$source_root/src/device/compress/tahquant" -j8 \
  BUILDDIR="$build_root" SUBOBJDIR="$plugin_work/tahquant" \
  CUDA_HOME="$CUDA_HOME" NVCC_GENCODE="$gencode"
make -C "$source_root/src/device/compress/taco" -j8 \
  BUILDDIR="$build_root" SUBOBJDIR="$plugin_work/taco" \
  CUDA_HOME="$CUDA_HOME" NVCC_GENCODE="$gencode"
make -C "$source_root/src/device/compress/zfp" zfp.build \
  BUILDDIR="$build_root" SUBOBJDIR="$plugin_work/zfp" \
  CUDA_HOME="$CUDA_HOME" NVCC_GENCODE="$gencode"

plugin_dir="$build_root/obj/device/compress/libcompress"
mkdir -p "$plugin_dir"
install -m 755 "$plugin_work/sdp4bit/libcompress/libsdp4bit.so" \
  "$plugin_dir/libsdp4bit.so"
install -m 755 "$plugin_work/tahquant/libcompress/libtahquant.so" \
  "$plugin_dir/libtahquant.so"
install -m 755 "$plugin_work/taco/libcompress/libtaco.so" \
  "$plugin_dir/libtaco.so"
install -m 755 "$plugin_work/zfp/libcompress/libzfp.so" \
  "$plugin_dir/libzfp.so"

export NCCL_HOME="$build_root"
test_targets=(
  alltoall_p2p_perf alltoall_comp_perf
  all_gather_perf all_gather_comp_perf
  reduce_scatter_perf reduce_scatter_comp_oneshot_perf
  reduce_scatter_comp_twoshot_perf
  all_reduce_perf all_reduce_comp_oneshot_perf
  all_reduce_comp_twoshot_perf all_reduce_comp_tripleshot_perf
)
target_paths=()
for target in "${test_targets[@]}"; do
  target_paths+=("$test_build/$target")
done
make -C "$source_root/tests/coccl-tests/src" -j4 \
  BUILDDIR="$test_build" CUDA_HOME="$CUDA_HOME" NCCL_HOME="$NCCL_HOME" \
  NVCC_GENCODE="$gencode" DEBUG=0 "${target_paths[@]}"

for target in m1-host-tests m1-plugin-load-test m2-host-tests \
              m3-host-tests m4-host-tests m5-host-tests m6-host-tests \
              m7-host-tests m8-host-tests m9-host-tests; do
  make -C "$source_root/tests/coccl-tests/src" "$target" \
    BUILDDIR="$host_test_build" CUDA_HOME="$CUDA_HOME" \
    NCCL_HOME="$NCCL_HOME" DEBUG=0
done

{
  printf 'M9 fixed compressor build manifest\n'
  printf 'CUDA_HOME=%s\nNVCC_GENCODE=%s\n' "$CUDA_HOME" "$gencode"
  for name in sdp4bit tahquant taco zfp; do
    library="$plugin_dir/lib${name}.so"
    printf '%s bytes=%s entry=' "$name" "$(stat -c %s "$library")"
    nm -D --defined-only "$library" | \
      awk '$3 == "cocclGetCompressorPlugin" {print $3}'
    "$CUDA_HOME/bin/cuobjdump" --list-elf "$library" | \
      sed "s/^/${name} /"
  done
} >"$result_root/build-manifest.txt"
