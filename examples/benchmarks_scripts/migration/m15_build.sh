#!/usr/bin/env bash
set -euo pipefail

source_root=${1:?usage: m15_build.sh SOURCE_ROOT}
cuda_root=${M15_CUDA_HOME:-/data/apps/cuda/12.4}
gencode=${NVCC_GENCODE:-'-gencode=arch=compute_80,code=sm_80'}
build_root=${M15_BUILD_ROOT:-/data/home/scyb672/run/lxc/COCCL-migrate/build}
host_build=${M15_HOST_TEST_DIR:-/data/home/scyb672/run/codex-tmp/coccl-m15-host}
gpu_build=${M15_GPU_TEST_DIR:-/data/home/scyb672/run/codex-tmp/coccl-m15-gpu}
temp_root=${M15_TEMP_ROOT:-/data/home/scyb672/run/codex-tmp/coccl-m15-pack-swizzle}
result_root=${M15_RESULT_ROOT:-/data/home/scyb672/run/lxc/COCCL-migrate/results/M15}
baseline_commit=${M15_BASELINE_COMMIT:-1439462}

export CUDA_HOME="$cuda_root"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$build_root/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

manifest="$build_root/obj/device/manifest"
test -f "$manifest"
mkdir -p "$host_build" "$gpu_build" "$temp_root" "$result_root"

host_objects=(
  "$build_root/obj/collectives.o"
  "$build_root/obj/group.o"
  "$build_root/obj/misc/compress.o"
  "$build_root/obj/misc/coccl_frame_exchange.o"
  "$build_root/obj/misc/coccl_group.o"
  "$build_root/obj/misc/coccl_pipeline_plan.o"
  "$build_root/obj/misc/coccl_pipeline_stage.o"
  "$build_root/obj/misc/coccl_pipeline_execute.o"
  "$build_root/obj/misc/coccl_runtime.o"
  "$build_root/obj/misc/coccl_allgather.o"
  "$build_root/obj/misc/coccl_alltoall.o"
  "$build_root/obj/misc/coccl_reducescatter.o"
  "$build_root/obj/misc/coccl_allreduce.o"
  "$build_root/obj/misc/coccl_sendrecv.o"
  "$build_root/obj/misc/nccl_comp_wrapper.o"
)
make -C "$source_root/src" "${host_objects[@]}" \
  BUILDDIR="$build_root" CUDA_HOME="$CUDA_HOME" \
  NVCC_GENCODE="$gencode" DEBUG=0 -j8

layout_object="$build_root/obj/device/pipeline/coccl_pipeline_layout.cu.o"
make -C "$source_root/src/device" "$layout_object" \
  BUILDDIR="$build_root" CUDA_HOME="$CUDA_HOME" \
  NVCC_GENCODE="$gencode" DEBUG=0 -j8

make -C "$source_root/src/device/compress/sdp4bit" \
  BUILDDIR="$build_root" SUBOBJDIR="$build_root/obj/device/compress" \
  NCCLDIR="$source_root" CUDA_HOME="$CUDA_HOME" \
  NVHPC_CUDA_HOME="$CUDA_HOME" NVCC_GENCODE="$gencode" -j8
dietgpu_plugin="$build_root/obj/device/compress/libcompress/libdietgpu.so"
if [[ ! -f "$dietgpu_plugin" ||
      "$source_root/src/device/compress/dietgpu/dietgpu_plugin.cu" \
        -nt "$dietgpu_plugin" ||
      "$source_root/src/include/compressor_plugin/coccl_compressor_plugin.h" \
        -nt "$dietgpu_plugin" ]]; then
  make -C "$source_root/src/device/compress/dietgpu" \
    BUILDDIR="$build_root" SUBOBJDIR="$build_root/obj/device/compress" \
    NCCLDIR="$source_root" CUDA_HOME="$CUDA_HOME" \
    NVHPC_CUDA_HOME="$CUDA_HOME" NVCC_GENCODE="$gencode" -j8
fi

read -r -a device_objects <"$manifest"
mapfile -t all_host_objects < <(
  while IFS= read -r object; do
    relative=${object#"$build_root/obj/"}
    [[ -f "$source_root/src/${relative%.o}.cc" ]] && printf '%s\n' "$object"
  done < <(find "$build_root/obj" -type f -name '*.o' \
           ! -path '*/device/*' | sort)
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

make -C "$source_root/tests/coccl-tests/src" m15-host-tests \
  BUILDDIR="$host_build" NCCL_HOME="$build_root" CUDA_HOME="$CUDA_HOME" \
  NVCC_GENCODE="$gencode" DEBUG=0 -j8
make -C "$source_root/tests/coccl-tests/src" \
  "$gpu_build/coccl_m5_pipeline_layout_test" \
  BUILDDIR="$gpu_build" NCCL_HOME="$build_root" CUDA_HOME="$CUDA_HOME" \
  NVCC_GENCODE="$gencode" DEBUG=0 -j8

baseline_source="$temp_root/m14_coccl_pipeline_layout.cu"
baseline_binary="$temp_root/coccl_m14_pipeline_layout_benchmark"
git -C "$source_root" show \
  "$baseline_commit:src/device/pipeline/coccl_pipeline_layout.cu" \
  >"$baseline_source"
"$CUDA_HOME/bin/nvcc" -ccbin g++ $gencode -std=c++17 -O3 -g \
  -DCOCCL_M15_LEGACY_LAYOUT_API \
  -I"$source_root/src/include" -I"$build_root/include" \
  -I"$CUDA_HOME/include" \
  "$source_root/tests/coccl-tests/src/coccl_m5_pipeline_layout_test.cu" \
  "$baseline_source" -L"$CUDA_HOME/lib64" -lcudart -lrt \
  -o "$baseline_binary"

{
  printf 'M15 Pack/Unpack hierarchical swizzle build manifest\n'
  printf 'baseline_commit=%s\n' "$baseline_commit"
  printf 'NVCC_GENCODE=%s\n' "$gencode"
  printf 'DEBUG=0\n'
  printf 'host_objects=%s\n' "${host_objects[*]}"
  printf 'device_object=%s\n' "$layout_object"
  printf 'host_tests=M15 path/recipe/plugin PASS\n'
} >"$result_root/build-manifest.txt"
