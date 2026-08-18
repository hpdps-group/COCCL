#!/usr/bin/env bash
set -euo pipefail

source_root=${1:?usage: m12_build.sh SOURCE_ROOT}
cuda_root=${M12_CUDA_HOME:-/data/apps/cuda/12.4}
gencode=${NVCC_GENCODE:-'-gencode=arch=compute_80,code=sm_80'}
build_root=${M12_BUILD_ROOT:-/data/home/scyb672/run/lxc/COCCL-migrate/build}
test_build=${M12_TESTS_DIR:-/data/home/scyb672/run/lxc/COCCL-migrate/tests/coccl-tests/build}
host_test_build=${M12_HOST_TEST_DIR:-/data/home/scyb672/run/codex-tmp/coccl-m12-host}
result_root=${M12_RESULT_ROOT:-/data/home/scyb672/run/lxc/COCCL-migrate/results/M12}

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
  "$build_root/obj/misc/coccl_pipeline_plan.o"
  "$build_root/obj/misc/coccl_pipeline_execute.o"
  "$build_root/obj/misc/coccl_allgather.o"
  "$build_root/obj/misc/coccl_alltoall.o"
  "$build_root/obj/misc/coccl_reducescatter.o"
  "$build_root/obj/misc/coccl_allreduce.o"
)
make -C "$source_root/src" "${objects[@]}" \
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

export NCCL_HOME="$build_root"
test_targets=(
  all_gather_comp_perf
  reduce_scatter_comp_oneshot_perf
  all_reduce_comp_twoshot_perf
)
target_paths=()
for target in "${test_targets[@]}"; do
  target_paths+=("$test_build/$target")
done
make -C "$source_root/tests/coccl-tests/src" \
  "${target_paths[@]}" \
  BUILDDIR="$test_build" NCCL_HOME="$build_root" CUDA_HOME="$CUDA_HOME" \
  NVCC_GENCODE="$gencode" -j8
make -C "$source_root/tests/coccl-tests/src" m5-layout-test \
  BUILDDIR="$test_build" NCCL_HOME="$build_root" CUDA_HOME="$CUDA_HOME" \
  NVCC_GENCODE="$gencode"

host_targets=(
  m1-host-tests m1-plugin-load-test m2-host-tests m3-host-tests
  m4-host-tests m5-host-tests m6-host-tests m7-host-tests
  m8-host-tests m9-host-tests m10-host-tests m11-host-tests
  m12-host-tests
)
make -C "$source_root/tests/coccl-tests/src" \
  "${host_targets[@]}" \
  BUILDDIR="$host_test_build" NCCL_HOME="$build_root" \
  CUDA_HOME="$CUDA_HOME" NVCC_GENCODE="$gencode"

{
  printf 'M12 standard in-place pipeline overlap build manifest\n'
  printf 'NVCC_GENCODE=%s\n' "$gencode"
  printf 'host_objects=%s\n' "${objects[*]}"
  printf 'gpu_tests=%s\n' "${test_targets[*]}"
  printf 'host_tests=M1-M12 PASS\n'
} >"$result_root/build-manifest.txt"
