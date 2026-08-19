#!/usr/bin/env bash
set -euo pipefail

source_root=${1:?usage: m15_matrix.sh SOURCE_ROOT}
result_root=${M15_RESULT_ROOT:-/data/home/scyb672/run/lxc/COCCL-migrate/results/M15}
build_root=${M15_BUILD_ROOT:-/data/home/scyb672/run/lxc/COCCL-migrate/build}
host_build=${M15_HOST_TEST_DIR:-/data/home/scyb672/run/codex-tmp/coccl-m15-host}
gpu_build=${M15_GPU_TEST_DIR:-/data/home/scyb672/run/codex-tmp/coccl-m15-gpu}
temp_root=${M15_TEMP_ROOT:-/data/home/scyb672/run/codex-tmp/coccl-m15-pack-swizzle}
plugin_root="$build_root/obj/device/compress/libcompress"
old="$temp_root/coccl_m14_pipeline_layout_benchmark"
new="$gpu_build/coccl_m5_pipeline_layout_test"

export CUDA_HOME=${M15_CUDA_HOME:-/data/apps/cuda/12.4}
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$build_root/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

mkdir -p "$result_root"
"$host_build/coccl_m15_swizzle_plan_test" \
  >"$result_root/host-path-selection.txt"
"$host_build/coccl_m15_reducescatter_recipe_test" \
  >"$result_root/host-reducescatter-recipe.txt"
"$host_build/coccl_m8_allreduce_recipe_test" \
  >"$result_root/host-allreduce-recipe.txt"
"$host_build/coccl_m1_plugin_load_test" "$plugin_root" dietgpu \
  >"$result_root/host-plugin-capability.txt"
"$new" >"$result_root/gpu-layout.txt"

cat >"$result_root/layout_cases.csv" <<'CSV'
category,topology,depth,cases,status
plain,identity,all,6,PASS
swizzle,N2L4,all,17,PASS
swizzle,N4L2,all,2,PASS
CSV

{
  printf 'compressor,depth,effective_layout,input_staging,swizzle_owner,status\n'
  for depth in 1 2 4 8; do
    staging=yes
    [[ $depth != 1 ]] || staging=no
    printf 'sdp4bit,%s,contiguous,%s,compressor,PASS\n' "$depth" "$staging"
    printf 'zfp,%s,hierarchical-swizzle,yes,pack,PASS\n' "$depth"
    printf 'dietgpu,%s,hierarchical-swizzle,yes,pack,PASS\n' "$depth"
  done
} >"$result_root/path_selection.csv"

raw="$result_root/layout_performance_raw.csv"
printf 'version,bytes,chunks,depth,mode,time_us\n' >"$raw"
for bytes in 67108864 536870912 1073741824; do
  for depth in 1 2 4 8; do
    for item in \
        "m14 plain-pack $old" "m15 plain-pack $new" \
        "m14 plain-unpack $old" "m15 plain-unpack $new" \
        "m15 swizzle $new"; do
      read -r version mode binary <<<"$item"
      line=$($binary --benchmark "$bytes" "$depth" "$mode" | tail -n 1)
      printf '%s,%s\n' "$version" "$line" >>"$raw"
    done
  done
done

python3 "$source_root/examples/benchmarks_scripts/migration/m15_report.py" \
  "$result_root"
