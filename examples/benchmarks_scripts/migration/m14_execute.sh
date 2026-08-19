#!/usr/bin/env bash
set -euo pipefail

source_root=${M14_SOURCE_ROOT:-/data/home/scyb672/run/codex-tmp/coccl-valid-publish-20260816}
result_root=${M14_RESULT_ROOT:-/data/home/scyb672/run/lxc/COCCL-migrate/results/M14}
build_root=${M14_BUILD_ROOT:-/data/home/scyb672/run/lxc/COCCL-migrate/build}
host_tests=${M14_HOST_TEST_DIR:-/data/home/scyb672/run/codex-tmp/coccl-m13/host}
gpu_tests=${M14_TESTS_DIR:-/data/home/scyb672/run/lxc/COCCL-migrate/tests/coccl-tests/build}
plugin_root=${M14_PLUGIN_ROOT:-$build_root/obj/device/compress/libcompress}
script_root="$source_root/examples/benchmarks_scripts/migration"
status_file="$result_root/execution.status"

export CUDA_HOME=${M14_CUDA_HOME:-/data/apps/cuda/12.4}
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$build_root/lib:$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "$result_root"
"$host_tests/coccl_m14_framed_plan_test" >"$result_root/host-framed-plan.txt"
"$host_tests/coccl_m13_frame_exchange_test" >"$result_root/host-frame-exchange.txt"
"$host_tests/coccl_m1_plugin_load_test" "$plugin_root" dietgpu \
  >"$result_root/host-plugin-load.txt"
"$gpu_tests/coccl_m14_dietgpu_codec_test" "$plugin_root/libdietgpu.so" \
  >"$result_root/gpu-codec.txt"

for mode in completion sweep endpoint memory fixed; do
  printf '%s %s-start\n' "$(date -Is)" "$mode" >>"$status_file"
  bash "$script_root/m14_matrix.sh" "$source_root" "$mode"
  printf '%s %s-complete\n' "$(date -Is)" "$mode" >>"$status_file"
done

python3 "$script_root/m14_report.py" "$result_root"
printf '%s complete\n' "$(date -Is)" >>"$status_file"
