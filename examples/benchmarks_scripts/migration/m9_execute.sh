#!/usr/bin/env bash
set -euo pipefail

source_root=${M9_SOURCE_ROOT:-/data/home/scyb672/run/lxc/COCCL-migrate}
result_root=${M9_RESULT_ROOT:-/data/home/scyb672/run/lxc/COCCL-migrate/results/M9}
build_root=${M9_BUILD_ROOT:-$source_root/build}
host_test_dir=${M9_HOST_TEST_DIR:-/data/home/scyb672/run/codex-tmp/coccl-m9-host}
tests=${M9_TESTS_DIR:-$source_root/tests/coccl-tests/build}
script_root="$source_root/examples/benchmarks_scripts/migration"
status_file="$result_root/execution.status"

export CUDA_HOME=${M9_CUDA_HOME:-/data/apps/cuda/12.4}
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$build_root/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

mkdir -p "$result_root"
"$host_test_dir/coccl_m9_fixed_compressor_test" \
  "$build_root/obj/device/compress/libcompress" \
  >"$result_root/host-fixed-compressors.txt"
"$tests/coccl_m5_pipeline_layout_test" \
  >"$result_root/layout-correctness.txt"

for compressor in native sdp4bit zfp; do
  printf '%s smoke-%s-start\n' "$(date -Is)" "$compressor" >>"$status_file"
  bash "$script_root/m9_matrix.sh" "$source_root" "$compressor" smoke
  printf '%s smoke-%s-complete\n' "$(date -Is)" "$compressor" >>"$status_file"
done

for compressor in native sdp4bit zfp; do
  printf '%s performance-%s-start\n' "$(date -Is)" "$compressor" >>"$status_file"
  bash "$script_root/m9_matrix.sh" "$source_root" "$compressor" performance
  printf '%s performance-%s-complete\n' "$(date -Is)" "$compressor" >>"$status_file"
done

for compressor in sdp4bit zfp; do
  printf '%s memory-%s-start\n' "$(date -Is)" "$compressor" >>"$status_file"
  bash "$script_root/m9_matrix.sh" "$source_root" "$compressor" memory
  printf '%s memory-%s-complete\n' "$(date -Is)" "$compressor" >>"$status_file"
done

python3 "$script_root/m9_report.py" "$result_root" \
  /data/home/scyb672/run/lxc/COCCL-migrate/results
printf '%s complete\n' "$(date -Is)" >>"$status_file"
