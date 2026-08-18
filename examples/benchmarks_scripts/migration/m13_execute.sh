#!/usr/bin/env bash
set -euo pipefail

source_root=${M13_SOURCE_ROOT:-/data/home/scyb672/run/codex-tmp/coccl-valid-publish-20260816}
result_root=${M13_RESULT_ROOT:-/data/home/scyb672/run/lxc/COCCL-migrate/results/M13}
build_root=${M13_BUILD_ROOT:-/data/home/scyb672/run/lxc/COCCL-migrate/build}
nccl_lib_root=${M13_NCCL_LIB_ROOT:-$build_root/lib}
host_test_dir=${M13_HOST_TEST_DIR:-/data/home/scyb672/run/codex-tmp/coccl-m13-host}
script_root="$source_root/examples/benchmarks_scripts/migration"
status_file="$result_root/execution.status"

export CUDA_HOME=${M13_CUDA_HOME:-/data/apps/cuda/12.4}
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$nccl_lib_root:$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "$result_root"
"$host_test_dir/coccl_m13_frame_exchange_test" \
  >"$result_root/host-frame-exchange.txt"
"$host_test_dir/coccl_m13_group_batch_test" \
  >"$result_root/host-group-batch.txt"

for profile in native fallback auto-sdp4bit sdp4bit zfp zfp-raw; do
  printf '%s smoke-%s-start\n' "$(date -Is)" "$profile" >>"$status_file"
  bash "$script_root/m13_matrix.sh" "$source_root" "$profile" smoke
  printf '%s smoke-%s-complete\n' "$(date -Is)" "$profile" >>"$status_file"
done
for profile in auto-sdp4bit sdp4bit zfp zfp-raw; do
  printf '%s deadlock-%s-start\n' "$(date -Is)" "$profile" >>"$status_file"
  bash "$script_root/m13_matrix.sh" "$source_root" "$profile" deadlock
  printf '%s deadlock-%s-complete\n' "$(date -Is)" "$profile" >>"$status_file"
done
for profile in native fallback sdp4bit zfp zfp-raw; do
  printf '%s performance-%s-start\n' "$(date -Is)" "$profile" >>"$status_file"
  bash "$script_root/m13_matrix.sh" "$source_root" "$profile" performance
  printf '%s performance-%s-complete\n' "$(date -Is)" "$profile" >>"$status_file"
done
for profile in native fallback sdp4bit zfp zfp-raw; do
  printf '%s memory-%s-start\n' "$(date -Is)" "$profile" >>"$status_file"
  bash "$script_root/m13_matrix.sh" "$source_root" "$profile" memory
  printf '%s memory-%s-complete\n' "$(date -Is)" "$profile" >>"$status_file"
done

python3 "$script_root/m13_report.py" "$result_root"
printf '%s complete\n' "$(date -Is)" >>"$status_file"
