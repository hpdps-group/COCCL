#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
current_root=${M21_CURRENT_ROOT:-/data/home/scyb672/run/lxc/COCCL-migrate}

export M20_CURRENT_ROOT="$current_root"
export M20_TEMP_ROOT=${M21_TEMP_ROOT:-/data/home/scyb672/run/codex-tmp/coccl-m21-pack-optimization}
export M20_RESULT_ROOT=${M21_RESULT_ROOT:-$current_root/results/M21}
export M20_RUNTIME_ROOT=${M21_RUNTIME_ROOT:-/data/home/scyb672/run/codex-tmp/coccl-m19-scope-policy/runtime-build}
export M20_PLUGIN_ROOT=${M21_PLUGIN_ROOT:-/data/home/scyb672/run/codex-tmp/coccl-m19-scope-policy/plugins/libcompress}
export M20_TEST_ROOT=${M21_TEST_ROOT:-$M20_TEMP_ROOT/gpu-tests}
export M20_HOST_ROOT=${M21_HOST_ROOT:-$M20_TEMP_ROOT/host-tests}
export M20_HOSTFILE=${M21_HOSTFILE:-${M20_HOSTFILE:-}}
export M20_CORRECTNESS_CASES=${M21_CORRECTNESS_CASES:-"1:4096 2:4096 4:4096 8:4096 2:4097 4:4097 4:4099 8:4097 8:4103"}

case ${2:-} in
  performance-single|performance-two-node)
    profile="$M20_RESULT_ROOT/sdp4bit-performance-profile.txt"
    mkdir -p "$M20_RESULT_ROOT"
    LD_LIBRARY_PATH="$M20_RUNTIME_ROOT/lib:$M20_PLUGIN_ROOT:${M21_CUDA_HOME:-/data/apps/cuda/12.8}/lib64:${LD_LIBRARY_PATH:-}" \
      "$M20_HOST_ROOT/coccl_m1_plugin_load_test" "$M20_PLUGIN_ROOT" \
      >"$profile"
    {
      printf 'profile=core-pack-ordinary-codec\n'
      printf 'fused_hierarchical_swizzle=0\n'
      printf 'new_path=core-swizzle-ordinary-codec\n'
      printf 'original_sdp4bit_path=swizzled-quantize\n'
      printf 'acceptance_reference=m20-nonfused,current-depth1,original-swizzled\n'
    } >>"$profile"
    ;;
esac

exec bash "$script_dir/m20_matrix.sh" "$@"
