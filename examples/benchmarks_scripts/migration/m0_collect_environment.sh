#!/usr/bin/env bash
set -euo pipefail

source_root=${1:?usage: m0_collect_environment.sh SOURCE_ROOT OUTPUT}
output=${2:?usage: m0_collect_environment.sh SOURCE_ROOT OUTPUT}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

source "$script_dir/m0_env.sh" "$source_root"
mkdir -p "$(dirname "$output")"

{
  printf 'timestamp=%s\n' "$(date -Is)"
  printf 'hostname=%s\n' "$(hostname -f)"
  printf 'source_root=%s\n' "$source_root"
  printf 'cuda_home=%s\n' "$CUDA_HOME"
  printf 'cuda_visible_devices=%s\n' "$CUDA_VISIBLE_DEVICES"
  printf '\n[uname]\n'
  uname -a
  printf '\n[module]\n'
  type module >/dev/null 2>&1 && module list
  printf '\n[nvcc]\n'
  "$CUDA_HOME/bin/nvcc" --version
  printf '\n[compiler]\n'
  g++ --version
  printf '\n[gpus]\n'
  nvidia-smi --query-gpu=index,name,uuid,driver_version,pstate,persistence_mode,clocks.current.sm,clocks.current.memory,clocks.max.sm,clocks.max.memory,power.limit,memory.total \
    --format=csv
  printf '\n[topology]\n'
  nvidia-smi topo -m
  printf '\n[nccl version macros]\n'
  grep -E '^#define NCCL_(MAJOR|MINOR|PATCH|SUFFIX|VERSION_CODE)' \
    "$source_root/build/include/nccl.h"
  printf '\n[cuda objects]\n'
  cuobjdump --list-elf "$source_root/build/lib/libnccl.so"
} >"$output" 2>&1
