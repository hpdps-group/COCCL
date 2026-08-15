#!/usr/bin/env bash
set -euo pipefail

output_root=${1:-/data/home/scyb672/run/lxc/COCCL-migrate/results/M0}
original=/data/home/scyb672/run/lxc/COCCL
migrate=/data/home/scyb672/run/lxc/COCCL-migrate
refine=/data/run01/scyb672/lxc/coccl-refine

mkdir -p "$output_root/revisions"

describe_git_tree() {
  local label=$1
  local root=$2

  printf '[%s]\n' "$label"
  printf 'path=%s\n' "$root"
  if git -C "$root" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    printf 'commit=%s\n' "$(git -C "$root" rev-parse HEAD)"
    printf 'branch=%s\n' "$(git -C "$root" branch --show-current)"
    git -C "$root" remote -v
    git -C "$root" status --short --branch
  else
    printf 'git_status=not-a-git-worktree\n'
  fi
  printf '\n'
}

{
  printf 'captured_at=%s\n\n' "$(date -Is)"
  describe_git_tree original "$original"
  describe_git_tree migrate-copy "$migrate"
  describe_git_tree refine-reference "$refine"
} >"$output_root/revisions.txt"

git -C "$original" status --porcelain=v2 --branch \
  >"$output_root/revisions/original-status.txt"
git -C "$original" diff --summary \
  >"$output_root/revisions/original-diff-summary.txt"
git -C "$original" diff -- \
  examples/benchmarks_scripts/communication_benchmarks_single_node.sh \
  examples/build_scripts/build.sh \
  examples/build_scripts/env.sh \
  tests/coccl-tests/src/Makefile \
  tests/coccl-tests/src/all_reduce_comp_oneShot.cu \
  >"$output_root/revisions/original-harness.patch"

git -C "$migrate" status --porcelain=v2 --branch \
  >"$output_root/revisions/migrate-status.txt"
