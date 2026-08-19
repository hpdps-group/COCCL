#!/usr/bin/env bash
set -euo pipefail

source_root=${1:?usage: m18_build.sh SOURCE_ROOT}
cuda_root=${M18_CUDA_HOME:-/data/apps/cuda/12.4}
mpi_root=${M18_MPI_HOME:-/data/apps/openmpi/4.1.5_cuda12.8}
current_root=${M18_CURRENT_ROOT:-/data/home/scyb672/run/lxc/COCCL-migrate}
temp_root=${M18_TEMP_ROOT:-/data/home/scyb672/run/codex-tmp/coccl-m18-autotune}
result_root=${M18_RESULT_ROOT:-$current_root/results/M18}
gencode=${NVCC_GENCODE:-'-gencode=arch=compute_80,code=sm_80'}

export CUDA_HOME="$cuda_root"
export PATH="$CUDA_HOME/bin:$mpi_root/bin:$PATH"
mkdir -p "$temp_root/host" "$temp_root/perf" "$result_root/model_snapshots"

# Rebuild stale Host objects and relink against the existing device manifest.
# The normal lib target always traverses every compressor plugin.
make -C "$source_root/src" m18-host-relink \
  --eval='
.PHONY: m18-host-relink
.SECONDEXPANSION:
m18-host-relink: $$(INCTARGETS) $$(LIBOBJ)
	@test -s $(DEVMANIFEST)
	@printf "Linking    %-35s > %s\n" $(LIBTARGET) $(LIBDIR)/$(LIBTARGET)
	@mkdir -p $(LIBDIR)
	$(CXX) $(CXXFLAGS) -shared -Wl,--no-as-needed -Wl,-soname,$(LIBSONAME) -o $(LIBDIR)/$(LIBTARGET) $(LIBOBJ) $$(cat $(DEVMANIFEST)) $(LDFLAGS)
	ln -sf $(LIBSONAME) $(LIBDIR)/$(LIBNAME)
	ln -sf $(LIBTARGET) $(LIBDIR)/$(LIBSONAME)
	@printf "Archiving  %-35s > %s\n" $(STATICLIBTARGET) $(LIBDIR)/$(STATICLIBTARGET)
	ar cr $(LIBDIR)/$(STATICLIBTARGET) $(LIBOBJ) $$(cat $(DEVMANIFEST))
' \
  BUILDDIR="$current_root/build" CUDA_HOME="$CUDA_HOME" \
  NVCC_GENCODE="$gencode" DEBUG=0 -j8

make -C "$source_root/tests/coccl-tests/src" \
  "$temp_root/host/coccl_m18_autotune_test" \
  "$temp_root/host/coccl_m18_model_selector" \
  BUILDDIR="$temp_root/host" NCCL_HOME="$current_root/build" \
  CUDA_HOME="$CUDA_HOME" NVCC_GENCODE="$gencode" DEBUG=0 -j8

make -C "$source_root/tests/coccl-tests/src" \
  "$temp_root/perf/all_reduce_perf" \
  BUILDDIR="$temp_root/perf" NCCL_HOME="$current_root/build" \
  CUDA_HOME="$CUDA_HOME" MPI=1 MPI_HOME="$mpi_root" \
  NVCC_GENCODE="$gencode" DEBUG=0 -j8

"$temp_root/host/coccl_m18_autotune_test" \
  >"$result_root/selector_cases_cpu.csv"

{
  printf 'M18 autotune build manifest\n'
  printf 'source_commit=%s\n' "$(git -C "$source_root" rev-parse HEAD)"
  printf 'NVCC_GENCODE=%s\n' "$gencode"
  printf 'DEBUG=0\n'
  printf 'MPI_HOME=%s\n' "$mpi_root"
  printf 'autotune_test=%s\n' "$temp_root/host/coccl_m18_autotune_test"
  printf 'model_selector=%s\n' "$temp_root/host/coccl_m18_model_selector"
  printf 'profile_smoke=%s\n' "$temp_root/perf/all_reduce_perf"
} >"$result_root/build-manifest.txt"
