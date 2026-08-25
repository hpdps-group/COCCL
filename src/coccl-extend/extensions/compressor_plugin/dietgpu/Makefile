SHELL := /usr/bin/env bash
MAKEFLAGS += -r
.SUFFIXES:

NCCLDIR ?= ../../../../..
COCCL_ROOT ?= ../../..
BUILDDIR ?= $(abspath $(NCCLDIR)/build)
SUBOBJDIR ?= $(BUILDDIR)/obj/coccl-extend/compressor_plugin
CMAKE ?= cmake
COCCL_CMAKE_JOBS ?= 8
CUDA_HOME ?= $(if $(NVHPC_CUDA_HOME),$(NVHPC_CUDA_HOME),/usr/local/cuda)

COCCL_CUDA_ARCHITECTURES ?= $(shell printf '%s\n' '$(NVCC_GENCODE)' | \
	grep -oE 'code=sm_[0-9]+' | sed 's/code=sm_//' | sort -nu | paste -sd ';' -)
COCCL_CUDA_ARCHITECTURES := $(if $(strip $(COCCL_CUDA_ARCHITECTURES)),$(strip $(COCCL_CUDA_ARCHITECTURES)),80)

OBJDIR := $(SUBOBJDIR)/dietgpu/cmake
LIBDIR := $(SUBOBJDIR)/libcompress
PLUGIN_SO := $(LIBDIR)/libdietgpu.so

all: dietgpu.build

.PHONY: dietgpu.configure dietgpu.build clean
dietgpu.configure:
	@mkdir -p $(OBJDIR) $(LIBDIR)
	$(CMAKE) -S . -B $(OBJDIR) \
	  -DCMAKE_BUILD_TYPE=Release \
	  -DCMAKE_CUDA_COMPILER=$(CUDA_HOME)/bin/nvcc \
	  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$(abspath $(LIBDIR)) \
	  -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=$(abspath $(OBJDIR)/lib) \
	  -DCOCCL_INCLUDE_DIR=$(abspath $(COCCL_ROOT)/include) \
	  -DCOCCL_NCCL_INCLUDE_DIR=$(abspath $(BUILDDIR)/include) \
	  -DCOCCL_CUDA_ARCHITECTURES="$(COCCL_CUDA_ARCHITECTURES)"

dietgpu.build: dietgpu.configure
	+$(CMAKE) --build $(OBJDIR) --target dietgpu \
	  --parallel $(COCCL_CMAKE_JOBS)
	@test -f $(PLUGIN_SO)

clean:
	rm -rf $(SUBOBJDIR)/dietgpu
	rm -f $(PLUGIN_SO)
