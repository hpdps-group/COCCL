SHELL := /usr/bin/env bash
MAKEFLAGS += -r
.SUFFIXES:

NCCLDIR ?= ../../../..
COCCL_ROOT ?= ../..
BUILDDIR ?= $(abspath $(NCCLDIR)/build)
SUBOBJDIR ?= $(BUILDDIR)/obj/coccl-extend/compressor_plugin

CMAKE ?= cmake
COCCL_CMAKE_JOBS ?= 8
CUDA_HOME ?= $(if $(NVHPC_CUDA_HOME),$(NVHPC_CUDA_HOME),/usr/local/cuda)
NVCC_GENCODE ?= -gencode=arch=compute_80,code=sm_80

OBJDIR := $(SUBOBJDIR)/zfp/cmake
LIBDIR := $(SUBOBJDIR)/libcompress
PLUGIN_SO := $(LIBDIR)/libzfp.so

all: zfp.build

.PHONY: zfp.configure zfp.build
zfp.configure:
	@mkdir -p $(OBJDIR) $(LIBDIR)
	$(CMAKE) -S . -B $(OBJDIR) \
	  -DCMAKE_BUILD_TYPE=Release \
	  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$(abspath $(LIBDIR)) \
	  -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=$(abspath $(OBJDIR)/lib) \
	  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
	  -DBUILD_SHARED_LIBS=OFF \
	  -DBUILD_COCCL=ON \
	  -DZFP_WITH_CUDA=ON \
	  -DCUDA_TOOLKIT_ROOT_DIR=$(CUDA_HOME) \
	  -DCUDA_BIN_DIR=$(CUDA_HOME)/bin \
	  -DCOCCL_INCLUDE_DIRS=$(abspath $(COCCL_ROOT)/include) \
	  -DCOCCL_NCCL_INCLUDE_DIR=$(abspath $(BUILDDIR)/include) \
	  -DCUDA_NVCC_FLAGS='$(NVCC_GENCODE)'

zfp.build: zfp.configure
	+$(CMAKE) --build $(OBJDIR) --target coccl_zfp_plugin \
	  --parallel $(COCCL_CMAKE_JOBS)
	@test -f $(PLUGIN_SO)

.PHONY: clean
clean:
	rm -rf $(SUBOBJDIR)/zfp
	rm -f $(PLUGIN_SO)
