SHELL := /usr/bin/env bash
MAKEFLAGS += -r
.SUFFIXES:
.SECONDARY:

NCCLDIR := ../../../..
BUILDDIR ?= $(abspath ../../../../build)
SUBOBJDIR ?= $(BUILDDIR)/obj/device/compress
CUDA_HOME ?= /usr/local/cuda
NVCC_GENCODE ?= -gencode=arch=compute_80,code=sm_80

CMAKE ?= cmake
ZFP_BUILDDIR := $(SUBOBJDIR)/zfp/build
ZFP_LIBDIR := $(SUBOBJDIR)/libcompress
ZFP_SO := $(ZFP_LIBDIR)/libzfp.so
ZFP_CMAKE_BUILD_TYPE ?= Release

all: zfp.build

zfp.configure:
	@mkdir -p $(ZFP_BUILDDIR)
	$(CMAKE) -S . -B $(ZFP_BUILDDIR) \
	  -DCMAKE_BUILD_TYPE=$(ZFP_CMAKE_BUILD_TYPE) \
	  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$(ZFP_LIBDIR) \
	  -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=$(ZFP_BUILDDIR)/lib \
	  -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=$(ZFP_BUILDDIR)/bin \
	  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
	  -DBUILD_SHARED_LIBS=OFF \
	  -DBUILD_COCCL=ON \
	  -DZFP_WITH_CUDA=ON \
	  -DCUDA_TOOLKIT_ROOT_DIR=$(CUDA_HOME) \
	  -DCUDA_BIN_DIR=$(CUDA_HOME)/bin \
	  -DCOCCL_INCLUDE_DIRS=$(abspath ../../../include) \
	  -DCOCCL_NCCL_INCLUDE_DIR=$(abspath $(BUILDDIR)/include) \
	  -DCUDA_NVCC_FLAGS="$(NVCC_GENCODE)"

zfp.build: zfp.configure
	+$(CMAKE) --build $(ZFP_BUILDDIR) --target coccl_zfp_plugin --parallel 8
	@test -f $(ZFP_SO)

.PHONY: all zfp.configure zfp.build clean
clean:
	rm -rf $(ZFP_BUILDDIR)
	rm -f $(ZFP_SO)
