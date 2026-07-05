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
ZFP_LIBDIR := $(ZFP_BUILDDIR)/lib
ZFP_CMAKE_SO := $(ZFP_LIBDIR)/libcuzfp.so
CUZFP_SO := $(SUBOBJDIR)/libcompress/libcuzfp.so
ZFP_CMAKE_BUILD_TYPE ?= Release

all: $(CUZFP_SO)

zfp.configure:
	@mkdir -p $(ZFP_BUILDDIR)
	$(CMAKE) -S . -B $(ZFP_BUILDDIR) \
	  -DCMAKE_BUILD_TYPE=$(ZFP_CMAKE_BUILD_TYPE) \
	  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$(ZFP_LIBDIR) \
	  -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=$(ZFP_LIBDIR) \
	  -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=$(ZFP_BUILDDIR)/bin \
	  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
	  -DBUILD_SHARED_LIBS=ON \
	  -DBUILD_COCCL=ON \
	  -DZFP_WITH_CUDA=ON \
	  -DCUDA_TOOLKIT_ROOT_DIR=$(CUDA_HOME) \
	  -DCUDA_BIN_DIR=$(CUDA_HOME)/bin \
	  -DCOCCL_INCLUDE_DIRS=$(abspath ../../../include) \
	  -DNCCL_INCLUDE_DIRS=$(abspath $(BUILDDIR)/include) \
	  -DCUDA_NVCC_FLAGS="$(NVCC_GENCODE)"

zfp.build: zfp.configure
	+$(CMAKE) --build $(ZFP_BUILDDIR) --target cuzfp

$(CUZFP_SO): zfp.build
	@mkdir -p $(dir $@)
	@if ! cmp -s $(ZFP_CMAKE_SO) $@; then \
	  printf "Installing %-35s > %s\n" "$(ZFP_CMAKE_SO)" "$@"; \
	  install -m 755 $(ZFP_CMAKE_SO) $@; \
	fi

.PHONY: all zfp.configure zfp.build clean
clean:
	rm -rf $(ZFP_BUILDDIR)
	rm -f $(CUZFP_SO)
