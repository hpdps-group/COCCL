SHELL := /usr/bin/env bash
MAKEFLAGS += -r
.SUFFIXES:

NCCLDIR ?= ../../../..
include $(NCCLDIR)/makefiles/common.mk

BUILDDIR ?= $(abspath $(NCCLDIR)/build)
SUBOBJDIR ?= $(BUILDDIR)/obj/device/compress
OBJDIR := $(SUBOBJDIR)/taco
PLUGIN_SO := $(SUBOBJDIR)/libcompress/libtaco.so

CUDA_HOME ?= $(if $(NVHPC_CUDA_HOME),$(NVHPC_CUDA_HOME),/usr/local/cuda)
NVCC_GENCODE ?= -gencode=arch=compute_80,code=sm_80
NVCCFLAGS := -std=c++17 -O3 -Xcompiler -fPIC \
	-I$(CUDA_HOME)/include \
	-I$(BUILDDIR)/include \
	-I$(NCCLDIR)/src/include \
	$(NVCC_GENCODE)
LDFLAGS := -L$(CUDA_HOME)/lib64 -lcudart

all: $(PLUGIN_SO)

$(OBJDIR)/taco.o: taco.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(PLUGIN_SO): $(OBJDIR)/taco.o
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_GENCODE) -shared $< $(LDFLAGS) -o $@

.PHONY: clean
clean:
	rm -rf $(OBJDIR)
	rm -f $(PLUGIN_SO)
