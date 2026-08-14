# COCCL build definitions.
#
# This file is intentionally usable in two ways:
# - included by NCCL makefiles with COCCL_MAKEFILE_INCLUDE_ONLY=1, where it
#   only exports COCCL paths/source lists;
# - invoked directly for COCCL-owned tools and compressor plugin targets.

COCCL_MAKEFILE := $(lastword $(MAKEFILE_LIST))
COCCL_ROOT ?= $(patsubst %/,%,$(dir $(COCCL_MAKEFILE)))
NCCLDIR ?= $(abspath $(COCCL_ROOT)/../..)

COCCL_INCLUDE_DIR = $(COCCL_ROOT)/include
COCCL_INCFLAGS = -I$(COCCL_INCLUDE_DIR)

COCCL_HOST_SRCFILES = \
	$(wildcard $(COCCL_ROOT)/*.cc) \
	$(wildcard $(COCCL_ROOT)/buffer/*.cc) \
	$(wildcard $(COCCL_ROOT)/debug/*.cc) \
	$(wildcard $(COCCL_ROOT)/param/*.cc) \
	$(wildcard $(COCCL_ROOT)/pipeline/*.cc) \
	$(wildcard $(COCCL_ROOT)/primitives/*.cc) \
	$(wildcard $(COCCL_ROOT)/tuning/*.cc) \
	$(wildcard $(COCCL_ROOT)/training_assist/*.cc)

COCCL_OBJDIR ?= $(BUILDDIR)/obj/coccl-extend
COCCL_DEVICE_DIR = $(COCCL_ROOT)/device
COCCL_DEVICE_OBJDIR = $(COCCL_OBJDIR)/device
COCCL_CUDA_SRCFILES = \
	$(wildcard $(COCCL_DEVICE_DIR)/pipeline/*.cu) \
	$(wildcard $(COCCL_DEVICE_DIR)/reduce_helper/*.cu)
COCCL_CUDA_OBJFILES = $(patsubst $(COCCL_ROOT)/%.cu,$(COCCL_OBJDIR)/%.cu.o,$(COCCL_CUDA_SRCFILES))
# Keep generated dependencies separate from the old device sub-build's *.cu.d
# files so an incremental build can migrate without reading stale source paths.
COCCL_CUDA_DEPFILES = $(COCCL_CUDA_OBJFILES:%=%.d)
COCCL_PLUGIN_DIR = $(COCCL_ROOT)/compressor_plugin
COCCL_PLUGIN_OBJDIR = $(COCCL_OBJDIR)/compressor_plugin
COCCL_PLUGIN_LIB_TARGET := coccl-plugin-libs
COCCL_CONFIG_CHECK_TARGET := coccl-config-check
COCCL_CONFIG_CHECKER := $(BUILDDIR)/bin/coccl-config-check
COCCL_NCCL_HEADER := $(BUILDDIR)/include/nccl.h
COCCL_CONFIG_CHECKER_SRCFILES := \
	$(COCCL_ROOT)/tools/coccl_config_check/coccl_config_check.cc \
	$(COCCL_ROOT)/param/coccl_config.cc \
	$(COCCL_ROOT)/debug/coccl_config_format.cc
COCCL_CONFIG_CHECKER_HEADERS := \
	$(COCCL_ROOT)/include/compressor_plugin/detail/coccl_compressor_abi.h \
	$(COCCL_ROOT)/include/config/coccl_config.h \
	$(COCCL_ROOT)/include/config/coccl_config_debug.h \
	$(COCCL_ROOT)/include/runtime/coccl_operation.h
COCCL_BUILD_TARGET := coccl-libs

ifeq ($(COCCL_MAKEFILE_INCLUDE_ONLY),1)
$(COCCL_OBJDIR)/%.cu.o: $(COCCL_ROOT)/%.cu
	@printf "Compiling  %-35s > %s\n" $< $@
	mkdir -p $(dir $@)
	$(NVCC) $(NVCUFLAGS) -Xcompiler -fPIC \
		-I$(INCDIR) -I$(NCCLDIR)/src -I$(NCCLDIR)/src/include \
		$(COCCL_INCFLAGS) -MMD -MF $@.d -c $< -o $@
endif

ifneq ($(COCCL_MAKEFILE_INCLUDE_ONLY),1)
CUDA_HOME ?= /usr/local/cuda

.PHONY: all $(COCCL_BUILD_TARGET) $(COCCL_PLUGIN_LIB_TARGET) \
	$(COCCL_CONFIG_CHECK_TARGET) clean

all: $(COCCL_BUILD_TARGET)

$(COCCL_BUILD_TARGET): $(COCCL_PLUGIN_LIB_TARGET)

$(COCCL_CONFIG_CHECK_TARGET): $(COCCL_CONFIG_CHECKER)

$(COCCL_CONFIG_CHECKER): $(COCCL_CONFIG_CHECKER_SRCFILES) \
	$(COCCL_CONFIG_CHECKER_HEADERS) $(COCCL_NCCL_HEADER)
	@printf "Linking    %-35s > %s\n" coccl-config-check $@
	mkdir -p $(dir $@)
	$(CXX) -std=c++17 -O2 -Wall -Wextra -DCOCCL_CONFIG_PARSER_ONLY \
		-I$(CUDA_HOME)/include -I$(BUILDDIR)/include \
		-I$(NCCLDIR)/src/include $(COCCL_INCFLAGS) \
		-o $@ $(COCCL_CONFIG_CHECKER_SRCFILES) -ldl \
		-Wl,--no-as-needed -L$(CUDA_HOME)/lib64 -lcudart -Wl,--as-needed \
		-Wl,-rpath,$(CUDA_HOME)/lib64

$(COCCL_NCCL_HEADER):
	$(MAKE) -C $(NCCLDIR)/src $@ BUILDDIR=$(abspath $(BUILDDIR))

$(COCCL_PLUGIN_LIB_TARGET):
	$(MAKE) -C $(COCCL_PLUGIN_DIR) all \
		SUBOBJDIR=$(abspath $(COCCL_PLUGIN_OBJDIR)) \
		BUILDDIR=$(abspath $(BUILDDIR)) \
		NCCLDIR=$(abspath $(NCCLDIR)) \
		COCCL_ROOT=$(abspath $(COCCL_ROOT)) \
		NVCC_GENCODE='$(NVCC_GENCODE)'

clean:
	rm -rf $(COCCL_DEVICE_OBJDIR)
	rm -f $(COCCL_CONFIG_CHECKER)
	$(MAKE) -C $(COCCL_PLUGIN_DIR) clean \
		SUBOBJDIR=$(abspath $(COCCL_PLUGIN_OBJDIR)) \
		BUILDDIR=$(abspath $(BUILDDIR)) \
		NCCLDIR=$(abspath $(NCCLDIR)) \
		COCCL_ROOT=$(abspath $(COCCL_ROOT))
endif
