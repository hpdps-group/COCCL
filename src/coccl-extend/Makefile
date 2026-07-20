# COCCL build definitions.
#
# This file is intentionally usable in two ways:
# - included by NCCL makefiles with COCCL_MAKEFILE_INCLUDE_ONLY=1, where it
#   only exports COCCL paths/source lists;
# - invoked directly for COCCL-owned compressor plugin targets.

COCCL_MAKEFILE := $(lastword $(MAKEFILE_LIST))
COCCL_ROOT ?= $(patsubst %/,%,$(dir $(COCCL_MAKEFILE)))
NCCLDIR ?= $(abspath $(COCCL_ROOT)/../..)

COCCL_INCLUDE_DIR = $(COCCL_ROOT)/include
COCCL_INCFLAGS = -I$(COCCL_INCLUDE_DIR)

COCCL_HOST_SRCFILES = \
	$(wildcard $(COCCL_ROOT)/*.cc) \
	$(wildcard $(COCCL_ROOT)/buffer/*.cc) \
	$(wildcard $(COCCL_ROOT)/old_impl/*.cc) \
	$(wildcard $(COCCL_ROOT)/primitives/*.cc) \
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
COCCL_LEGACY_REDUCE_HELPER = $(COCCL_PLUGIN_OBJDIR)/libcompress/libcoccl_reduce_helper.so
COCCL_PLUGIN_LIB_TARGET := coccl-plugin-libs
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
.PHONY: all $(COCCL_BUILD_TARGET) $(COCCL_PLUGIN_LIB_TARGET) clean

all: $(COCCL_BUILD_TARGET)

$(COCCL_BUILD_TARGET): $(COCCL_PLUGIN_LIB_TARGET)
	# This DSO belonged to the retired lazy-loaded reduce helper build.
	rm -f $(COCCL_LEGACY_REDUCE_HELPER)

$(COCCL_PLUGIN_LIB_TARGET):
	$(MAKE) -C $(COCCL_PLUGIN_DIR) all \
		SUBOBJDIR=$(abspath $(COCCL_PLUGIN_OBJDIR)) \
		BUILDDIR=$(abspath $(BUILDDIR)) \
		NCCLDIR=$(abspath $(NCCLDIR)) \
		COCCL_ROOT=$(abspath $(COCCL_ROOT)) \
		NVCC_GENCODE='$(NVCC_GENCODE)'

clean:
	rm -rf $(COCCL_DEVICE_OBJDIR)
	$(MAKE) -C $(COCCL_PLUGIN_DIR) clean \
		SUBOBJDIR=$(abspath $(COCCL_PLUGIN_OBJDIR)) \
		BUILDDIR=$(abspath $(BUILDDIR)) \
		NCCLDIR=$(abspath $(NCCLDIR)) \
		COCCL_ROOT=$(abspath $(COCCL_ROOT))
endif
