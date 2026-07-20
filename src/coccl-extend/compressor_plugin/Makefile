SHELL := /usr/bin/env bash
MAKEFLAGS += -r
.SUFFIXES:
.SECONDARY:

NCCLDIR ?= $(abspath ../../..)
COCCL_ROOT ?= $(abspath ..)
include $(NCCLDIR)/makefiles/common.mk
include $(NCCLDIR)/makefiles/version.mk

BUILDDIR ?= $(abspath $(NCCLDIR)/build)
SUBOBJDIR ?= $(BUILDDIR)/obj/coccl-extend/compressor_plugin
LIBCOMPRESSDIR := $(SUBOBJDIR)/libcompress

PLUGIN_DIRS := $(patsubst %/Makefile,%,$(wildcard */Makefile))

all: plugins

plugins:
	@for plugin in $(PLUGIN_DIRS); do \
		printf "Building COCCL compressor plugin %s\n" "$$plugin"; \
		$(MAKE) -C "$$plugin" \
			SUBOBJDIR=$(abspath $(SUBOBJDIR)) \
			BUILDDIR=$(abspath $(BUILDDIR)) \
			NCCLDIR=$(abspath $(NCCLDIR)) \
			COCCL_ROOT=$(abspath $(COCCL_ROOT)) \
			NVCC_GENCODE='$(NVCC_GENCODE)'; \
	done

.PHONY: all plugins clean
clean:
	@for plugin in $(PLUGIN_DIRS); do \
		rm -rf "$(SUBOBJDIR)/$$plugin"; \
		rm -f "$(LIBCOMPRESSDIR)/lib$$plugin.so"; \
	done
