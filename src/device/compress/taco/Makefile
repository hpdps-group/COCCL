SHELL := /usr/bin/env bash
MAKEFLAGS += -r
.SUFFIXES:
.SECONDARY:

# Directories
NCCLDIR := ../../../..
include $(NCCLDIR)/makefiles/common.mk
include $(NCCLDIR)/makefiles/version.mk

# Ensure BUILDDIR is set
BUILDDIR ?= $(abspath $(NCCLDIR)/build)

# Object and library directories
COMPRESS_DIR := $(BUILDDIR)/obj/device/compress
OBJDIR := $(COMPRESS_DIR)/taco
LIBDIR := $(COMPRESS_DIR)/libcompress

# Create directories if not exist
$(shell mkdir -p $(OBJDIR))
$(shell mkdir -p $(LIBDIR))

# Target shared library
QUAN_SO := $(LIBDIR)/libtaco.so

# CUDA configuration
ifeq ($(NVHPC_CUDA_HOME),)
  NVHPC_CUDA_HOME := /usr/local/cuda
endif

CUDA_LIB := $(NVHPC_CUDA_HOME)/lib64
CUDA_INC := $(NVHPC_CUDA_HOME)/include

# Include paths
INCFLAGS := -I$(CUDA_INC) \
            -I. \
            -I../.. \
            -I../tahquant/includes \
            -Ihadamard \
            -I$(BUILDDIR)/include \
            -I$(NCCLDIR)/src/include \
            -I$(NCCLDIR)/src/device

# NVCC compile flags
NVCUFLAGS += $(INCFLAGS) \
             -Xcompiler -fPIC \
             $(NVCC_GENCODE)

# Linker flags
NVLDFLAGS := -L$(CUDA_LIB) -lcudart

# Source files
CUDA_SRCS = taco.cu
CUDA_OBJS = $(addprefix $(OBJDIR)/, $(CUDA_SRCS:.cu=.o))
DEPS = $(CUDA_OBJS:.o=.d)

# Debug info
$(info ===== TACO Module Configuration =====)
$(info BUILDDIR: $(BUILDDIR))
$(info OBJDIR:   $(OBJDIR))
$(info Target:   $(QUAN_SO))
$(info Sources:  $(CUDA_SRCS))
$(info Objects:  $(CUDA_OBJS))
$(info =====================================)

# Default target
all: $(QUAN_SO)

# Pretty print
SAY = @bash -c 'printf "%-15s %s\n" "$$1" "$$2"' SAY

# Compile CUDA files
$(OBJDIR)/%.o: %.cu taco.h
	@$(SAY) "Compiling" "$<"
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCUFLAGS) -dc $< -o $@

# Generate dependencies
$(OBJDIR)/%.d: %.cu
	@$(SAY) "Dependencies" "$<"
	@mkdir -p $(dir $@)
	@$(NVCC) $(NVCUFLAGS) -M $< | sed 's|^.*:|$(@:.d=.o) $@:|' > $@

# Link shared library
$(QUAN_SO): $(CUDA_OBJS)
	@$(SAY) "Linking" "$@"
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCUFLAGS) $(NVLDFLAGS) $(CUDA_OBJS) -shared -o $@
	@echo "Successfully built TACO compression library: $@"

# Include dependency files
-include $(DEPS)

# Clean target
.PHONY: clean
clean:
	@echo "Cleaning TACO module..."
	rm -rf $(OBJDIR)
	rm -f $(QUAN_SO)

# Info target
.PHONY: info
info:
	@echo "========================================="
	@echo "TACO Quantization Module Configuration"
	@echo "========================================="
	@echo "  Current Dir:    $(PWD)"
	@echo "  NCCLDIR:        $(NCCLDIR)"
	@echo "  BUILDDIR:       $(BUILDDIR)"
	@echo "  OBJDIR:         $(OBJDIR)"
	@echo "  LIBDIR:         $(LIBDIR)"
	@echo "  Target:         $(QUAN_SO)"
	@echo "  CUDA_HOME:      $(NVHPC_CUDA_HOME)"
	@echo "  NVCC:           $(NVCC)"
	@echo "  Sources:        $(CUDA_SRCS)"
	@echo "  Objects:        $(CUDA_OBJS)"
	@echo "========================================="

# Help target
.PHONY: help
help:
	@echo "TACO Quantization Module Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all       - Build TACO compression library (default)"
	@echo "  clean     - Remove object files and library"
	@echo "  info      - Display configuration information"
	@echo "  help      - Show this help message"
	@echo ""
	@echo "Usage:"
	@echo "  make              # Build the library"
	@echo "  make clean        # Clean build artifacts"
	@echo "  make info         # Show configuration"
