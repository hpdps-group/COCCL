SHELL := /usr/bin/env bash
MAKEFLAGS += -r
.SUFFIXES:
.SECONDARY:

# Directories
NCCLDIR := ../../../..
include $(NCCLDIR)/makefiles/common.mk
include $(NCCLDIR)/makefiles/version.mk

BUILDDIR ?= $(abspath ../../../../build)
# OBJDIR := $(BUILDDIR)/obj/device/compress/sdp4bit

OBJDIR := $(SUBOBJDIR)/sdp4bit/
# $(info PWD is ${PWD})
# $(info SUBOBJDIR is ${SUBOBJDIR})


# QUAN_SO := $(BUILDDIR)/obj/device/compress/libcompress/libsdp4bit.so
QUAN_SO := $(SUBOBJDIR)/libcompress/libsdp4bit.so

# SRCDIR := $(PWD)
# BUILDDIR := $(SRCDIR)/build
# OBJDIR := $(BUILDDIR)

INCDIR := includes
QUAN_DIR := quantization
HADAMRD_DIR := hadamard

# TORCH_INCLUDE_CPU := $(shell python -c "import torch; import os; print(os.path.join(torch.__path__[0], 'include', 'torch', 'csrc', 'api', 'include'))")
# TORCH_INCLUDE_CUDA := $(shell python -c "from torch.utils.cpp_extension import include_paths; print(include_paths(cuda=True)[0])" 2>/dev/null || echo "")
# TORCH_LIB := $(shell python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")

# ifeq ($(strip $(TORCH_INCLUDE_CUDA)),)
#     $(error "PyTorch CUDA include path not found! Ensure PyTorch with CUDA support is installed")
# endif
ifeq ($(NVHPC_CUDA_HOME),)
  NVHPC_CUDA_HOME := /usr/local/cuda
endif

# PY_INCLUDE := $(shell python -c "import sysconfig; print(sysconfig.get_path('include'))")

# $(info PY_INCLUDE = $(PY_INCLUDE))

CUDA_LIB := $(NVHPC_CUDA_HOME)/lib64
CUDA_INC := $(NVHPC_CUDA_HOME)/include

# Compiler flags
NVCC_GENCODE ?= -gencode=arch=compute_80,code=sm_80

INCFLAGS := -I$(CUDA_INC) \
            -I$(INCDIR) \
            -I. -I../.. -I$(BUILDDIR)/include -I../../../include \
# -I. -I$(INCDIR)
# @echo $(INCFLAGS)

NVCUFLAGS += $(INCFLAGS) \
            -U__CUDA_NO_HALF_OPERATORS__ \
            -U__CUDA_NO_HALF_CONVERSIONS__ \
            -U__CUDA_NO_HALF2_OPERATORS__ \
            -U__CUDA_NO_BFLOAT16_OPERATORS__ \
            -U__CUDA_NO_BFLOAT16_CONVERSIONS__ \
            -U__CUDA_NO_BFLOAT162_OPERATORS__ \
            -U__CUDA_NO_BFLOAT162_CONVERSIONS__ \
            -DBF16_AVAILABLE \
            --compiler-options "-fPIC" \
            --expt-relaxed-constexpr \
            $(NVCC_GENCODE)
# -std=c++17 
# -Xcompiler -fPIC -O3 
#  -std=c++17 

NVLDFLAGS := -L$(CUDA_LIB) \
             -lcudart
# -O3 -fPIC
CXXFLAGS += --compiler-options "-fPIC"  $(INCFLAGS)
# Output files
# EXEC = $(BUILDDIR)/spd4bit_test
# SO_LIB = $(BUILDDIR)/spd4bit_quan.so

# Source files
SRCS = $(wildcard ./*.cu $(QUAN_DIR)/*.cu $(HADAMRD_DIR)/*.cu ./*.cc $(QUAN_DIR)/*.cc $(HADAMRD_DIR)/*.cc )
# $(info OBJDIR is ${OBJDIR})

# OBJS = $(patsubst $(SRCDIR)/%, $(OBJDIR)/%, $(SRCS:.cc=.o))
OBJS = $(patsubst %, $(OBJDIR)/%.o, $(SRCS))
# $(info OBJS is ${OBJS})

# OBJS := $(OBJS:.cu=.o)
DEPS = $(OBJS:.o=.d)

all: $(QUAN_SO)

SAY = @bash -c 'path="$$2"; [[ "$$(realpath "$$2")" =~ ^$(subst .,\.,$(abspath $(NCCLDIR)))/(.*)$$ ]] && path="$${BASH_REMATCH[1]}"; printf "%-15s %s\n" "$$1" "$$path"' SAY

COMPILE.cu = $(NVCC) $(NVCUFLAGS) -dc $2 -o $1
COMPILE.cpp = $(CXX) $(CXXFLAGS) -c $2 -o $1
COMPILE.cc = $(CXX) $(CXXFLAGS) -c $2 -o $1
define COMPILE
@$(SAY) "Compiling" $2;\
 mkdir -p $(dir $1);\
 $(call COMPILE$(suffix $2),$1,$2)
endef


DEPENDS.cu = $(NVCC) $(NVCUFLAGS) -M -dc $1
DEPENDS.cc = $(CXX) $(CXXFLAGS) -M -c $1
DEPENDS.cpp = $(CXX) $(CXXFLAGS) -M -c $1
define DEPENDS
@$(SAY) "Dependencies" $2;\
 mkdir -p $(dir $1);\
 mk=$$($(call DEPENDS$(suffix $2),$2));\
 [[ $$mk =~ ^[^:]*:(.*)$$ ]];\
 files=$${BASH_REMATCH[1]};\
 files=$$(for x in $$files; do case "$$x" in '\'|$$'\t') ;; *) echo "$$x"; esac; done);\
 files=$$(for x in $$files; do [[ "$$(realpath "$$x")" == "$$(realpath "$(NCCLDIR)")"* ]] && echo "$$x"; done);\
 echo "$(patsubst %.d,%.o,$1) $1: " $$files > $1
endef



$(OBJDIR)/%.o: % $(OBJDIR)/%.d
	$(call COMPILE,$@,$<)

$(OBJDIR)/%.d: %
	$(call DEPENDS,$@,$<)

$(QUAN_SO): $(OBJS)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCUFLAGS) $(NVLDFLAGS) $(OBJS) -shared  -o $@




# $(EXEC): $(OBJS)
# 	@mkdir -p $(dir $@)
# 	$(NVCC) $(OBJS) $(NVCUFLAGS) $(NVLDFLAGS)  -o $@ 
# $(OBJDIR)/sdp4bit_test: $(LIB_OBJS)
#     $(NVCC) $(NVCUFLAGS) $^ $(NVLDFLAGS) -o $@ \
#             -D_GLIBCXX_USE_CXX11_ABI=0


# $(OBJDIR)/libs: $(LIB_OBJS)
# 	@$(NVCC) $^ $(NVCUFLAGS) $(NVLDFLAGS) \
#            -shared -o $(OBJDIR)/spd4bit_quan.so -D_GLIBCXX_USE_CXX11_ABI=0
#            ar rcs $(OBJDIR)/spd4bit_quan.a $^

# Compile CUDA source to object files
# $(OBJDIR)/%.o: % $(OBJDIR)/%.d
# 	$(call COMPILE,$@,$<)

# # Generate dependencies for CUDA and C++ files
# $(OBJDIR)/%.d: %
# 	$(call DEPENDS,$@,$<)

-include $(wildcard $(OBJDIR)/*.d)
-include $(wildcard $(OBJDIR)/compress/sdp4bit/*.d)
-include $(wildcard $(OBJDIR)/compress/sdp4bit/hadamard/*.d)
-include $(wildcard $(OBJDIR)/compress/sdp4bit/quantization/*.d)

# $(OBJDIR)/genobj/%.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/%.d
# 	$(call COMPILE,$@,$(OBJDIR)/gensrc/$*)


# %.o: %.cu
# 	@$(NVCC) -c $< -o $@ $(COMMON_INCLUDES) $(NVCUFLAGS) \
#             --compiler-options '-fPIC'  \
#             -D_GLIBCXX_USE_CXX11_ABI=0
# #'--expt-relaxed-constexpr'
# # 链接规则
# $(OUTPUT_NAME): $(OBJ_FILES)
# 	@$(NVCC) $^ $(COMMON_INCLUDES) $(COMMON_LDFLAGS) \
#             -o $@ -D_GLIBCXX_USE_CXX11_ABI=0
.PHONY: clean
clean:
	rm -rf $(OBJDIR)
	rm -f $(EXEC) $(SO_LIB)
