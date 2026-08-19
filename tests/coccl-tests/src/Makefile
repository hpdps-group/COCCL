#
# Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#

CUDA_HOME ?= /usr/local/cuda
PREFIX ?= /usr/local
VERBOSE ?= 0
DEBUG ?= 0

CUDA_LIB ?= $(CUDA_HOME)/lib64
CUDA_INC ?= $(CUDA_HOME)/include
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDARTLIB ?= cudart

CUDA_VERSION = $(strip $(shell which $(NVCC) >/dev/null && $(NVCC) --version | grep release | sed 's/.*release //' | sed 's/\,.*//'))
CUDA_MAJOR = $(shell echo $(CUDA_VERSION) | cut -d "." -f 1)
CUDA_MINOR = $(shell echo $(CUDA_VERSION) | cut -d "." -f 2)
# Better define NVCC_GENCODE in your environment to the minimal set
# of archs to reduce compile time.
ifeq ($(shell test "0$(CUDA_MAJOR)" -eq 12 -a "0$(CUDA_MINOR)" -ge 8 -o "0$(CUDA_MAJOR)" -ge 13; echo $$?),0)
# Include Blackwell support if we're using CUDA12.8 or above
NVCC_GENCODE ?=	-gencode=arch=compute_80,code=sm_80 \
		-gencode=arch=compute_90,code=sm_90 \
		-gencode=arch=compute_100,code=sm_100 \
		-gencode=arch=compute_120,code=sm_120 \
		-gencode=arch=compute_120,code=compute_120
else ifeq ($(shell test "0$(CUDA_MAJOR)" -ge 12; echo $$?),0)
NVCC_GENCODE ?= -gencode=arch=compute_60,code=sm_60 \
                -gencode=arch=compute_61,code=sm_61 \
                -gencode=arch=compute_70,code=sm_70 \
		-gencode=arch=compute_80,code=sm_80 \
		-gencode=arch=compute_90,code=sm_90 \
		-gencode=arch=compute_90,code=compute_90
else ifeq ($(shell test "0$(CUDA_MAJOR)" -ge 11; echo $$?),0)
NVCC_GENCODE ?= -gencode=arch=compute_60,code=sm_60 \
                -gencode=arch=compute_61,code=sm_61 \
                -gencode=arch=compute_70,code=sm_70 \
		-gencode=arch=compute_80,code=sm_80 \
		-gencode=arch=compute_80,code=compute_80
else
NVCC_GENCODE ?= -gencode=arch=compute_35,code=sm_35 \
                -gencode=arch=compute_50,code=sm_50 \
                -gencode=arch=compute_60,code=sm_60 \
                -gencode=arch=compute_61,code=sm_61 \
                -gencode=arch=compute_70,code=sm_70 \
                -gencode=arch=compute_70,code=compute_70
endif

NVCUFLAGS  := -ccbin $(CXX) $(NVCC_GENCODE) -std=c++11
CXXFLAGS   := -std=c++11

LDFLAGS    := -L${CUDA_LIB} -lcudart -lrt
NVLDFLAGS  := -L${CUDA_LIB} -l${CUDARTLIB} -lrt

ifeq ($(DEBUG), 0)
NVCUFLAGS += -O3 -g
CXXFLAGS  += -O3 -g
else
NVCUFLAGS += -O0 -G -g
CXXFLAGS  += -O0 -g -ggdb3
endif

ifneq ($(VERBOSE), 0)
NVCUFLAGS += -Xcompiler -Wall,-Wextra,-Wno-unused-parameter
else
.SILENT:
endif

.PHONY: build clean m1-host-tests m1-plugin-load-test m2-host-tests \
	m3-host-tests m4-host-tests m5-host-tests m5-layout-test m6-host-tests \
	m7-host-tests m8-host-tests m9-host-tests m10-host-tests m11-host-tests \
	m12-host-tests m13-host-tests m14-host-tests m14-codec-build \
	m15-host-tests m15-layout-test

BUILDDIR ?= ../build
ifneq ($(NCCL_HOME), "")
NVCUFLAGS += -I$(NCCL_HOME)/include/
NVLDFLAGS += -L$(NCCL_HOME)/lib
endif

ifeq ($(MPI), 1)
NVCUFLAGS += -DMPI_SUPPORT -I$(MPI_HOME)/include
NVLDFLAGS += -L$(MPI_HOME)/lib -L$(MPI_HOME)/lib64 -lmpi
endif
ifeq ($(MPI_IBM),1)
NVCUFLAGS += -DMPI_SUPPORT
NVLDFLAGS += -lmpi_ibm
endif
LIBRARIES += nccl
NVLDFLAGS += $(LIBRARIES:%=-l%)

DST_DIR := $(BUILDDIR)
SRC_FILES := $(wildcard *.cu)
OBJ_FILES := $(SRC_FILES:%.cu=${DST_DIR}/%.o)
BIN_FILES_LIST := all_reduce all_gather broadcast reduce_scatter reduce alltoall scatter gather sendrecv hypercube \
alltoall_comp alltoall_p2p reduce_scatter_ring reduce_scatter_comp all_gather_comp all_gather_ring\
all_reduce_twoShot all_reduce_ring all_reduce_optim all_reduce_comp_twoshot \
reduce_scatter_comp_oneshot reduce_scatter_comp_twoshot \
all_gather_comp_twoshot all_reduce_comp_tripleshot sendrecv_comp \
all_reduce_comp_oneshot
BIN_FILES := $(BIN_FILES_LIST:%=${DST_DIR}/%_perf)

build: ${BIN_FILES}

COCCL_SRC := ../../../src
COCCL_INC := -I$(COCCL_SRC)/include -I$(NCCL_HOME)/include -I$(CUDA_INC)
M1_CXXFLAGS := $(filter-out -std=c++11,$(CXXFLAGS)) -std=c++17
M1_SDK_TEST := $(DST_DIR)/coccl_compressor_sdk_test
M1_CONFIG_READER_TEST := $(DST_DIR)/coccl_config_reader_test
M1_CONFIG_TEST := $(DST_DIR)/coccl_m1_config_test
M1_PLUGIN_LOAD_TEST := $(DST_DIR)/coccl_m1_plugin_load_test
M1_SDP_CONFIG := ../../../examples/benchmarks_scripts/configs/m1_sdp4bit.toml
M1_ZFP_CONFIG := ../../../examples/benchmarks_scripts/configs/m1_zfp.toml
M1_PLUGIN_DIR := $(NCCL_HOME)/obj/device/compress/libcompress
M2_RUNTIME_TEST := $(DST_DIR)/coccl_m2_runtime_test
M2_GROUP_TEST := $(DST_DIR)/coccl_m2_group_test
M3_BUFFER_TEST := $(DST_DIR)/coccl_m3_buffer_test
M4_VMM_TEST := $(DST_DIR)/coccl_m4_vmm_test
M5_PIPELINE_PLAN_TEST := $(DST_DIR)/coccl_m5_pipeline_plan_test
M5_PIPELINE_STAGE_TEST := $(DST_DIR)/coccl_m5_pipeline_stage_test
M5_PIPELINE_LAYOUT_TEST := $(DST_DIR)/coccl_m5_pipeline_layout_test
M6_ALLGATHER_PLAN_TEST := $(DST_DIR)/coccl_m6_allgather_plan_test
M6_ALLGATHER_STAGE_TEST := $(DST_DIR)/coccl_m6_allgather_stage_test
M7_REDUCESCATTER_PLAN_TEST := $(DST_DIR)/coccl_m7_reducescatter_plan_test
M7_REDUCESCATTER_STAGE_TEST := $(DST_DIR)/coccl_m7_reducescatter_stage_test
M8_ALLREDUCE_PLAN_TEST := $(DST_DIR)/coccl_m8_allreduce_plan_test
M8_ALLREDUCE_RECIPE_TEST := $(DST_DIR)/coccl_m8_allreduce_recipe_test
M9_FIXED_COMPRESSOR_TEST := $(DST_DIR)/coccl_m9_fixed_compressor_test
M10_UNIFIED_ARENA_TEST := $(DST_DIR)/coccl_m10_unified_arena_test
M11_SIZE_QUERY_TEST := $(DST_DIR)/coccl_m11_size_query_test
M11_WORKSPACE_TEST := $(DST_DIR)/coccl_m11_workspace_test
M11_SIZE_QUERY_STUB := coccl_m11_size_query_stub.cc
M12_INPLACE_PLAN_TEST := $(DST_DIR)/coccl_m12_inplace_plan_test
M13_FRAME_EXCHANGE_TEST := $(DST_DIR)/coccl_m13_frame_exchange_test
M13_GROUP_BATCH_TEST := $(DST_DIR)/coccl_m13_group_batch_test
M14_FRAME_STAGE_STUB := coccl_m14_frame_stage_stub.cc
M14_FRAMED_PLAN_TEST := $(DST_DIR)/coccl_m14_framed_plan_test
M14_DIETGPU_CODEC_TEST := $(DST_DIR)/coccl_m14_dietgpu_codec_test
M15_SWIZZLE_PLAN_TEST := $(DST_DIR)/coccl_m15_swizzle_plan_test
M15_REDUCESCATTER_RECIPE_TEST := $(DST_DIR)/coccl_m15_reducescatter_recipe_test

m15-layout-test: $(M5_PIPELINE_LAYOUT_TEST)
	$(M5_PIPELINE_LAYOUT_TEST)

m15-host-tests: $(M15_SWIZZLE_PLAN_TEST) \
		$(M15_REDUCESCATTER_RECIPE_TEST) $(M8_ALLREDUCE_RECIPE_TEST) \
		$(M1_PLUGIN_LOAD_TEST)
	$(M15_SWIZZLE_PLAN_TEST)
	$(M15_REDUCESCATTER_RECIPE_TEST)
	$(M8_ALLREDUCE_RECIPE_TEST)
	$(M1_PLUGIN_LOAD_TEST) $(M1_PLUGIN_DIR) dietgpu

m14-codec-build: $(M14_DIETGPU_CODEC_TEST)

m14-host-tests: $(M14_FRAMED_PLAN_TEST) $(M13_FRAME_EXCHANGE_TEST) \
		$(M1_PLUGIN_LOAD_TEST)
	$(M14_FRAMED_PLAN_TEST)
	$(M13_FRAME_EXCHANGE_TEST)
	$(M1_PLUGIN_LOAD_TEST) $(M1_PLUGIN_DIR) dietgpu

m13-host-tests: $(M13_FRAME_EXCHANGE_TEST) $(M13_GROUP_BATCH_TEST)
	$(M13_FRAME_EXCHANGE_TEST)
	$(M13_GROUP_BATCH_TEST)

m12-host-tests: $(M12_INPLACE_PLAN_TEST)
	$(M12_INPLACE_PLAN_TEST)

m11-host-tests: $(M11_SIZE_QUERY_TEST) $(M11_WORKSPACE_TEST)
	$(M11_SIZE_QUERY_TEST)
	$(M11_WORKSPACE_TEST)

m10-host-tests: $(M10_UNIFIED_ARENA_TEST)
	$(M10_UNIFIED_ARENA_TEST)

m9-host-tests: $(M9_FIXED_COMPRESSOR_TEST)
	$(M9_FIXED_COMPRESSOR_TEST) $(M1_PLUGIN_DIR)

m8-host-tests: $(M8_ALLREDUCE_PLAN_TEST) $(M8_ALLREDUCE_RECIPE_TEST)
	$(M8_ALLREDUCE_PLAN_TEST)
	$(M8_ALLREDUCE_RECIPE_TEST)

m7-host-tests: $(M7_REDUCESCATTER_PLAN_TEST) $(M7_REDUCESCATTER_STAGE_TEST)
	$(M7_REDUCESCATTER_PLAN_TEST)
	$(M7_REDUCESCATTER_STAGE_TEST)

m6-host-tests: $(M6_ALLGATHER_PLAN_TEST) $(M6_ALLGATHER_STAGE_TEST)
	$(M6_ALLGATHER_PLAN_TEST)
	$(M6_ALLGATHER_STAGE_TEST)

m5-layout-test: $(M5_PIPELINE_LAYOUT_TEST)
	$(M5_PIPELINE_LAYOUT_TEST)

m5-host-tests: $(M5_PIPELINE_PLAN_TEST) $(M5_PIPELINE_STAGE_TEST)
	$(M5_PIPELINE_PLAN_TEST)
	$(M5_PIPELINE_STAGE_TEST)

m4-host-tests: $(M4_VMM_TEST)
	$(M4_VMM_TEST)

m3-host-tests: $(M3_BUFFER_TEST)
	$(M3_BUFFER_TEST)

m2-host-tests: $(M2_RUNTIME_TEST) $(M2_GROUP_TEST)
	$(M2_RUNTIME_TEST)
	$(M2_GROUP_TEST)

m1-host-tests: $(M1_SDK_TEST) $(M1_CONFIG_READER_TEST) $(M1_CONFIG_TEST)
	$(M1_SDK_TEST)
	$(M1_CONFIG_READER_TEST)
	$(M1_CONFIG_TEST) $(M1_SDP_CONFIG) $(M1_ZFP_CONFIG)

m1-plugin-load-test: $(M1_PLUGIN_LOAD_TEST)
	$(M1_PLUGIN_LOAD_TEST) $(M1_PLUGIN_DIR)

$(M1_SDK_TEST): coccl_compressor_sdk_test.cc
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(M1_CXXFLAGS) $(COCCL_INC) -o $@ $< -L$(CUDA_LIB) -lcudart

$(M1_CONFIG_READER_TEST): coccl_config_reader_test.cc
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(M1_CXXFLAGS) $(COCCL_INC) -o $@ $<

$(M1_CONFIG_TEST): coccl_m1_config_test.cc \
		$(COCCL_SRC)/misc/coccl_config.cc \
		$(COCCL_SRC)/misc/coccl_operation.cc
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(M1_CXXFLAGS) -DCOCCL_CONFIG_PARSER_ONLY $(COCCL_INC) \
		-o $@ $^

$(M1_PLUGIN_LOAD_TEST): coccl_m1_plugin_load_test.cc
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(M1_CXXFLAGS) $(COCCL_INC) -o $@ $< -ldl

$(M9_FIXED_COMPRESSOR_TEST): coccl_m9_fixed_compressor_test.cc
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(M1_CXXFLAGS) $(COCCL_INC) -o $@ $< -ldl

$(M10_UNIFIED_ARENA_TEST): coccl_m10_unified_arena_test.cc \
		$(COCCL_SRC)/misc/coccl_pipeline_plan.cc $(M11_SIZE_QUERY_STUB)
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(M1_CXXFLAGS) $(COCCL_INC) -I$(COCCL_SRC) -o $@ $^

$(M11_SIZE_QUERY_TEST): coccl_m11_size_query_test.cc \
		$(COCCL_SRC)/misc/coccl_compressor_size.cc
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(M1_CXXFLAGS) $(COCCL_INC) -I$(COCCL_SRC) -o $@ $^

$(M11_WORKSPACE_TEST): coccl_m11_workspace_test.cc \
		$(COCCL_SRC)/misc/coccl_pipeline_plan.cc $(M11_SIZE_QUERY_STUB)
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(M1_CXXFLAGS) $(COCCL_INC) -I$(COCCL_SRC) -o $@ $^

$(M12_INPLACE_PLAN_TEST): coccl_m12_inplace_plan_test.cc \
		$(COCCL_SRC)/misc/coccl_pipeline_plan.cc $(M11_SIZE_QUERY_STUB)
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(M1_CXXFLAGS) $(COCCL_INC) -I$(COCCL_SRC) -o $@ $^

$(M13_FRAME_EXCHANGE_TEST): coccl_m13_frame_exchange_test.cc \
		$(COCCL_SRC)/misc/coccl_frame_exchange.cc
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(M1_CXXFLAGS) $(COCCL_INC) -I$(COCCL_SRC) -o $@ $^

$(M13_GROUP_BATCH_TEST): coccl_m13_group_batch_test.cc \
		$(COCCL_SRC)/misc/coccl_group.cc
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(M1_CXXFLAGS) $(COCCL_INC) -I$(COCCL_SRC) -o $@ $^

$(M14_FRAMED_PLAN_TEST): coccl_m14_framed_plan_test.cc \
		$(COCCL_SRC)/misc/coccl_pipeline_plan.cc $(M11_SIZE_QUERY_STUB)
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(M1_CXXFLAGS) $(COCCL_INC) -I$(COCCL_SRC) -o $@ $^

$(M14_DIETGPU_CODEC_TEST): coccl_m14_dietgpu_codec_test.cu
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(NVCC) -o $@ $(NVCUFLAGS) $(COCCL_INC) $< -ldl $(LDFLAGS)

$(M15_SWIZZLE_PLAN_TEST): coccl_m15_swizzle_plan_test.cc \
		$(COCCL_SRC)/misc/coccl_pipeline_plan.cc $(M11_SIZE_QUERY_STUB)
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(M1_CXXFLAGS) $(COCCL_INC) -I$(COCCL_SRC) -o $@ $^

$(M15_REDUCESCATTER_RECIPE_TEST): coccl_m15_reducescatter_recipe_test.cc \
		$(COCCL_SRC)/misc/coccl_reducescatter.cc
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(M1_CXXFLAGS) $(COCCL_INC) -I$(COCCL_SRC) -o $@ $^

$(M2_RUNTIME_TEST): coccl_m2_runtime_test.cc \
		$(COCCL_SRC)/misc/coccl_runtime.cc
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(M1_CXXFLAGS) $(COCCL_INC) -I$(COCCL_SRC) -o $@ $^

$(M2_GROUP_TEST): coccl_m2_group_test.cc \
		$(COCCL_SRC)/misc/coccl_group.cc
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(M1_CXXFLAGS) $(COCCL_INC) -I$(COCCL_SRC) -o $@ $^

$(M3_BUFFER_TEST): coccl_m3_buffer_test.cc \
		$(COCCL_SRC)/misc/coccl_buffer_management.cc \
		$(COCCL_SRC)/misc/coccl_buffer_legacy.cc
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(M1_CXXFLAGS) $(COCCL_INC) -I$(COCCL_SRC) -o $@ $^

$(M4_VMM_TEST): coccl_m4_vmm_test.cc \
		$(COCCL_SRC)/misc/coccl_buffer_management.cc \
		$(COCCL_SRC)/misc/coccl_buffer_legacy.cc \
		$(COCCL_SRC)/misc/coccl_buffer_vmm.cc
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(M1_CXXFLAGS) $(COCCL_INC) -I$(COCCL_SRC) -o $@ $^

$(M5_PIPELINE_PLAN_TEST): coccl_m5_pipeline_plan_test.cc \
		$(COCCL_SRC)/misc/coccl_pipeline_plan.cc $(M11_SIZE_QUERY_STUB)
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(M1_CXXFLAGS) $(COCCL_INC) -I$(COCCL_SRC) -o $@ $^

$(M5_PIPELINE_STAGE_TEST): coccl_m5_pipeline_stage_test.cc \
		$(COCCL_SRC)/misc/coccl_pipeline_stage.cc $(M14_FRAME_STAGE_STUB)
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(M1_CXXFLAGS) $(COCCL_INC) -I$(COCCL_SRC) -o $@ $^ \
		-L$(CUDA_LIB) -lcudart

$(M5_PIPELINE_LAYOUT_TEST): coccl_m5_pipeline_layout_test.cu \
		$(COCCL_SRC)/device/pipeline/coccl_pipeline_layout.cu Makefile
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(NVCC) -o $@ $(filter-out -std=c++11,$(NVCUFLAGS)) -std=c++17 \
		$(COCCL_INC) \
		$(filter-out Makefile,$^) $(LDFLAGS)

$(M6_ALLGATHER_PLAN_TEST): coccl_m6_allgather_plan_test.cc \
		$(COCCL_SRC)/misc/coccl_pipeline_plan.cc $(M11_SIZE_QUERY_STUB)
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(M1_CXXFLAGS) $(COCCL_INC) -I$(COCCL_SRC) -o $@ $^

$(M6_ALLGATHER_STAGE_TEST): coccl_m6_allgather_stage_test.cc \
		$(COCCL_SRC)/misc/coccl_pipeline_stage.cc $(M14_FRAME_STAGE_STUB)
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(M1_CXXFLAGS) $(COCCL_INC) -I$(COCCL_SRC) -o $@ $^ \
		-L$(CUDA_LIB) -lcudart

$(M7_REDUCESCATTER_PLAN_TEST): coccl_m7_reducescatter_plan_test.cc \
		$(COCCL_SRC)/misc/coccl_pipeline_plan.cc $(M11_SIZE_QUERY_STUB)
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(M1_CXXFLAGS) $(COCCL_INC) -I$(COCCL_SRC) -o $@ $^

$(M7_REDUCESCATTER_STAGE_TEST): coccl_m7_reducescatter_stage_test.cc \
		$(COCCL_SRC)/misc/coccl_pipeline_stage.cc $(M14_FRAME_STAGE_STUB)
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(M1_CXXFLAGS) $(COCCL_INC) -I$(COCCL_SRC) -o $@ $^ \
		-L$(CUDA_LIB) -lcudart

$(M8_ALLREDUCE_PLAN_TEST): coccl_m8_allreduce_plan_test.cc \
		$(COCCL_SRC)/misc/coccl_pipeline_plan.cc $(M11_SIZE_QUERY_STUB)
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(M1_CXXFLAGS) $(COCCL_INC) -I$(COCCL_SRC) -o $@ $^

$(M8_ALLREDUCE_RECIPE_TEST): coccl_m8_allreduce_recipe_test.cc \
		$(COCCL_SRC)/misc/coccl_allreduce.cc
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(M1_CXXFLAGS) $(COCCL_INC) -I$(COCCL_SRC) -o $@ $^

clean:
	rm -rf ${DST_DIR}

TEST_VERIFIABLE_SRCDIR := ../verifiable
TEST_VERIFIABLE_BUILDDIR := $(BUILDDIR)/verifiable
include ../verifiable/verifiable.mk

${DST_DIR}/%.o: %.cu common.h $(TEST_VERIFIABLE_HDRS)
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) -c $<

${DST_DIR}/timer.o: timer.cc timer.h
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(CXX) $(CXXFLAGS) -o $@ -c timer.cc

${DST_DIR}/%_perf:${DST_DIR}/%.o ${DST_DIR}/common.o ${DST_DIR}/timer.o $(TEST_VERIFIABLE_OBJS)
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $^ ${NVLDFLAGS}
