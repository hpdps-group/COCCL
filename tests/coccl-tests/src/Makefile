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

ifneq ($(filter 0 release,$(DEBUG)),)
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

.PHONY: build clean host-tests config-tests plugin-load-test runtime-tests \
	buffer-tests pipeline-tests pipeline-layout-test compression-tests \
	framed-tests autotune-tests autotune-model-selector-build training-tests \
	training-mpi-build dietgpu-codec-build dietgpu-integer-build \
	collective-correctness-build coll-config-build

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
alltoall_comp alltoall_p2p all_gather_comp all_reduce_comp_twoshot \
reduce_scatter_comp_oneshot reduce_scatter_comp_twoshot \
all_reduce_comp_tripleshot sendrecv_comp all_reduce_comp_oneshot
BIN_FILES := $(BIN_FILES_LIST:%=${DST_DIR}/%_perf)

build: ${BIN_FILES}

COCCL_SRC := ../../../src
COCCL_ROOT := $(COCCL_SRC)/coccl-extend
COCCL_CORE := $(COCCL_ROOT)/core
COCCL_EXTENSIONS := $(COCCL_ROOT)/extensions
COCCL_INC := -I$(COCCL_ROOT)/include -I$(COCCL_ROOT) \
	-I$(COCCL_SRC)/include -I$(COCCL_SRC)/include/plugin \
	-I$(NCCL_HOME)/include -I$(CUDA_INC) -DNCCL_OS_LINUX
TEST_DEBUG_STUB_OBJ := $(DST_DIR)/coccl_test_debug_stub.o
TEST_CXXFLAGS := $(filter-out -std=c++11,$(CXXFLAGS)) -std=c++17 \
	$(TEST_DEBUG_STUB_OBJ)
SDK_TEST := $(DST_DIR)/coccl_compressor_sdk_test
CONFIG_READER_TEST := $(DST_DIR)/coccl_config_reader_test
CONFIG_POLICY_TEST := $(DST_DIR)/coccl_config_policy_test
PLUGIN_LOAD_TEST := $(DST_DIR)/coccl_plugin_load_test
SDP_CONFIG := ../../../examples/benchmarks_scripts/configs/sdp4bit.toml
ZFP_CONFIG := ../../../examples/benchmarks_scripts/configs/zfp.toml
PLUGIN_DIR := $(NCCL_HOME)/obj/coccl-extend/compressor_plugin/libcompress
RUNTIME_TEST := $(DST_DIR)/coccl_runtime_test
GROUP_TEST := $(DST_DIR)/coccl_group_test
BUFFER_MANAGER_TEST := $(DST_DIR)/coccl_buffer_manager_test
VMM_BUFFER_TEST := $(DST_DIR)/coccl_vmm_buffer_test
PIPELINE_PLAN_TEST := $(DST_DIR)/coccl_pipeline_plan_test
PIPELINE_STAGE_TEST := $(DST_DIR)/coccl_pipeline_stage_test
PIPELINE_LAYOUT_TEST := $(DST_DIR)/coccl_pipeline_layout_test
ALLGATHER_PLAN_TEST := $(DST_DIR)/coccl_allgather_plan_test
ALLGATHER_STAGE_TEST := $(DST_DIR)/coccl_allgather_stage_test
REDUCESCATTER_PLAN_TEST := $(DST_DIR)/coccl_reducescatter_plan_test
REDUCESCATTER_STAGE_TEST := $(DST_DIR)/coccl_reducescatter_stage_test
ALLREDUCE_PLAN_TEST := $(DST_DIR)/coccl_allreduce_plan_test
ALLREDUCE_RECIPE_TEST := $(DST_DIR)/coccl_allreduce_recipe_test
FIXED_COMPRESSOR_TEST := $(DST_DIR)/coccl_fixed_compressor_test
UNIFIED_ARENA_TEST := $(DST_DIR)/coccl_unified_arena_test
SIZE_QUERY_TEST := $(DST_DIR)/coccl_size_query_test
PIPELINE_WORKSPACE_TEST := $(DST_DIR)/coccl_pipeline_workspace_test
SIZE_QUERY_STUB := coccl_size_query_stub.cc
INPLACE_PLAN_TEST := $(DST_DIR)/coccl_inplace_plan_test
FRAME_EXCHANGE_TEST := $(DST_DIR)/coccl_frame_exchange_test
GROUP_BATCH_TEST := $(DST_DIR)/coccl_group_batch_test
FRAME_STAGE_STUB := coccl_frame_stage_stub.cc
FRAMED_PIPELINE_PLAN_TEST := $(DST_DIR)/coccl_framed_pipeline_plan_test
DIETGPU_CODEC_TEST := $(DST_DIR)/coccl_dietgpu_codec_test
SWIZZLE_PLAN_TEST := $(DST_DIR)/coccl_swizzle_plan_test
REDUCESCATTER_RECIPE_TEST := $(DST_DIR)/coccl_reducescatter_recipe_test
COLLECTIVE_CORRECTNESS_TEST := $(DST_DIR)/coccl_collective_correctness
COLL_CONFIG_TEST := $(DST_DIR)/coccl_coll_config_test
AUTOTUNE_TEST := $(DST_DIR)/coccl_autotune_test
AUTOTUNE_MODEL_SELECTOR := $(DST_DIR)/coccl_autotune_model_selector
AUTOTUNE_TOPOLOGY_TEST := $(DST_DIR)/coccl_autotune_topology_test
AUTOTUNE_PIPELINE_QUERY_TEST := $(DST_DIR)/coccl_autotune_pipeline_query_test
AUTOTUNE_COST_MODEL_SRC := $(COCCL_CORE)/tuning/coccl_autotune_cost_model.cc
AUTOTUNE_INTERNAL := $(COCCL_CORE)/tuning/coccl_autotune_internal.h
SCOPE_CONFIG_TEST := $(DST_DIR)/coccl_scope_config_test
HIERARCHICAL_GRAPH_TEST := $(DST_DIR)/coccl_hierarchical_graph_test
CODEC_FLOW_TEST := $(DST_DIR)/coccl_codec_flow_test
SELECTOR_SCOPE_TEST := $(DST_DIR)/coccl_selector_scope_test
REMAINDER_PLAN_TEST := $(DST_DIR)/coccl_remainder_plan_test
TRAINING_CLASSIFIER_TEST := $(DST_DIR)/coccl_training_classifier_test
TRAINING_KNOWN_ROLES_TEST := $(DST_DIR)/coccl_training_known_roles
TRAINING_CONFIG := ../../../examples/training_scripts/configs/training.toml
CONFIG_TEST_TMP := $(DST_DIR)/config-test-tmp
DIETGPU_INTEGER_TEST := $(DST_DIR)/coccl_dietgpu_integer_test

HOST_TEST_BINS := $(SDK_TEST) $(CONFIG_READER_TEST) $(CONFIG_POLICY_TEST) \
	$(PLUGIN_LOAD_TEST) $(RUNTIME_TEST) $(GROUP_TEST) \
	$(BUFFER_MANAGER_TEST) $(VMM_BUFFER_TEST) $(PIPELINE_PLAN_TEST) \
	$(PIPELINE_STAGE_TEST) $(ALLGATHER_PLAN_TEST) $(ALLGATHER_STAGE_TEST) \
	$(REDUCESCATTER_PLAN_TEST) $(REDUCESCATTER_STAGE_TEST) \
	$(ALLREDUCE_PLAN_TEST) $(ALLREDUCE_RECIPE_TEST) \
	$(FIXED_COMPRESSOR_TEST) $(UNIFIED_ARENA_TEST) $(SIZE_QUERY_TEST) \
	$(PIPELINE_WORKSPACE_TEST) $(INPLACE_PLAN_TEST) $(FRAME_EXCHANGE_TEST) \
	$(GROUP_BATCH_TEST) $(FRAMED_PIPELINE_PLAN_TEST) $(AUTOTUNE_TEST) \
	$(AUTOTUNE_TOPOLOGY_TEST) \
	$(AUTOTUNE_MODEL_SELECTOR) \
	$(SCOPE_CONFIG_TEST) $(HIERARCHICAL_GRAPH_TEST) $(CODEC_FLOW_TEST) \
	$(SELECTOR_SCOPE_TEST) $(REMAINDER_PLAN_TEST) $(TRAINING_CLASSIFIER_TEST)

$(HOST_TEST_BINS): | $(TEST_DEBUG_STUB_OBJ)

$(TEST_DEBUG_STUB_OBJ): coccl_test_debug_stub.cc
	@mkdir -p $(DST_DIR)
	$(CXX) $(filter-out $(TEST_DEBUG_STUB_OBJ),$(TEST_CXXFLAGS)) \
		$(COCCL_INC) -c $< -o $@

host-tests: config-tests runtime-tests buffer-tests pipeline-tests \
	compression-tests framed-tests autotune-tests training-tests

config-tests: $(SDK_TEST) $(CONFIG_READER_TEST) $(CONFIG_POLICY_TEST) \
		$(SCOPE_CONFIG_TEST)
	@mkdir -p $(CONFIG_TEST_TMP)
	$(SDK_TEST)
	$(CONFIG_READER_TEST)
	$(CONFIG_POLICY_TEST) $(SDP_CONFIG) $(ZFP_CONFIG)
	$(SCOPE_CONFIG_TEST) $(CONFIG_TEST_TMP) $(SDP_CONFIG) $(ZFP_CONFIG) \
		../../../examples/benchmarks_scripts/configs/dietgpu.toml \
		../../../src/coccl-extend/extensions/configs/normal-mixed.toml \
		$(TRAINING_CONFIG)

plugin-load-test: $(PLUGIN_LOAD_TEST)
	$(PLUGIN_LOAD_TEST) $(PLUGIN_DIR)

runtime-tests: $(RUNTIME_TEST) $(GROUP_TEST)
	$(RUNTIME_TEST)
	$(GROUP_TEST)

buffer-tests: $(BUFFER_MANAGER_TEST) $(VMM_BUFFER_TEST)
	$(BUFFER_MANAGER_TEST)
	$(VMM_BUFFER_TEST)

pipeline-tests: $(PIPELINE_PLAN_TEST) $(PIPELINE_STAGE_TEST) \
		$(ALLGATHER_PLAN_TEST) $(ALLGATHER_STAGE_TEST) \
		$(REDUCESCATTER_PLAN_TEST) $(REDUCESCATTER_STAGE_TEST) \
		$(ALLREDUCE_PLAN_TEST) $(ALLREDUCE_RECIPE_TEST) \
		$(UNIFIED_ARENA_TEST) $(SIZE_QUERY_TEST) \
		$(PIPELINE_WORKSPACE_TEST) $(INPLACE_PLAN_TEST) \
		$(SWIZZLE_PLAN_TEST) $(REDUCESCATTER_RECIPE_TEST) \
		$(HIERARCHICAL_GRAPH_TEST) $(REMAINDER_PLAN_TEST)
	$(PIPELINE_PLAN_TEST)
	$(PIPELINE_STAGE_TEST)
	$(ALLGATHER_PLAN_TEST)
	$(ALLGATHER_STAGE_TEST)
	$(REDUCESCATTER_PLAN_TEST)
	$(REDUCESCATTER_STAGE_TEST)
	$(ALLREDUCE_PLAN_TEST)
	$(ALLREDUCE_RECIPE_TEST)
	$(UNIFIED_ARENA_TEST)
	$(SIZE_QUERY_TEST)
	$(PIPELINE_WORKSPACE_TEST)
	$(INPLACE_PLAN_TEST)
	$(SWIZZLE_PLAN_TEST)
	$(REDUCESCATTER_RECIPE_TEST)
	$(HIERARCHICAL_GRAPH_TEST)
	$(REMAINDER_PLAN_TEST)

pipeline-layout-test: $(PIPELINE_LAYOUT_TEST)
	$(PIPELINE_LAYOUT_TEST)

compression-tests: plugin-load-test $(FIXED_COMPRESSOR_TEST) \
		$(CODEC_FLOW_TEST)
	$(PLUGIN_LOAD_TEST) $(PLUGIN_DIR) dietgpu
	$(FIXED_COMPRESSOR_TEST) $(PLUGIN_DIR)
	$(CODEC_FLOW_TEST)

framed-tests: $(FRAMED_PIPELINE_PLAN_TEST) $(FRAME_EXCHANGE_TEST) \
		$(GROUP_BATCH_TEST)
	$(FRAMED_PIPELINE_PLAN_TEST)
	$(FRAME_EXCHANGE_TEST)
	$(GROUP_BATCH_TEST)

autotune-tests: $(AUTOTUNE_TEST) $(AUTOTUNE_TOPOLOGY_TEST) \
		$(AUTOTUNE_PIPELINE_QUERY_TEST) \
		$(SELECTOR_SCOPE_TEST)
	$(AUTOTUNE_TEST)
	$(AUTOTUNE_TOPOLOGY_TEST)
	$(AUTOTUNE_PIPELINE_QUERY_TEST)
	$(SELECTOR_SCOPE_TEST)

training-tests: $(TRAINING_CLASSIFIER_TEST)
	$(TRAINING_CLASSIFIER_TEST)

autotune-model-selector-build: $(AUTOTUNE_MODEL_SELECTOR)

training-mpi-build: $(TRAINING_KNOWN_ROLES_TEST)

dietgpu-codec-build: $(DIETGPU_CODEC_TEST)

dietgpu-integer-build: $(DIETGPU_INTEGER_TEST)

collective-correctness-build: $(COLLECTIVE_CORRECTNESS_TEST)

coll-config-build: $(COLL_CONFIG_TEST)

$(SDK_TEST): coccl_compressor_sdk_test.cc
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(TEST_CXXFLAGS) $(COCCL_INC) -o $@ $< -L$(CUDA_LIB) -lcudart

$(CONFIG_READER_TEST): coccl_config_reader_test.cc
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(TEST_CXXFLAGS) $(COCCL_INC) -o $@ $<

$(CONFIG_POLICY_TEST): coccl_config_policy_test.cc \
		$(COCCL_CORE)/config/coccl_config.cc \
		$(COCCL_CORE)/runtime/coccl_operation.cc
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(TEST_CXXFLAGS) -DCOCCL_CONFIG_PARSER_ONLY $(COCCL_INC) \
		-o $@ $^

$(PLUGIN_LOAD_TEST): coccl_plugin_load_test.cc
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(TEST_CXXFLAGS) $(COCCL_INC) -o $@ $< -ldl \
		-L$(CUDA_LIB) -Wl,--no-as-needed -lcudart

$(FIXED_COMPRESSOR_TEST): coccl_fixed_compressor_test.cc
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(TEST_CXXFLAGS) $(COCCL_INC) -o $@ $< -ldl

$(UNIFIED_ARENA_TEST): coccl_unified_arena_test.cc \
		$(COCCL_CORE)/pipeline/coccl_pipeline_plan.cc $(SIZE_QUERY_STUB)
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(TEST_CXXFLAGS) $(COCCL_INC) -I$(COCCL_SRC) -o $@ $^

$(SIZE_QUERY_TEST): coccl_size_query_test.cc \
		$(COCCL_CORE)/compression/coccl_compressor_size.cc
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(TEST_CXXFLAGS) $(COCCL_INC) -I$(COCCL_SRC) -o $@ $^

$(PIPELINE_WORKSPACE_TEST): coccl_pipeline_workspace_test.cc \
		$(COCCL_CORE)/pipeline/coccl_pipeline_plan.cc $(SIZE_QUERY_STUB)
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(TEST_CXXFLAGS) $(COCCL_INC) -I$(COCCL_SRC) -o $@ $^

$(INPLACE_PLAN_TEST): coccl_inplace_plan_test.cc \
		$(COCCL_CORE)/pipeline/coccl_pipeline_plan.cc $(SIZE_QUERY_STUB)
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(TEST_CXXFLAGS) $(COCCL_INC) -I$(COCCL_SRC) -o $@ $^

$(FRAME_EXCHANGE_TEST): coccl_frame_exchange_test.cc \
		$(COCCL_CORE)/pipeline/coccl_frame_exchange.cc
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(TEST_CXXFLAGS) $(COCCL_INC) -I$(COCCL_SRC) -o $@ $^

$(GROUP_BATCH_TEST): coccl_group_batch_test.cc \
		$(COCCL_CORE)/runtime/coccl_group.cc
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(TEST_CXXFLAGS) $(COCCL_INC) -I$(COCCL_SRC) -o $@ $^

$(FRAMED_PIPELINE_PLAN_TEST): coccl_framed_pipeline_plan_test.cc \
		$(COCCL_CORE)/pipeline/coccl_pipeline_plan.cc $(SIZE_QUERY_STUB)
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(TEST_CXXFLAGS) $(COCCL_INC) -I$(COCCL_SRC) -o $@ $^

$(DIETGPU_CODEC_TEST): coccl_dietgpu_codec_test.cu
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(NVCC) -o $@ $(NVCUFLAGS) $(COCCL_INC) $< -ldl $(LDFLAGS)

$(SWIZZLE_PLAN_TEST): coccl_swizzle_plan_test.cc \
		$(COCCL_CORE)/pipeline/coccl_pipeline_plan.cc $(SIZE_QUERY_STUB)
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(TEST_CXXFLAGS) $(COCCL_INC) -I$(COCCL_SRC) -o $@ $^

$(REDUCESCATTER_RECIPE_TEST): coccl_reducescatter_recipe_test.cc \
		$(COCCL_EXTENSIONS)/primitives/coccl_reducescatter.cc \
		$(COCCL_EXTENSIONS)/primitives/coccl_hierarchical_reduction.cc
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(TEST_CXXFLAGS) $(COCCL_INC) -I$(COCCL_SRC) -o $@ $^

$(COLLECTIVE_CORRECTNESS_TEST): coccl_collective_correctness.cu
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(NVCC) -o $@ $(NVCUFLAGS) $< $(NVLDFLAGS)

$(COLL_CONFIG_TEST): coccl_coll_config_test.cu
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(NVCC) -o $@ $(NVCUFLAGS) $< $(NVLDFLAGS)

$(AUTOTUNE_TEST): coccl_autotune_test.cc \
		$(AUTOTUNE_COST_MODEL_SRC) $(AUTOTUNE_INTERNAL)
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(TEST_CXXFLAGS) $(COCCL_INC) -I$(COCCL_SRC) \
		-o $@ $< $(AUTOTUNE_COST_MODEL_SRC)

$(AUTOTUNE_MODEL_SELECTOR): coccl_autotune_model_selector.cc \
		$(AUTOTUNE_COST_MODEL_SRC) $(AUTOTUNE_INTERNAL)
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(TEST_CXXFLAGS) $(COCCL_INC) -I$(COCCL_SRC) \
		-o $@ $< $(AUTOTUNE_COST_MODEL_SRC)

$(AUTOTUNE_TOPOLOGY_TEST): coccl_autotune_topology_test.cc \
		$(COCCL_CORE)/tuning/coccl_autotune_topology.cc \
		$(AUTOTUNE_COST_MODEL_SRC) $(AUTOTUNE_INTERNAL)
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(TEST_CXXFLAGS) $(COCCL_INC) -I$(COCCL_SRC) \
		-o $@ $< $(COCCL_CORE)/tuning/coccl_autotune_topology.cc \
		$(AUTOTUNE_COST_MODEL_SRC)

$(AUTOTUNE_PIPELINE_QUERY_TEST): coccl_autotune_pipeline_query_test.cc \
		$(COCCL_CORE)/tuning/coccl_autotune_pipeline.cc \
		$(AUTOTUNE_COST_MODEL_SRC)
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(TEST_CXXFLAGS) $(COCCL_INC) -I$(COCCL_SRC) \
		-o $@ $^

$(SCOPE_CONFIG_TEST): coccl_scope_config_test.cc \
		$(COCCL_CORE)/config/coccl_config.cc \
		$(COCCL_CORE)/config/coccl_config_format.cc \
		$(COCCL_CORE)/runtime/coccl_operation.cc
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(TEST_CXXFLAGS) -DCOCCL_CONFIG_PARSER_ONLY $(COCCL_INC) \
		-o $@ $^

$(HIERARCHICAL_GRAPH_TEST): coccl_hierarchical_graph_test.cc \
		$(COCCL_EXTENSIONS)/primitives/coccl_hierarchical_reduction.cc \
		$(COCCL_CORE)/pipeline/coccl_pipeline_plan.cc $(SIZE_QUERY_STUB)
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(TEST_CXXFLAGS) $(COCCL_INC) -I$(COCCL_SRC) -o $@ $^

$(CODEC_FLOW_TEST): coccl_codec_flow_test.cc \
		$(COCCL_CORE)/pipeline/coccl_pipeline_stage.cc
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(TEST_CXXFLAGS) $(COCCL_INC) -I$(COCCL_SRC) -o $@ $^ \
		-L$(CUDA_LIB) -lcudart

$(SELECTOR_SCOPE_TEST): coccl_selector_scope_test.cc \
		$(COCCL_CORE)/tuning/coccl_autotune_selector.cc \
		$(COCCL_CORE)/tuning/coccl_autotune_cost_model.cc \
		$(COCCL_CORE)/runtime/coccl_prepared_call.cc
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(TEST_CXXFLAGS) $(COCCL_INC) -I$(COCCL_SRC) -o $@ $^

$(DIETGPU_INTEGER_TEST): coccl_dietgpu_integer_test.cu
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(NVCC) -o $@ $(NVCUFLAGS) $< $(NVLDFLAGS)

$(RUNTIME_TEST): coccl_runtime_test.cc \
		$(COCCL_CORE)/runtime/coccl_runtime.cc \
		$(COCCL_CORE)/runtime/coccl_prepared_call.cc
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(TEST_CXXFLAGS) $(COCCL_INC) -I$(COCCL_SRC) -o $@ $^

$(GROUP_TEST): coccl_group_test.cc \
		$(COCCL_CORE)/runtime/coccl_group.cc
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(TEST_CXXFLAGS) $(COCCL_INC) -I$(COCCL_SRC) -o $@ $^

$(BUFFER_MANAGER_TEST): coccl_buffer_manager_test.cc \
		$(COCCL_CORE)/memory/coccl_buffer_management.cc \
		$(COCCL_CORE)/memory/coccl_buffer_legacy.cc
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(TEST_CXXFLAGS) $(COCCL_INC) -I$(COCCL_SRC) -o $@ $^

$(VMM_BUFFER_TEST): coccl_vmm_buffer_test.cc \
		$(COCCL_CORE)/memory/coccl_buffer_management.cc \
		$(COCCL_CORE)/memory/coccl_buffer_legacy.cc \
		$(COCCL_CORE)/memory/coccl_buffer_vmm.cc
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(TEST_CXXFLAGS) $(COCCL_INC) -I$(COCCL_SRC) -o $@ $^

$(PIPELINE_PLAN_TEST): coccl_pipeline_plan_test.cc \
		$(COCCL_CORE)/pipeline/coccl_pipeline_plan.cc $(SIZE_QUERY_STUB)
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(TEST_CXXFLAGS) $(COCCL_INC) -I$(COCCL_SRC) -o $@ $^

$(REMAINDER_PLAN_TEST): coccl_remainder_plan_test.cc \
		$(COCCL_CORE)/pipeline/coccl_pipeline_plan.cc $(SIZE_QUERY_STUB)
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(TEST_CXXFLAGS) $(COCCL_INC) -I$(COCCL_SRC) -o $@ $^

$(TRAINING_CLASSIFIER_TEST): coccl_training_classifier_test.cc \
		$(COCCL_CORE)/training/coccl_training_classifier.cc \
		$(COCCL_CORE)/training/coccl_training_assist.cc \
		$(COCCL_CORE)/runtime/coccl_runtime.cc \
		$(COCCL_CORE)/runtime/coccl_prepared_call.cc \
		$(COCCL_CORE)/config/coccl_config.cc \
		$(COCCL_CORE)/runtime/coccl_operation.cc
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(TEST_CXXFLAGS) -DCOCCL_CONFIG_PARSER_ONLY \
		-DCOCCL_TRAINING_CONFIG_FILE='"$(TRAINING_CONFIG)"' \
		-DCOCCL_TRAINING_TMP_CONFIG='"$(DST_DIR)/training-required-fields.toml"' \
		$(COCCL_INC) -I$(COCCL_SRC) \
		-ffunction-sections -fdata-sections -Wl,--gc-sections \
		-pthread -o $@ $^

$(TRAINING_KNOWN_ROLES_TEST): coccl_training_known_roles.cc
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(MPI_HOME)/bin/mpicxx $(TEST_CXXFLAGS) $(COCCL_INC) -I$(COCCL_SRC) \
		-o $@ $< -L$(NCCL_HOME)/lib -lnccl \
		-L$(CUDA_LIB) -lcudart -lpthread

$(PIPELINE_STAGE_TEST): coccl_pipeline_stage_test.cc \
		$(COCCL_CORE)/pipeline/coccl_pipeline_stage.cc $(FRAME_STAGE_STUB)
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(TEST_CXXFLAGS) $(COCCL_INC) -I$(COCCL_SRC) -o $@ $^ \
		-L$(CUDA_LIB) -lcudart

$(PIPELINE_LAYOUT_TEST): coccl_pipeline_layout_test.cu \
		$(COCCL_CORE)/device/pipeline/coccl_pipeline_layout.cu Makefile
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(NVCC) -o $@ $(filter-out -std=c++11,$(NVCUFLAGS)) -std=c++17 \
		$(COCCL_INC) \
		$(filter-out Makefile,$^) $(LDFLAGS)

$(ALLGATHER_PLAN_TEST): coccl_allgather_plan_test.cc \
		$(COCCL_CORE)/pipeline/coccl_pipeline_plan.cc $(SIZE_QUERY_STUB)
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(TEST_CXXFLAGS) $(COCCL_INC) -I$(COCCL_SRC) -o $@ $^

$(ALLGATHER_STAGE_TEST): coccl_allgather_stage_test.cc \
		$(COCCL_CORE)/pipeline/coccl_pipeline_stage.cc $(FRAME_STAGE_STUB)
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(TEST_CXXFLAGS) $(COCCL_INC) -I$(COCCL_SRC) -o $@ $^ \
		-L$(CUDA_LIB) -lcudart

$(REDUCESCATTER_PLAN_TEST): coccl_reducescatter_plan_test.cc \
		$(COCCL_CORE)/pipeline/coccl_pipeline_plan.cc $(SIZE_QUERY_STUB)
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(TEST_CXXFLAGS) $(COCCL_INC) -I$(COCCL_SRC) -o $@ $^

$(REDUCESCATTER_STAGE_TEST): coccl_reducescatter_stage_test.cc \
		$(COCCL_CORE)/pipeline/coccl_pipeline_stage.cc $(FRAME_STAGE_STUB)
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(TEST_CXXFLAGS) $(COCCL_INC) -I$(COCCL_SRC) -o $@ $^ \
		-L$(CUDA_LIB) -lcudart

$(ALLREDUCE_PLAN_TEST): coccl_allreduce_plan_test.cc \
		$(COCCL_CORE)/pipeline/coccl_pipeline_plan.cc $(SIZE_QUERY_STUB)
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(TEST_CXXFLAGS) $(COCCL_INC) -I$(COCCL_SRC) -o $@ $^

$(ALLREDUCE_RECIPE_TEST): coccl_allreduce_recipe_test.cc \
		$(COCCL_EXTENSIONS)/primitives/coccl_allreduce.cc \
		$(COCCL_EXTENSIONS)/primitives/coccl_hierarchical_reduction.cc
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(CXX) $(TEST_CXXFLAGS) $(COCCL_INC) -I$(COCCL_SRC) -o $@ $^

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
