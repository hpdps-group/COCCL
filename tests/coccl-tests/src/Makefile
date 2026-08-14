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

.PHONY: build clean pipeline-layout-test group-deferred-test group-batch-test \
	training-classifier-test config-reader-test compressor-sdk-test \
	compressor-size-test autotune-selector-test pipeline-stage-test \
	pipeline-plan-test frame-exchange-test dietgpu-codec-test

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
COCCL_ROOT := ../../../src/coccl-extend
COCCL_INC := -I$(COCCL_ROOT)/include -I$(COCCL_ROOT)
SRC_FILES := $(wildcard *.cu)
OBJ_FILES := $(SRC_FILES:%.cu=${DST_DIR}/%.o)
BIN_FILES_LIST := all_reduce all_gather broadcast reduce_scatter reduce alltoall scatter gather sendrecv hypercube \
alltoall_comp alltoall_p2p all_gather_comp \
all_reduce_comp_oneshot all_reduce_comp_twoshot all_reduce_comp_tripleshot \
reduce_scatter_comp_oneshot reduce_scatter_comp_twoshot sendrecv_comp
BIN_FILES := $(BIN_FILES_LIST:%=${DST_DIR}/%_perf)

build: ${BIN_FILES}

COCCL_LAYOUT_TEST := $(DST_DIR)/coccl_pipeline_layout_test
COCCL_LAYOUT_SRC := $(COCCL_ROOT)/core/device/pipeline/coccl_pipeline_layout.cu
COCCL_LAYOUT_INC := $(COCCL_ROOT)/include

pipeline-layout-test: $(COCCL_LAYOUT_TEST)

COCCL_GROUP_TEST := $(DST_DIR)/coccl_group_deferred_test
COCCL_GROUP_BATCH_TEST := $(DST_DIR)/coccl_group_batch_test
COCCL_TRAINING_TEST := $(DST_DIR)/coccl_training_classifier_test
COCCL_CONFIG_READER_TEST := $(DST_DIR)/coccl_config_reader_test
COCCL_COMPRESSOR_SDK_TEST := $(DST_DIR)/coccl_compressor_sdk_test
COCCL_COMPRESSOR_SIZE_TEST := $(DST_DIR)/coccl_compressor_size_test
COCCL_AUTOTUNE_SELECTOR_TEST := $(DST_DIR)/coccl_autotune_selector_test
COCCL_PIPELINE_STAGE_TEST := $(DST_DIR)/coccl_pipeline_stage_test
COCCL_PIPELINE_PLAN_TEST := $(DST_DIR)/coccl_pipeline_plan_test
COCCL_FRAME_EXCHANGE_TEST := $(DST_DIR)/coccl_frame_exchange_test
COCCL_DIETGPU_CODEC_TEST := $(DST_DIR)/coccl_dietgpu_codec_test
COCCL_PIPELINE_STAGE_SRC := \
	$(COCCL_ROOT)/core/pipeline/coccl_pipeline_stage.cc
COCCL_PIPELINE_PLAN_SRC := \
	$(COCCL_ROOT)/core/pipeline/coccl_pipeline_plan.cc
COCCL_PIPELINE_INTERNAL_HEADER := \
	$(COCCL_ROOT)/core/pipeline/coccl_pipeline_internal.h
COCCL_AUTOTUNE_COST_MODEL_SRC := \
	$(COCCL_ROOT)/core/tuning/coccl_autotune_cost_model.cc
COCCL_AUTOTUNE_INTERNAL_HEADER := \
	$(COCCL_ROOT)/core/tuning/coccl_autotune_internal.h
COCCL_COMPRESSOR_SDK_HEADERS := \
	$(COCCL_ROOT)/include/compressor_plugin/coccl_compressor_plugin.h \
	$(COCCL_ROOT)/include/compressor_plugin/detail/coccl_compressor_abi.h

COCCL_TRAINING_SRC := \
	$(COCCL_ROOT)/core/config/coccl_config.cc \
	$(COCCL_ROOT)/core/config/coccl_config_format.cc \
	$(COCCL_ROOT)/core/config/coccl_config_log.cc \
	$(COCCL_ROOT)/core/training/coccl_training_assist.cc \
	$(COCCL_ROOT)/core/training/coccl_training_classifier.cc \
	$(COCCL_ROOT)/core/runtime/coccl_operation.cc \
	$(COCCL_ROOT)/core/runtime/coccl_comm.cc \
	$(COCCL_ROOT)/core/runtime/coccl_runtime.cc

group-deferred-test: $(COCCL_GROUP_TEST)
group-batch-test: $(COCCL_GROUP_BATCH_TEST)
training-classifier-test: $(COCCL_TRAINING_TEST)
config-reader-test: $(COCCL_CONFIG_READER_TEST)
compressor-sdk-test: $(COCCL_COMPRESSOR_SDK_TEST)
compressor-size-test: $(COCCL_COMPRESSOR_SIZE_TEST)
autotune-selector-test: $(COCCL_AUTOTUNE_SELECTOR_TEST)
pipeline-stage-test: $(COCCL_PIPELINE_STAGE_TEST)
pipeline-plan-test: $(COCCL_PIPELINE_PLAN_TEST)
frame-exchange-test: $(COCCL_FRAME_EXCHANGE_TEST)
dietgpu-codec-test: $(COCCL_DIETGPU_CODEC_TEST)

$(COCCL_GROUP_TEST): coccl_group_deferred_test.cc
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(CXX) $(CXXFLAGS) -I$(CUDA_INC) -I$(NCCL_HOME)/include -o $@ $< \
		-L$(NCCL_HOME)/lib -lnccl $(LDFLAGS) -pthread

$(COCCL_GROUP_BATCH_TEST): coccl_group_batch_test.cc \
		$(COCCL_ROOT)/core/runtime/coccl_group.cc
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(CXX) $(filter-out -std=c++11,$(CXXFLAGS)) -std=c++17 \
		-ffunction-sections -fdata-sections -Wl,--gc-sections \
		-I$(CUDA_INC) -I$(NCCL_HOME)/include -I../../../src/include \
		$(COCCL_INC) -o $@ $^

$(COCCL_TRAINING_TEST): coccl_training_classifier_test.cc $(COCCL_TRAINING_SRC)
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(CXX) $(filter-out -std=c++11,$(CXXFLAGS)) -std=c++17 \
		-ffunction-sections -fdata-sections -Wl,--gc-sections \
		-I$(CUDA_INC) -I$(NCCL_HOME)/include -I../../../src/include \
		$(COCCL_INC) \
		-DCOCCL_TEST_CONFIG_FILE=\"$(abspath $(COCCL_ROOT)/extensions/configs/training.toml)\" \
		-o $@ $^ -pthread

$(COCCL_CONFIG_READER_TEST): coccl_config_reader_test.cc \
	$(COCCL_COMPRESSOR_SDK_HEADERS)
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(CXX) $(filter-out -std=c++11,$(CXXFLAGS)) -std=c++17 \
		-I$(CUDA_INC) -I$(NCCL_HOME)/include \
		$(COCCL_INC) -o $@ $<

$(COCCL_COMPRESSOR_SDK_TEST): coccl_compressor_sdk_test.cc \
	$(COCCL_COMPRESSOR_SDK_HEADERS)
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(CXX) $(filter-out -std=c++11,$(CXXFLAGS)) -std=c++17 \
		-I$(CUDA_INC) -I$(NCCL_HOME)/include \
		$(COCCL_INC) -o $@ $<

$(COCCL_COMPRESSOR_SIZE_TEST): coccl_compressor_size_test.cc \
	$(COCCL_COMPRESSOR_SDK_HEADERS)
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(CXX) $(filter-out -std=c++11,$(CXXFLAGS)) -std=c++17 \
		-I$(CUDA_INC) -I$(NCCL_HOME)/include \
		$(COCCL_INC) -o $@ $< -ldl

$(COCCL_AUTOTUNE_SELECTOR_TEST): coccl_autotune_selector_test.cc \
	$(COCCL_AUTOTUNE_COST_MODEL_SRC) $(COCCL_AUTOTUNE_INTERNAL_HEADER)
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(CXX) $(filter-out -std=c++11,$(CXXFLAGS)) -std=c++17 \
		-I$(CUDA_INC) -I$(NCCL_HOME)/include -I../../../src/include \
		$(COCCL_INC) -o $@ $< \
		$(COCCL_AUTOTUNE_COST_MODEL_SRC)

$(COCCL_PIPELINE_STAGE_TEST): coccl_pipeline_stage_test.cc \
		$(COCCL_PIPELINE_STAGE_SRC)
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(CXX) $(filter-out -std=c++11,$(CXXFLAGS)) -std=c++17 \
		-ffunction-sections -fdata-sections -Wl,--gc-sections \
		-I$(CUDA_INC) -I$(NCCL_HOME)/include -I../../../src/include \
		$(COCCL_INC) -o $@ $^

$(COCCL_PIPELINE_PLAN_TEST): coccl_pipeline_plan_test.cc \
		$(COCCL_PIPELINE_PLAN_SRC) $(COCCL_PIPELINE_INTERNAL_HEADER)
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(CXX) $(filter-out -std=c++11,$(CXXFLAGS)) -std=c++17 \
		-ffunction-sections -fdata-sections -Wl,--gc-sections \
		-I$(CUDA_INC) -I$(NCCL_HOME)/include -I../../../src/include \
		$(COCCL_INC) \
		-o $@ coccl_pipeline_plan_test.cc $(COCCL_PIPELINE_PLAN_SRC)

$(COCCL_FRAME_EXCHANGE_TEST): coccl_frame_exchange_test.cc \
		$(COCCL_ROOT)/core/pipeline/coccl_frame_exchange.cc
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(CXX) $(filter-out -std=c++11,$(CXXFLAGS)) -std=c++17 \
		-I$(CUDA_INC) -I$(NCCL_HOME)/include -I../../../src/include \
		$(COCCL_INC) -o $@ $^

$(COCCL_DIETGPU_CODEC_TEST): coccl_dietgpu_codec_test.cu \
		$(COCCL_COMPRESSOR_SDK_HEADERS)
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) $(filter-out -std=c++11,$(NVCUFLAGS)) -std=c++17 \
		-I$(NCCL_HOME)/include $(COCCL_INC) \
		-o $@ $< -ldl -lcudart

$(COCCL_LAYOUT_TEST): coccl_pipeline_layout_test.cu $(COCCL_LAYOUT_SRC)
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $(COCCL_INC) \
		-I$(NCCL_HOME)/include -I../../../src/include \
		$^ -L$(NCCL_HOME)/lib -lnccl -lcudart -lrt

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
