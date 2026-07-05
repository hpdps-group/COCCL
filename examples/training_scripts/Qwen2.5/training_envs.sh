#!/bin/bash
CUDA_PATH=/path/to/cuda
CUDNN_PATH=/path/to/cudnn
GCC_PATH=/path/to/gcc
COCCL_PATH=/path/to/coccl
FRAMEWORK_PATH=/path/to/Pai-Megatron-Patch
MODEL_PATH=/path/to/Qwen2.5-3B-to-mcore
DATASET_PATH=/path/to/open-web-math_text_document
LOG_DIR=/path/to/logs/qwen2.5/3B
NCCL_LOG_DIR=/path/to/nccl_logs/qwen2.5/3B

QWEN_EXAMPLE_PATH=$FRAMEWORK_PATH/examples/qwen2_5
OUTPUT_PATH=$QWEN_EXAMPLE_PATH/output_mcore_qwen2.5_pretrain
