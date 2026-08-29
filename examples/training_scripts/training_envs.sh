#!/usr/bin/env bash

# Set these paths once, or export the same variables before launching.
training_scripts_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

CUDA_HOME=${CUDA_HOME:-}
MEGATRON_ROOT=${MEGATRON_ROOT:-$training_scripts_dir/Megatron-LM}

# QWEN3_DATA_PREFIX is the prefix shared by the Megatron MMAP .bin/.idx files.
QWEN3_DATA_PREFIX=${QWEN3_DATA_PREFIX:-}
QWEN3_TOKENIZER_DIR=${QWEN3_TOKENIZER_DIR:-}
QWEN3_DATA_CACHE_DIR=${QWEN3_DATA_CACHE_DIR:-}
# Optional Megatron Core checkpoint used as the initial model.
QWEN3_LOAD_DIR=${QWEN3_LOAD_DIR:-}

TRAIN_OUTPUT_ROOT=${TRAIN_OUTPUT_ROOT:-}
TRAIN_LOG_ROOT=${TRAIN_LOG_ROOT:-}

# COCCL_ROOT is inferred from this repository when left empty. The bundled
# config describes two 4-GPU nodes with TP=2, PP=2, DP=2.
COCCL_ROOT=${COCCL_ROOT:-}
COCCL_CONFIG_FILE=${COCCL_CONFIG_FILE:-}
