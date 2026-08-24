#!/usr/bin/env bash

# Edit these paths once, or export the same variables before launching.
CUDA_HOME=${CUDA_HOME:-}
training_scripts_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PAI_MEGATRON_ROOT=${PAI_MEGATRON_ROOT:-$training_scripts_dir/Pai-Megatron-Patch}

QWEN25_DATA_PREFIX=${QWEN25_DATA_PREFIX:-}
QWEN25_MODEL_PATH=${QWEN25_MODEL_PATH:-}
QWEN3_DATA_PREFIX=${QWEN3_DATA_PREFIX:-}
QWEN3_MODEL_PATH=${QWEN3_MODEL_PATH:-}

TRAIN_OUTPUT_ROOT=${TRAIN_OUTPUT_ROOT:-}
TRAIN_LOG_ROOT=${TRAIN_LOG_ROOT:-}

# COCCL_ROOT is inferred from this repository when left empty. The bundled
# training config matches the default 2-node, 4-GPU, TP=2, PP=2 topology.
COCCL_ROOT=${COCCL_ROOT:-}
COCCL_CONFIG_FILE=${COCCL_CONFIG_FILE:-}
