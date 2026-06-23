# Qwen2.5 Training with TACO (COCCL + Megatron)

This directory contains training scripts for **Qwen2.5 models** using **Megatron-LM** with **TACO communication compression**.

---

## 📦 Environment Setup

```bash
export CUDA_HOME=/usr/local/cuda
export PROJECT_ROOT=/path/to/TACO-COCCL

export COCCL_PATH=$PROJECT_ROOT/coccl-acc
export MEGATRON_PATH=$PROJECT_ROOT/Megatron

export DATA_PATH=/path/to/dataset
export CKPT_PATH=/path/to/qwen-ckpt
export LOG_PATH=$PROJECT_ROOT/logs
```

---

## 🚀 Run Training

```bash
bash run_qwen_taco.sh
```

---

## 📂 Dataset Format

```bash
DATA_PATH/
├── *.bin
├── *.idx
```

---

## 📝 Logs

```bash
$LOG_PATH/train.log
$LOG_PATH/nccl.log
```

---

## 📜 Script Examples

### Baseline (No Compression)

```bash
coccl_scripts/qwen2.5/0.5B/Qwen2.5_0.5B_Baseline.sh
coccl_scripts/qwen2.5/7B/Qwen2.5_7B_Baseline.sh
```

### TACO Enabled

```bash
coccl_scripts/qwen2.5/0.5B/Qwen2.5_0.5B_Taco.sh
coccl_scripts/qwen2.5/7B/Qwen2.5_7B_Taco.sh
```

---

## ⚙️ Training Arguments

```bash
ENV=$1                          # dsw (single-node) / dlc (multi-node)
MODEL_SIZE=$2                   # 0.5B / 1.5B / 3B / 7B / 14B / 32B / 72B
BATCH_SIZE=$3                   # Micro batch size
GLOBAL_BATCH_SIZE=$4            # Global batch size
LR=$5
MIN_LR=$6
SEQ_LEN=$7
PAD_LEN=$8
PR=${9}                         # fp16 / bf16 / fp8
TP=${10}                        # Tensor parallel
PP=${11}                        # Pipeline parallel
CP=${12}                        # Context parallel
SP=${13}                        # Sequence parallel (true/false)
DO=${14}                        # ZeRO-1 optimizer
FL=${15}                        # Flash Attention
SFT=${16}                       # Fine-tuning
AC=${17}                        # Activation checkpoint
OPTIMIZER_OFFLOAD=${18}         # Offload: false/static/auto
SAVE_INTERVAL=${19}
DATASET_PATH=${20}
VALID_DATASET_PATH=${21}
PRETRAIN_CHECKPOINT_PATH=${22}
TRAIN_TOKENS_OR_ITERS=${23}
WARMUP_TOKENS_OR_ITERS=${24}
OUTPUT_BASEPATH=${25}
```

---

## 📎 Notes

* Ensure dataset is preprocessed into Megatron format
* Multi-node training requires correct NCCL/MPI setup
* Recommend validating on small configs before scaling

---
