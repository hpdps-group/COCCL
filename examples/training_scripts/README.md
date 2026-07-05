# Training Examples

This directory provides end-to-end training examples for evaluating COCCL with Megatron-LM.

## 1. Activate the Training Environment

Load the required training environment before launching training. This example uses conda:

```bash
source activate python3.10 # conda
```

## 2. Configure Paths

Edit `training_envs.sh` in the target example directory and set the paths to CUDA, COCCL, the training framework, model checkpoints, and datasets.

Each example keeps its own path configuration:

| Example | Env file | Training framework path | Upstream repository |
| --- | --- | --- | --- |
| GPT-2 | `GPT2/training_envs.sh` | `Megatron_PATH=/path/to/Megatron-LM` | [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) |
| LingV2 | `LingV2/training_envs.sh` | `FRAMEWORK_PATH=/path/to/LingV2` | [inclusionAI/Ling-V2](https://github.com/inclusionAI/Ling-V2) |
| Qwen2.5 | `Qwen2.5/training_envs.sh` | `FRAMEWORK_PATH=/path/to/Pai-Megatron-Patch` | [alibaba/Pai-Megatron-Patch](https://github.com/alibaba/Pai-Megatron-Patch) |

For example:

```bash
CUDA_PATH=/path/to/cuda
COCCL_PATH=/path/to/COCCL_PPoPP_AE
FRAMEWORK_PATH=/path/to/training/framework
MODEL_PATH=/path/to/model
DATASET_PATH=/path/to/dataset
```

## 3. Launch Training

Go to the target example directory and launch the selected script with the number of nodes and GPUs per node. For example:

```bash
cd GPT2
bash train_gpt_coccl.sh <nnodes> <gpus_per_node>

cd LingV2
bash train_ling_coccl.sh <nnodes> <gpus_per_node>

cd Qwen2.5
bash train_qwen_coccl.sh <nnodes> <gpus_per_node>
```
