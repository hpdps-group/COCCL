# Qwen Training

The training example supports Qwen2.5 and Qwen3 through one launcher. It uses
the model runners from
[Pai-Megatron-Patch](https://github.com/alibaba/Pai-Megatron-Patch).

1. Activate the Python environment used by Pai-Megatron-Patch.
2. Edit `training_envs.sh` and fill in CUDA, framework, data, model, output, and
   log paths.
3. Set `COCCL_CONFIG_FILE` when using a topology other than the default
   2-node, 4-GPU, TP=2, PP=2, DP=2 example.

Run Qwen2.5 on each node:

```bash
# Node 0
MASTER_ADDR=10.0.0.1 MASTER_PORT=29501 \
bash train_qwen_coccl.sh qwen25 2 4 0

# Node 1
MASTER_ADDR=10.0.0.1 MASTER_PORT=29501 \
bash train_qwen_coccl.sh qwen25 2 4 1
```

Use `qwen3` as the first argument for Qwen3. Common training controls are
environment variables:

```bash
TP_SIZE=2 PP_SIZE=2 CP_SIZE=1 \
TRAIN_ITERS=100 DP_OVERLAP=off \
bash train_qwen_coccl.sh qwen3 2 4 0
```

The parallel sizes in the selected COCCL training TOML must match the job.
`NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA`, and other machine-specific NCCL settings
should be exported by the user before launching.
