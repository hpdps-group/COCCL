# Qwen Training With COCCL

English | [简体中文](README_zh-CN.md)

This example runs Qwen2.5 or Qwen3 with COCCL through
[Pai-Megatron-Patch](Pai-Megatron-Patch). The framework is pinned as a Git
submodule.

## Before You Run

Initialize the framework if the repository was cloned without submodules:

```bash
git submodule update --init --recursive \
  examples/training_scripts/Pai-Megatron-Patch
```

Activate the Python environment used by Pai-Megatron-Patch, then edit
`training_envs.sh`. Set:

- `CUDA_HOME`.
- The data prefix and model path for Qwen2.5 or Qwen3.
- `TRAIN_OUTPUT_ROOT` and `TRAIN_LOG_ROOT`.
- `COCCL_ROOT`, `COCCL_CONFIG_FILE`, or `PAI_MEGATRON_ROOT` only when the
  defaults should be overridden.

COCCL is loaded in place of NCCL. Confirm that PyTorch uses a shared NCCL
library:

```bash
TORCH_CUDA_LIB=$(python -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent / "lib/libtorch_cuda.so")')
ldd "$TORCH_CUDA_LIB" | grep libnccl
```

If this prints nothing, PyTorch embeds NCCL statically and `LD_PRELOAD` cannot
replace it. Use a dynamically linked PyTorch build. The launcher preloads the
COCCL `libnccl.so.2`; compressor libraries come from the TOML file and do not
belong in `LD_LIBRARY_PATH`.

## Configure COCCL

The default [`configs/training.toml`](configs/training.toml) loads SDP4Bit for
the validated DP policies. TP and PP remain on native NCCL.

Keep these classifier settings consistent with the training command:

- `data_parallel_size`, `tensor_parallel_size`, and
  `pipeline_parallel_size` must describe the job topology.
- `dp_strategy` must match DDP, the sharded distributed optimizer (`sdp`), or
  FSDP.
- `sequence_parallel` must match the TP communication pattern.

The launcher computes `DP = world_size / (TP * PP * CP)`. Put that result in
`data_parallel_size`. Keep machine-specific settings such as
`NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA`, and `CUDA_VISIBLE_DEVICES` in the job
environment.

## Launch

Run the same command on every node, changing only the node rank. For two nodes
with four GPUs each:

```bash
# Node 0
MASTER_ADDR=10.0.0.1 MASTER_PORT=29501 \
bash train_qwen_coccl.sh qwen25 2 4 0

# Node 1
MASTER_ADDR=10.0.0.1 MASTER_PORT=29501 \
bash train_qwen_coccl.sh qwen25 2 4 1
```

Use `qwen3` for Qwen3. Override training controls through the environment:

```bash
TP_SIZE=2 PP_SIZE=2 CP_SIZE=1 \
TRAIN_ITERS=100 DP_OVERLAP=off \
bash train_qwen_coccl.sh qwen3 2 4 0
```

`DP_OVERLAP` accepts `on` or `off`. The launcher also accepts
`MICRO_BATCH_SIZE`, `GLOBAL_BATCH_SIZE`, `SEQUENCE_LENGTH`, `WARMUP_ITERS`,
`EVAL_INTERVAL`, and `EVAL_ITERS`.

Checkpoints are off by default. Set `SAVE_CHECKPOINTS=on` and optionally
`SAVE_INTERVAL` to enable them.

## Check Automatic Routing

Training code continues to call standard NCCL APIs. COCCL classifies each
communicator and then selects its DP, TP, or PP policy:

- Unique configured DP, TP, or PP sizes are classified immediately.
- Ambiguous DP and TP communicators are identified from repeated collective
  patterns, topology, DP strategy, and sequence parallelism.
- Messages below 1 MiB are not classifier evidence.
- Unknown communicators stay on native NCCL.
- PP direction is inferred from the local rank and peer.

Enable routing logs when validating a new topology:

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COCCL_TUNING \
bash train_qwen_coccl.sh qwen3 2 4 0
```

Look for `role=DP|TP|PP` and `COCCL route` before enabling a compressor for
that role.
