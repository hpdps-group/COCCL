# Qwen3 Training With COCCL

English | [简体中文](README_zh-CN.md)

This example runs Qwen3-0.6B Base with COCCL on NVIDIA Megatron-LM. The bundled
submodule pins the tested Megatron release, `core_v0.15.3`.

## Prepare

Initialize Megatron-LM if it was not fetched with the repository:

```bash
git submodule update --init examples/training_scripts/Megatron-LM
```

Create and activate an environment that satisfies the
[Megatron-LM requirements](https://github.com/NVIDIA/Megatron-LM), then edit
[`training_envs.sh`](training_envs.sh). Set:

- `CUDA_HOME` and, when needed, `MEGATRON_ROOT`;
- `QWEN3_DATA_PREFIX`, the prefix of the preprocessed Megatron MMAP
  `.bin`/`.idx` pair;
- `QWEN3_TOKENIZER_DIR`, a Hugging Face Qwen3-0.6B tokenizer directory;
- `QWEN3_LOAD_DIR` when continuing from a Megatron Core checkpoint;
- `TRAIN_OUTPUT_ROOT` and `TRAIN_LOG_ROOT`.

Build COCCL before launching. The script loads
`$COCCL_ROOT/build/lib/libnccl.so.2` in place of NCCL. Confirm that PyTorch
uses a shared NCCL library:

```bash
TORCH_CUDA_LIB=$(python -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent / "lib/libtorch_cuda.so")')
ldd "$TORCH_CUDA_LIB" | grep libnccl
```

No output means that PyTorch embeds NCCL statically and cannot be replaced
through `LD_PRELOAD`.

## Configure COCCL

[`configs/training.toml`](configs/training.toml) enables SDP4Bit for DP
AllGather and ReduceScatter, enables autotuning, and leaves TP and PP on
native NCCL. Its default topology is two nodes with four GPUs per node:

```text
TP=2, PP=2, DP=2, sequence parallel enabled
```

Update `training.classifier` whenever the launch topology changes. This
launcher uses Megatron's distributed optimizer, so keep `dp_strategy =
"sdp"`. It enables sequence parallelism when `TP_SIZE > 1`.

## Launch

Run the same command on every node and change only `node_rank`:

```bash
# Node 0
bash train_qwen3_coccl.sh 2 4 0 10.0.0.1

# Node 1
bash train_qwen3_coccl.sh 2 4 1 10.0.0.1
```

The arguments are:

```text
train_qwen3_coccl.sh <nnodes> <gpus_per_node> [node_rank]
                     [master_addr] [master_port]
```

Common controls are environment variables:

```bash
TP_SIZE=2 PP_SIZE=2 TRAIN_ITERS=200 DP_OVERLAP=on \
  bash train_qwen3_coccl.sh 2 4 0 10.0.0.1
```

- `DP_OVERLAP=on|off` controls Megatron gradient-reduce and parameter-gather
  overlap.
- `MICRO_BATCH_SIZE`, `GLOBAL_BATCH_SIZE`, `SEQUENCE_LENGTH`, `EVAL_INTERVAL`,
  and `EVAL_ITERS` control the training run.
- `TRANSFORMER_IMPL=auto|local|transformer_engine` selects the transformer
  implementation.
- `COCCL_ROOT` and `COCCL_CONFIG_FILE` override the repository and config
  inferred by the launcher.

Checkpoint saving is intentionally disabled; `QWEN3_LOAD_DIR` is used only
to load the initial Megatron Core model checkpoint.

## Check Automatic Routing

The training code continues to call standard NCCL APIs. COCCL classifies each
communicator and applies the matching DP, TP, or PP policy. To inspect the
decision:

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COCCL_TUNING \
  bash train_qwen3_coccl.sh 2 4 0 10.0.0.1
```

Confirm that the log reports the expected `role=DP|TP|PP` and selected COCCL
algorithm before enabling compression for a new topology.
