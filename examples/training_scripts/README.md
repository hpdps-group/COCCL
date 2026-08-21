# Qwen Training With COCCL

This directory contains one launcher for Qwen2.5 and Qwen3. The
[Pai-Megatron-Patch](Pai-Megatron-Patch) directory is a Git submodule, so
GitHub renders it as a link to the exact upstream revision used by this
repository.

## Prepare Pai-Megatron-Patch

If COCCL was cloned without `--recurse-submodules`, initialize the framework:

```bash
git submodule update --init --recursive \
  examples/training_scripts/Pai-Megatron-Patch
```

The launcher uses this checkout by default. Set `PAI_MEGATRON_ROOT` in
`training_envs.sh` or the environment only when using another checkout.
Install and activate the Python environment required by Pai-Megatron-Patch
before launching training.

## Verify Dynamic NCCL Loading

COCCL replaces NCCL at the shared-library boundary; no framework collective
source changes are required. Confirm that PyTorch links NCCL dynamically:

```bash
TORCH_CUDA_LIB=$(python -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent / "lib/libtorch_cuda.so")')
ldd "$TORCH_CUDA_LIB" | grep libnccl
```

If no NCCL dependency is shown, that PyTorch build embeds NCCL statically and
cannot be redirected with `LD_LIBRARY_PATH` or `LD_PRELOAD`. Use a dynamically
linked PyTorch build. Do not overwrite the system NCCL installation.

`train_qwen_coccl.sh` prepends `build/lib/libnccl.so.2` through `LD_PRELOAD`.
Compressor DSOs are loaded from `compressor_plugins.library_path` in the TOML
configuration, so the plugin directory does not belong in `LD_LIBRARY_PATH`.
The startup output prints the selected COCCL config. To inspect the library
before a full run, use the same environment and repeat the `ldd` command;
`libnccl.so.2` should resolve to the COCCL build directory.

## Configure Paths And Topology

Edit `training_envs.sh` once or export the same variables before launch:

| Variable | Required value |
| --- | --- |
| `CUDA_HOME` | CUDA toolkit used by the job. |
| `QWEN25_DATA_PREFIX`, `QWEN3_DATA_PREFIX` | Indexed training dataset prefix for each model. |
| `QWEN25_MODEL_PATH`, `QWEN3_MODEL_PATH` | Input model/checkpoint path. |
| `TRAIN_OUTPUT_ROOT` | Training output and checkpoint root. |
| `TRAIN_LOG_ROOT` | Per-model, per-node log root. |
| `COCCL_ROOT` | Optional external COCCL checkout; inferred from this repository by default. |
| `COCCL_CONFIG_FILE` | Optional training TOML; defaults to `examples/training_scripts/configs/training.toml`. |

The job computes `DP = world_size / (TP * PP * CP)`. The selected TOML must
declare matching values in `training.classifier`. `dp_strategy` must match the
framework optimizer (`ddp`, sharded distributed optimizer/`sdp`, or `fsdp`),
and `sequence_parallel` must match the TP communication pattern.

Machine-specific NCCL settings such as `NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA`,
and `CUDA_VISIBLE_DEVICES` remain the user's responsibility.

## Launch

Run the same command on every node and change only the node rank. For a
2-node, 4-GPU-per-node Qwen2.5 job:

```bash
# Node 0
MASTER_ADDR=10.0.0.1 MASTER_PORT=29501 \
bash train_qwen_coccl.sh qwen25 2 4 0

# Node 1
MASTER_ADDR=10.0.0.1 MASTER_PORT=29501 \
bash train_qwen_coccl.sh qwen25 2 4 1
```

Use `qwen3` as the first argument for Qwen3. Common controls are environment
variables:

```bash
TP_SIZE=2 PP_SIZE=2 CP_SIZE=1 \
TRAIN_ITERS=100 DP_OVERLAP=off \
bash train_qwen_coccl.sh qwen3 2 4 0
```

`DP_OVERLAP` accepts `on` or `off`. Other useful controls include
`MICRO_BATCH_SIZE`, `GLOBAL_BATCH_SIZE`, `SEQUENCE_LENGTH`, `WARMUP_ITERS`,
`EVAL_INTERVAL`, and `EVAL_ITERS`. Checkpoints are disabled by default, even
though the upstream Pai launchers add `--save`. Set `SAVE_CHECKPOINTS=on` and
optionally `SAVE_INTERVAL` to enable checkpoint output.

## Automatic DP, TP, And PP Detection

Training mode observes standard NCCL calls, so integration does not require
the framework to label a communicator. COCCL uses the configured communicator
sizes first:

- A unique DP or TP size classifies collective communicators immediately.
- A matching PP size classifies Send/Recv communicators as PP.
- When DP and TP sizes are ambiguous, COCCL observes repeated training
  iterations and combines AllReduce, ReduceScatter, AllGather, topology,
  ordering, DP strategy, and sequence-parallel patterns.
- Ambiguous messages below 1 MiB are not used as classifier evidence.
- Calls remain native NCCL until classification is committed. Unknown
  communicators continue to use native NCCL.

Once a role is known, COCCL routes collectives through `training.dp.*` or
`training.tp.*`. PP Send/Recv uses `training.pp.sendrecv.forward` or
`training.pp.sendrecv.backward`; direction is inferred from local rank and
peer.

The bundled [`configs/training.toml`](configs/training.toml) loads only
SDP4Bit and enables the SDP4Bit DP policies used by the validated Qwen runs.
It does not load ZFP. TP and PP remain on native NCCL. Add role-specific
policies only after validating the selected codec for training.

Use tuning logs to verify the detected role and selected policy:

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COCCL_TUNING \
bash train_qwen_coccl.sh qwen3 2 4 0
```

Look for `COCCL training ... role=DP|TP|PP` and `COCCL route ...` lines. The
declared sizes and observed framework communication must agree before treating
the selected compressor as valid.
