# COCCL Examples

The examples provide three user-facing workflows. Every script resolves the
COCCL repository from its own location, so it can be launched from any working
directory.

## 1. Build

```bash
CUDA_HOME=/path/to/cuda \
MPI_HOME=/path/to/mpi \
CUDA_ARCH=80 \
bash /path/to/COCCL/examples/build_scripts/build.sh
```

This builds COCCL, bundled compressor plugins, and MPI-enabled communication
tests. See [build_scripts/README.md](build_scripts/README.md) for optional
build controls.

## 2. Communication

Single node:

```bash
CUDA_HOME=/path/to/cuda GPUS_PER_NODE=4 \
bash /path/to/COCCL/examples/benchmarks_scripts/communication_benchmarks.sh single
```

Multiple nodes:

```bash
CUDA_HOME=/path/to/cuda \
MPI_HOME=/path/to/mpi \
HOSTFILE=/path/to/hostfile \
NNODES=2 GPUS_PER_NODE=4 \
bash /path/to/COCCL/examples/benchmarks_scripts/communication_benchmarks.sh multi
```

See [benchmarks_scripts/README.md](benchmarks_scripts/README.md) for the test
matrix and tuning variables.

## 3. Qwen Training

Edit [training_scripts/training_envs.sh](training_scripts/training_envs.sh),
activate the Pai-Megatron-Patch Python environment, and launch the same script
on every node:

```bash
MASTER_ADDR=10.0.0.1 MASTER_PORT=29501 \
bash /path/to/COCCL/examples/training_scripts/train_qwen_coccl.sh qwen25 2 4 0
```

The first argument is `qwen25` or `qwen3`; the final argument is the node
rank. See [training_scripts/README.md](training_scripts/README.md) for the
multi-node example and topology requirements.
