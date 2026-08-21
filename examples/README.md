# COCCL Examples

The examples cover the normal user path: build COCCL, benchmark communication,
then run Qwen training. Every script can be launched from any directory.

## 1. Build

```bash
CUDA_HOME=/path/to/cuda \
MPI_HOME=/path/to/mpi \
CMAKE_HOME=/path/to/cmake-prefix \
CUDA_ARCH=80 \
bash /path/to/COCCL/examples/build_scripts/build.sh
```

This builds COCCL, bundled plugins, the configuration checker, and
MPI-enabled tests. See [build_scripts](build_scripts).

## 2. Benchmark Communication

Single node:

```bash
CUDA_HOME=/path/to/cuda MPI_HOME=/path/to/mpi GPUS_PER_NODE=4 \
bash /path/to/COCCL/examples/benchmarks_scripts/communication_benchmarks.sh single
```

Multiple nodes:

```bash
CUDA_HOME=/path/to/cuda MPI_HOME=/path/to/mpi \
HOSTFILE=/path/to/hostfile NNODES=2 GPUS_PER_NODE=4 \
bash /path/to/COCCL/examples/benchmarks_scripts/communication_benchmarks.sh multi
```

See [benchmarks_scripts](benchmarks_scripts) for the default matrix and
overrides.

## 3. Train Qwen

Initialize the framework submodule, then edit `training_envs.sh`:

```bash
git submodule update --init --recursive \
  examples/training_scripts/Pai-Megatron-Patch
```

Launch the same command on each node and change only the final node rank:

```bash
MASTER_ADDR=10.0.0.1 MASTER_PORT=29501 \
bash /path/to/COCCL/examples/training_scripts/train_qwen_coccl.sh qwen3 2 4 0
```

The launcher also accepts `qwen25`. See
[training_scripts](training_scripts) for required paths, topology settings,
and automatic DP/TP/PP routing.
