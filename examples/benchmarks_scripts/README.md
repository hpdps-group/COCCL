# Communication Benchmarks

English | [简体中文](README_zh-CN.md)

Build COCCL and `tests/coccl-tests` before running this script. Both modes
require `MPI_HOME`; single-node mode does not invoke `mpirun`.

## Run

Single node:

```bash
CUDA_HOME=/path/to/cuda MPI_HOME=/path/to/mpi GPUS_PER_NODE=4 \
bash communication_benchmarks.sh single
```

Multiple nodes:

```bash
CUDA_HOME=/path/to/cuda MPI_HOME=/path/to/mpi \
HOSTFILE=/path/to/hostfile NNODES=2 GPUS_PER_NODE=4 \
bash communication_benchmarks.sh multi
```

The default matrix compares native NCCL with SDP4Bit and ZFP at depths 1, 2,
4, and 8. Multi-node mode also runs ReduceScatter TwoShot and AllReduce
TripleShot.

## Change The Matrix

Set only the values you need to override:

```bash
COCCL_BENCH_COMPRESSORS="sdp4bit" \
COCCL_BENCH_DEPTHS="1 4" \
COCCL_BENCH_WARMUP=20 \
COCCL_BENCH_ITERATIONS=30 \
bash communication_benchmarks.sh single
```

For a smoke run:

```bash
COCCL_BENCH_COMPRESSORS="sdp4bit" \
COCCL_BENCH_DEPTHS="1" \
COCCL_BENCH_BEGIN=1MB \
COCCL_BENCH_END=4MB \
COCCL_BENCH_ALLREDUCE_BEGIN=1MB \
COCCL_BENCH_ONESHOT_BEGIN=1MB \
COCCL_BENCH_ONESHOT_END=4MB \
COCCL_BENCH_WARMUP=1 \
COCCL_BENCH_ITERATIONS=2 \
bash communication_benchmarks.sh single
```

Multi-node mode forwards `NCCL_SOCKET_IFNAME`, `NCCL_IB_DISABLE`,
`NCCL_IB_HCA`, and `NCCL_LOCAL_REGISTER` when set.

## Files

- `configs/`: release TOML examples for SDP4Bit, ZFP, and dietGPU.
- `build/examples/benchmark-configs/`: generated configs with the requested
  pipeline depth.

`COCCL_ROOT` defaults to the repository containing this script. Generated
files never modify the checked-in examples.
