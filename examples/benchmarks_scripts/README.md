# Communication Benchmarks

Build COCCL and `tests/coccl-tests` first. The same script runs the single-node
and multi-node suites.

Single node:

```bash
CUDA_HOME=/path/to/cuda \
GPUS_PER_NODE=4 \
bash /path/to/COCCL/examples/benchmarks_scripts/communication_benchmarks.sh single
```

Multiple nodes:

```bash
CUDA_HOME=/path/to/cuda \
MPI_HOME=/path/to/mpi \
HOSTFILE=/path/to/hostfile \
NNODES=2 \
GPUS_PER_NODE=4 \
bash /path/to/COCCL/examples/benchmarks_scripts/communication_benchmarks.sh multi
```

The default suite measures native NCCL and COCCL with SDP4Bit and ZFP at
pipeline depths 1, 2, 4, and 8. Multi-node mode additionally runs
ReduceScatter TwoShot and AllReduce TripleShot. Override the matrix with:

```bash
COCCL_BENCH_COMPRESSORS="sdp4bit" \
COCCL_BENCH_DEPTHS="1 4" \
COCCL_BENCH_WARMUP=20 \
COCCL_BENCH_ITERATIONS=30 \
bash communication_benchmarks.sh single
```

`COCCL_ROOT` defaults to the repository containing the script. Generated
runtime configurations are written below `build/examples/benchmark-configs`.
