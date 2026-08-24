# 通信性能测试

[English](README.md) | 简体中文

运行脚本前，请先构建 COCCL 和 `tests/coccl-tests`。两种模式都需要设置 `MPI_HOME`；单节点模式不会调用 `mpirun`。

## 运行

单节点：

```bash
CUDA_HOME=/path/to/cuda MPI_HOME=/path/to/mpi GPUS_PER_NODE=4 \
bash communication_benchmarks.sh single
```

多节点：

```bash
CUDA_HOME=/path/to/cuda MPI_HOME=/path/to/mpi \
HOSTFILE=/path/to/hostfile NNODES=2 GPUS_PER_NODE=4 \
bash communication_benchmarks.sh multi
```

默认测试矩阵会在 pipeline depth 为 1、2、4 和 8 时对比原生 NCCL、SDP4Bit 和 ZFP。多节点模式还会测试 ReduceScatter TwoShot 和 AllReduce TripleShot。

## 调整测试矩阵

只需设置需要覆盖的参数：

```bash
COCCL_BENCH_COMPRESSORS="sdp4bit" \
COCCL_BENCH_DEPTHS="1 4" \
COCCL_BENCH_WARMUP=20 \
COCCL_BENCH_ITERATIONS=30 \
bash communication_benchmarks.sh single
```

快速测试示例：

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

多节点模式会在已设置时转发 `NCCL_SOCKET_IFNAME`、`NCCL_IB_DISABLE`、`NCCL_IB_HCA` 和 `NCCL_LOCAL_REGISTER`。

## 文件

- `configs/`：SDP4Bit、ZFP 和 dietGPU 的发布版 TOML 示例。
- `build/examples/benchmark-configs/`：按照指定 pipeline depth 生成的配置。

`COCCL_ROOT` 默认指向脚本所在的仓库。生成的文件不会修改仓库中已有的示例。
