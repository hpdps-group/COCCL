# NCCL 测试

[English](README.md) | 简体中文

这些测试用于检查 [NCCL](http://github.com/nvidia/nccl) 操作的性能和正确性。

## 构建

直接运行 `make` 即可构建测试。

如果 CUDA 未安装在 `/usr/local/cuda`，可以指定 `CUDA_HOME`。同样，如果 NCCL 未安装在 `/usr`，可以指定 `NCCL_HOME`。

```shell
$ make CUDA_HOME=<path to cuda> NCCL_HOME=<path to nccl>
```

NCCL tests 依赖 MPI 才能运行多进程和多节点测试。要启用 MPI 支持，需要设置 `MPI=1`，并将 `MPI_HOME` 指向 MPI 安装路径。

```shell
$ make MPI=1 MPI_HOME=<path to mpi> CUDA_HOME=<path to cuda> NCCL_HOME=<path to nccl>
```

## 用法

NCCL tests 可以运行在多进程、多线程以及每个线程使用多个 CUDA 设备的场景。进程数量由 MPI 管理，因此不作为参数传给测试程序。总 rank 数（即 CUDA 设备数）等于（进程数）×（每个进程的线程数）×（每个线程的 GPU 数量）。

### 快速示例

在单节点的 8 张 GPU 上运行（`-g 8`），消息大小从 8 Bytes 扫描到 128 MBytes：

```shell
$ ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 8
```

在 8 个节点上各运行 8 个 MPI 进程，共使用 64 张 GPU：
（此场景要求 nccl-tests 二进制文件在构建时设置 `MPI=1`。）

```shell
$ mpirun -np 64 -N 8 ./build/all_reduce_perf -b 8 -e 8G -f 2 -g 1
```

### 性能

各项数值的含义，特别是 `busbw` 列，见[性能说明](doc/PERFORMANCE.md)。

### 参数

所有测试支持同一组参数：

* GPU 数量
  * `-t,--nthreads <num threads>`：每个进程的线程数，默认为 1。
  * `-g,--ngpus <GPUs per thread>`：每个线程的 GPU 数量，默认为 1。
* 扫描大小
  * `-b,--minbytes <min size in bytes>`：起始消息大小，默认为 32M。
  * `-e,--maxbytes <max size in bytes>`：结束消息大小，默认为 32M。
  * 递增方式可以使用固定步长或乘法因子，两者只能选择一个。
    * `-i,--stepbytes <increment size>`：相邻消息大小之间的固定增量，默认为 1M。
    * `-f,--stepfactor <increment factor>`：消息大小的乘法因子，默认禁用。
* NCCL 操作参数
  * `-o,--op <sum/prod/min/max/avg/all>`：指定归约操作。仅用于 AllReduce、Reduce 或 ReduceScatter 等归约通信，默认为 Sum。
  * `-d,--datatype <nccltype/all>`：指定 datatype，默认为 Float。
  * `-r,--root <root/all>`：指定 root，仅用于 broadcast 或 reduce 等有 root 的操作，默认为 0。
* 性能
  * `-n,--iters <iteration count>`：迭代次数，默认为 20。
  * `-w,--warmup_iters <warmup iteration count>`：不计时的预热迭代次数，默认为 5。
  * `-m,--agg_iters <aggregation count>`：每次迭代聚合的操作数，默认为 1。
  * `-N,--run_cycles <cycle count>`：运行并打印每个 cycle，默认为 1；设为 0 时无限运行。
  * `-a,--average <0/1/2/3>`：报告所有 rank 的汇总性能（仅 `MPI=1`），`0=Rank0,1=Avg,2=Min,3=Max`，默认为 1。
* 测试操作
  * `-p,--parallel_init <0/1>`：使用线程并行初始化 NCCL，默认为 0。
  * `-c,--check <check iteration count>`：执行指定次数并检查每次结果的正确性。GPU 数量较多时可能很慢，默认为 1。
  * `-z,--blocking <0/1>`：使 NCCL collective 阻塞，即每次 collective 后由 CPU 等待并同步，默认为 0。
  * `-G,--cudagraph <num graph launches>`：将迭代捕获为 CUDA graph，并重放指定次数，默认为 0。
  * `-C,--report_cputime <0/1>]`：报告 CPU 时间而不是延迟，默认为 0。
  * `-R,--local_register <1/0>`：在 Send/Recv buffer 上启用本地内存注册，默认为 0。
  * `-T,--timeout <time in seconds>`：为每个测试设置超时时间，默认禁用。

## 版权

NCCL tests 采用 BSD 许可证。所有源代码及配套文档版权归 NVIDIA CORPORATION 所有，Copyright (c) 2016-2024。保留所有权利。
