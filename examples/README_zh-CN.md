# COCCL 示例

[English](README.md) | 简体中文

这些示例覆盖普通用户的完整使用流程：构建 COCCL、测试通信性能，然后运行 Qwen 训练。所有脚本都可以从任意目录启动。

## 1. 构建

```bash
CUDA_HOME=/path/to/cuda \
MPI_HOME=/path/to/mpi \
CMAKE_HOME=/path/to/cmake-prefix \
CUDA_ARCH=80 \
bash /path/to/COCCL/examples/build_scripts/build.sh
```

该脚本会构建 COCCL、内置插件、配置检查器和支持 MPI 的测试程序。详见[构建脚本](build_scripts/README_zh-CN.md)。

## 2. 通信性能测试

单节点：

```bash
CUDA_HOME=/path/to/cuda MPI_HOME=/path/to/mpi GPUS_PER_NODE=4 \
bash /path/to/COCCL/examples/benchmarks_scripts/communication_benchmarks.sh single
```

多节点：

```bash
CUDA_HOME=/path/to/cuda MPI_HOME=/path/to/mpi \
HOSTFILE=/path/to/hostfile NNODES=2 GPUS_PER_NODE=4 \
bash /path/to/COCCL/examples/benchmarks_scripts/communication_benchmarks.sh multi
```

默认测试矩阵和覆盖参数见[通信性能测试脚本](benchmarks_scripts/README_zh-CN.md)。

## 3. 训练 Qwen

初始化训练框架子模块，然后编辑 `training_envs.sh`：

```bash
git submodule update --init --recursive \
  examples/training_scripts/Pai-Megatron-Patch
```

在每个节点运行相同命令，只修改最后的节点 rank：

```bash
MASTER_ADDR=10.0.0.1 MASTER_PORT=29501 \
bash /path/to/COCCL/examples/training_scripts/train_qwen_coccl.sh qwen3 2 4 0
```

启动脚本也支持 `qwen25`。所需路径、拓扑配置以及 DP/TP/PP 自动路由说明见[训练脚本](training_scripts/README_zh-CN.md)。
