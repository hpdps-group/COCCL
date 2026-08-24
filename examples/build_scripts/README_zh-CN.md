# 构建 COCCL

[English](README.md) | 简体中文

设置 CUDA 和 MPI 的安装路径，然后从任意目录运行脚本：

```bash
CUDA_HOME=/path/to/cuda \
MPI_HOME=/path/to/mpi \
CMAKE_HOME=/path/to/cmake-prefix \
CUDA_ARCH=80 \
bash /path/to/COCCL/examples/build_scripts/build.sh
```

该脚本会构建 COCCL、所有内置压缩器插件、配置检查器以及支持 MPI 的通信测试。`COCCL_ROOT` 默认指向脚本所在的仓库。`BUILD_JOBS` 控制并行编译任务数，`NVCC_GENCODE` 可以覆盖根据 `CUDA_ARCH` 生成的值。如果 `cmake` 已经在 `PATH` 中，则无需设置 `CMAKE_HOME`。
