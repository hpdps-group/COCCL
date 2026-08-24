# Build COCCL

English | [简体中文](README_zh-CN.md)

Set the CUDA and MPI installation paths, then run the script from any
directory:

```bash
CUDA_HOME=/path/to/cuda \
MPI_HOME=/path/to/mpi \
CMAKE_HOME=/path/to/cmake-prefix \
CUDA_ARCH=80 \
bash /path/to/COCCL/examples/build_scripts/build.sh
```

The script builds COCCL, all bundled compressor plugins, the configuration
checker, and the MPI-enabled communication tests. `COCCL_ROOT` defaults to the
repository containing the script. `BUILD_JOBS` controls parallel compilation,
and `NVCC_GENCODE` can replace the value generated from `CUDA_ARCH`.
`CMAKE_HOME` is optional when `cmake` is already available in `PATH`.
