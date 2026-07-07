# Integrating a Compressor

COCCL loads compressors as CUDA shared libraries at communicator initialization time. A compressor is selected by environment variables, loaded from `NCCL_COMPRESSORS_LIB_PATH`, configured from `NCCL_COMPRESSORS_CONFIG_PATH`, and called through the `ncclCompressor_t` interface in `src/include/compressor.h`.

The paper describes this abstraction as UCPM: required `compress`, `decompress`, and config-registration routines, plus optional fused `decomp_reduce` and `decomp_reduce_recomp` routines. In this artifact, those routines are implemented as fields of `ncclCompressor_t`. 

## Built-in compressor examples

COCCL includes four integrated compressors by default. They are both usable compressors and reference examples for adding a new compressor:

| Compressor | Source directory | Runtime name | Shared library | Exported symbol |
| --- | --- | --- | --- | --- |
| SDP4Bit | `src/device/compress/sdp4bit/` | `sdp4bit` | `libsdp4bit.so` | `sdp4bit` |
| TAHQuant | `src/device/compress/tahquant/` | `tahquant` | `libtahquant.so` | `tahquant` |
| cuZFP | `src/device/compress/zfp/` | `cuzfp` | `libcuzfp.so` | `cuzfp` |
| TACO | `src/device/compress/taco/` | `taco` | `libtaco.so` | `taco` |


Their config files live under `src/device/compress/configs/` and `src/device/compress/configs_training/`. Use these three directories as examples for the expected source layout, Makefile integration, `ncclCompressor_t` descriptor, config parsing, and runtime naming convention.

### 1. Implement the compressor shared library

Create a new directory under `src/device/compress/`, for example:

```text
src/device/compress/mycomp/
  Makefile
  mycomp.cu
```

The compressor name must be consistent in four places:

- The runtime name in `NCCL_COMPRESSORS`, for example `mycomp`.
- The shared object filename, `libmycomp.so`.
- The exported C symbol, `mycomp`.
- The `ncclCompressor_t.name` field, `"mycomp"`.

COCCL opens the library as:

```text
$NCCL_COMPRESSORS_LIB_PATH/lib<name>.so
```

and resolves the compressor descriptor with:

```text
dlsym(handle, "<name>")
```

A minimal compressor descriptor looks like this:

```cpp
#include "compressor.h"

extern "C" __attribute__((visibility("default"))) const ncclCompressor_t mycomp{
  .name = "mycomp",
  .compress = myCompCompress,
  .decompress = myCompDecompress,
  .decompReduce = nullptr,
  .decompReduceComp = nullptr,
  .parseConfig = parseMyCompConfig
};
```

Implement the callback signatures exactly as defined in `src/include/compressor.h`.

- `compress` receives one original chunk count (`orgChunkCount`) and the number of chunks (`numChunks`, usually the rank count). It must set `*compbuff`, `*compChunkCount`, and `*compDatatype`.
- `decompress` must reconstruct `numChunks` chunks into the output buffer supplied by COCCL.
- `parseConfig` must allocate or initialize the compressor-specific config object. It must tolerate `configFile == nullptr` and use defaults.
- `decompReduce` is optional. If it is `nullptr`, COCCL falls back to decompressing and then reducing.
- `decompReduceComp` is optional. If it is `nullptr`, COCCL falls back to decompress, reduce, and recompress.

Use the CUDA stream passed by COCCL for all kernels and async copies. If the compressor allocates `*compbuff`, use `cudaMallocAsync` or `cudaMallocFromPoolAsync` with the `compMemPool` argument when it is non-null. Current compressors are written for `ncclFloat32`, `ncclFloat16`, and `ncclBfloat16` inputs and usually emit `ncclInt8` compressed data.

### 2. Build the compressor

`src/device/Makefile` enters `src/device/compress/`, and `src/device/compress/Makefile` builds every compressor subdirectory that contains a `Makefile`. For an in-tree compressor, copy the structure of `sdp4bit/Makefile` or `tahquant/Makefile` and change the output shared object:

```make
QUAN_SO := $(SUBOBJDIR)/libcompress/libmycomp.so
```

Then build COCCL as usual. After the build, verify that the compressor library and exported symbol exist:

```bash
ls -l <path to coccl>/build/obj/device/compress/libcompress/libmycomp.so
nm -D <path to coccl>/build/obj/device/compress/libcompress/libmycomp.so | grep ' mycomp$'
```

If the compressor is built outside the COCCL tree, copy `libmycomp.so` into the directory used by `NCCL_COMPRESSORS_LIB_PATH`.

### 3. Add config files

COCCL constructs config paths from the selected collective and compressor name:

```text
$NCCL_COMPRESSORS_CONFIG_PATH/<name>/<name>_<suffix>.config
```

For `mycomp`, create files such as:

```text
src/device/compress/configs/mycomp/mycomp_AR.config
src/device/compress/configs/mycomp/mycomp_RS.config
src/device/compress/configs/mycomp/mycomp_AG.config
```

The suffixes are:

| Collective | Default suffix | Inter-node suffix |
| --- | --- | --- |
| AllToAll | `A2A` | `A2A_Inter` |
| AllReduce | `AR` | `AR_Inter` |
| AllGather | `AG` | `AG_Inter` |
| ReduceScatter | `RS` | `RS_Inter` |
| SendRecv | `SR` | `SR_BWD` |

The helper in `src/include/compress_utils.h` reads simple `key: value` pairs. For example:

```yaml
groupCount: 128
quantBits: 4
quantType: Symmetric
hadamard: 0
```

Your `parseConfig` callback is responsible for interpreting these keys. COCCL passes `nodes` and `devicesPerNodes` to `parseConfig`, so a compressor can derive topology-aware defaults when needed.

### 4. Enable the compressor at runtime

Set the global loader variables first. `NCCL_ENABLE_COMPRESS=1` without `NCCL_COMPRESSORS` is not enough, because COCCL needs the compressor list before it can load any shared libraries.

```bash
export COCCL_PATH=<path to coccl>
export NCCL_HOME=$COCCL_PATH/build
export LD_LIBRARY_PATH=$NCCL_HOME/lib:$LD_LIBRARY_PATH

export NCCL_ENABLE_COMPRESS=1
export NCCL_COMPRESSORS=mycomp
export NCCL_COMPRESSORS_LIB_PATH=$COCCL_PATH/build/obj/device/compress/libcompress
export NCCL_COMPRESSORS_CONFIG_PATH=$COCCL_PATH/src/device/compress/configs
export NCCL_COMPRESS_ENABLE_THRESHOLD=10485760  # optional: only compress messages >= 10 MB
```

Then enable compression for the collectives you want to test:

```bash
export NCCL_ENABLE_ALLREDUCE_COMPRESS=1
export NCCL_ALLREDUCE_COMPRESSORS=mycomp

export NCCL_ENABLE_REDUCESCATTER_COMPRESS=1
export NCCL_REDUCESCATTER_COMPRESSORS=mycomp
export NCCL_REDUCESCATTER_INTER_COMPRESSORS=mycomp

export NCCL_ENABLE_ALLGATHER_COMPRESS=1
export NCCL_ALLGATHER_COMPRESSORS=mycomp

export NCCL_ENABLE_ALLTOALL_COMPRESS=1
export NCCL_ALLTOALL_COMPRESSORS=mycomp

export NCCL_ENABLE_SENDRECV_COMPRESS=1
export NCCL_SENDRECV_COMPRESSORS=mycomp
export NCCL_SENDRECV_BWD_COMPRESSORS=mycomp
```

For 3D parallel training, the usual mapping is:

- Data parallelism: `AllGather` and `ReduceScatter`.
- Tensor parallelism: `AllReduce`.
- Pipeline parallelism: `SendRecv`.

`NCCL_COMPRESSORS` may list all loadable compressors, for example `sdp4bit,tahquant,mycomp`. Each enabled collective should then select the compressor it actually uses through its own `NCCL_*_COMPRESSORS` variable. Do not include spaces in comma-separated lists.

For inter-node collectives, the `*_INTER_COMPRESSORS` variable falls back to the default collective variable when it is unset. For send/recv, `NCCL_SENDRECV_BWD_COMPRESSORS` falls back to `NCCL_SENDRECV_COMPRESSORS` when unset.

### 5. Smoke test

Use `NCCL_DEBUG=INFO` and include the compression subsystem while bringing up a new compressor:

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,COMPRESS
```

After building `tests/coccl-tests`, run a small compressed collective:

```bash
mpirun -np 2 \
  -x LD_LIBRARY_PATH=$CUDA_HOME/lib64:$NCCL_HOME/lib:$MPI_HOME/lib \
  -x NCCL_ENABLE_COMPRESS=1 \
  -x NCCL_COMPRESSORS=mycomp \
  -x NCCL_COMPRESSORS_LIB_PATH=$NCCL_COMPRESSORS_LIB_PATH \
  -x NCCL_COMPRESSORS_CONFIG_PATH=$NCCL_COMPRESSORS_CONFIG_PATH \
  -x NCCL_ENABLE_ALLREDUCE_COMPRESS=1 \
  -x NCCL_ALLREDUCE_COMPRESSORS=mycomp \
  $COCCL_PATH/tests/coccl-tests/build/all_reduce_comp_twoshot_perf -b 1M -e 64M -f 2 -g 1 -c 1
```

If the run fails during communicator initialization, first check the shared-library name, the exported symbol name, and the config path generated from the suffix table above. The existing scripts `examples/benchmarks_scripts`and `examples/training_scripts` provide complete examples for communication benchmarks and LLM training.
