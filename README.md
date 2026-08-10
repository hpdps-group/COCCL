# COCCL_PPoPP_AE

## Build

To build the library :

```shell
git clone https://github.com/maomaohpp/COCCL_PPoPP_AE.git
chmod 777 -R COCCL_PPoPP_AE
cd COCCL_PPoPP_AE
bash build.sh /path/to/cuda \
/path/to/mpi \
/path/to/COCCL_PPoPP_AE \
"-gencode=arch=compute_90,code=sm_90"
# use the corresponding NVCC_GENCODE for your hardware
```

## Intrgrated with Framework

Please install the necessary environment dependencies for training frameworks such as [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) or [PyTorch](https://github.com/pytorch/pytorch).

To switch from the NCCL library to the COCCL library, follow the steps below:

1. Confirm whether the NCCL library used by PyTorch is a dynamic library.

   - Confirm the location of the PyTorch library.
     If you know that PyTorch is installed in a specific directory, you can search directly within that directory. For example, after confirming that PyTorch resides in `/usr/local/lib`, running the query command successfully pinpointed the exact path to the `libtorch.so` file, as shown below:

     ```bash
     find /usr/local/lib -name "libtorch*"
     # The example results are as follows:
     /usr/local/lib/python3.10/dist-packages/torch/lib/libtorchcuda.so
     /usr/local/lib/python3.10/dist-packages/torch/lib/libtorch.so
     /usr/local/lib/python3.10/dist-packages/torch/lib/libtorchbindtest.so
     ```

   - Use the `ldd` command to inspect the PyTorch library’s dependency on the NCCL library.

     ```bash
     ldd libtorch.so | grep nccl
     ```

     If the command returns results in the following format, it indicates that PyTorch depends on NCCL as a dynamic library. You may then proceed to configure COCCL according to the subsequent steps.

     ```bash
     libnccl.so.2=>/usr/lib/x86_64-linux-gnu/libnccl.so.2(0x00007feab3b27000)
     ```

     If the command returns no results, this indicates that PyTorch relies on NCCL as a static (non-dynamic) library and therefore cannot be switched to COCCL. To proceed with COCCL configuration, you must use a PyTorch version that depends on the NCCL dynamic library.

2. Before running the training script, please specify the paths to COCCL and the compressor dynamic libraries in COCCL_PPoPP_AE/training_scripts/training_envs.sh.

   ```bash
   #!/bin/bash
   export CUDA_PATH=/path/to/cuda
   export COCCL_PATH=/path/to/COCCL_PPoPP_AE
   export MEGATRON_PATH=/path/to/Megatron
   export DATASET_PATH=/path/to/dataset
   ```

## COCCL Configuration

COCCL reads exactly two environment variables. `COCCL_ENABLE=0` disables it;
when enabled, all remaining settings come from one TOML file:

```bash
export COCCL_ENABLE=1
export COCCL_CONFIG_FILE=/path/to/coccl.toml
```

The file is parsed and validated once per process. Invalid TOML, an unknown
key, an incompatible plugin, or a policy that references a plugin outside the
catalog disables COCCL and falls back to native NCCL. Relative
`compressor_plugins.library_path` values are resolved relative to the TOML
file.

```toml
schema_version = 2

[runtime]
mode = "normal" # normal | training
compression_threshold_bytes = 8388608

[compressor_plugins]
compressors = ["sdp4bit", "tahquant"]
library_path = "/path/to/compressor/libs"

[pipeline]
depth = 4

[normal.reduce_scatter]
compressor = "sdp4bit"

[normal.reduce_scatter.config.default]
groupCount = 128
quantBits = 4
quantType = "Symmetric"

[normal.reduce_scatter.config.hierarchical]
inQuantBits = 4
outQuantBits = 4
```

`pipeline.depth` is an explicit slice count in the range 1 through 16. A depth
of 1 executes the declared stages serially; values of 2 or greater enable slice
overlap.

The plugin catalog only loads `lib<compressor>.so`; it never forms an
execution policy. A primitive is compressed only when its policy contains a
non-empty `compressor` name. `threshold_bytes` optionally overrides
the runtime default for that policy. Normal mode supports `all_gather`,
`reduce_scatter`, `all_reduce`, `all_to_all`, and one bidirectional `sendrecv`
policy. `default` and `hierarchical` are independent parameter sets selected by
the primitive path; `hierarchical` does not inherit from `default`.

Training mode uses `training.dp` and `training.tp` policies for AllGather,
ReduceScatter, and AllReduce. Pipeline parallel Send/Recv is configured
independently under `training.pp.sendrecv.forward` and
`training.pp.sendrecv.backward`; a missing direction does not inherit from the
other direction. Until communicator classification is committed, or when its
role remains unknown, communication uses native NCCL.

Complete examples are provided in
[`normal.toml`](src/coccl-extend/configs/normal.toml) and
[`training.toml`](src/coccl-extend/configs/training.toml). The
[`zfp_taco.toml`](src/coccl-extend/configs/zfp_taco.toml) example shows a
CMake-based ZFP plugin and a Makefile-based TACO plugin. Compressor options
are scalar TOML values passed as key/value strings to the selected plugin. The
plugin's `coccl::ConfigReader` applies defaults, performs strict type and range
checks, and rejects unknown options.

### Adding a compressor

Compressor plugins use the C++17 SDK header
`src/coccl-extend/include/compressor_plugin/compressor_plugin.h`. A stateless plugin only
implements the two required operations and registers its name:

```cpp
#include "compressor_plugin/compressor_plugin.h"

struct MyCompressor {
  static coccl::Status compress(const coccl::Input& input,
                                coccl::Output& output,
                                coccl::Context& context) {
    const size_t compressedBytes =
        launchCompress(input.data(), output.data(), input.elements(),
                       context.stream());
    return output.commitBytes(compressedBytes, input.chunks());
  }

  static coccl::Status decompress(const coccl::Input& input,
                                  coccl::Output& output,
                                  coccl::Context& context) {
    launchDecompress(output.data(), input.data(), output.elements(),
                     context.stream());
    return ncclSuccess;
  }
};

COCCL_REGISTER_COMPRESSOR("mycompressor", MyCompressor);
```

The registration macro generates the ABI v5 descriptor, operation dispatch,
configuration lifecycle, and fixed `cocclGetCompressorPlugin` entry point.
`configure()`, `decompressReduce()`, and `decompressReduceCompress()` are
optional and detected at compile time. `Context::scratch()`,
`Context::persistent()`, and `Context::instance()` allocate resources only
when the plugin actually calls them; a normal stateless path creates none of
their runtime metadata.

Plugins keep their own Makefile or CMake project. They only need C++17, the
COCCL SDK include directory, PIC/shared-library flags, and the conventional
output name `lib<compressor>.so`.

With `NCCL_DEBUG=INFO`, COCCL prints the normalized effective configuration
once when the process first loads the TOML file. The output includes inherited
thresholds, resolved plugin paths, selected compressors, and the independently
parsed default and hierarchical compressor parameters. Values passed through
the plugin ABI are shown as strings because that is their final in-memory
representation.

The build also produces a standalone checker that validates TOML structure,
loads every catalog plugin, checks its ABI descriptor, and runs each policy's
parameters through the SDK-generated configuration callback without creating
a GPU context or NCCL communicator:

```bash
build/bin/coccl-config-check src/coccl-extend/configs/normal.toml
build/bin/coccl-config-check --nodes 2 --devices-per-node 8 \
  src/coccl-extend/configs/training.toml
```

The topology arguments only populate the compressor configuration context;
they default to two nodes and eight devices per node. The checker exits with a
nonzero status for malformed TOML, missing/incompatible plugins, unknown
compressor options, or plugin parameter validation failures.
