# COCCL

A collective communication library for easily integrating and configuring GPU
compression.

## Introduction

COCCL is a compression-aware GPU collective communication library built on
NCCL 2.21.5. It keeps NCCL-compatible APIs and adds compressed AllGather,
ReduceScatter, AllReduce, AllToAll, Send, and Recv pipelines. Existing NCCL
applications enable COCCL with a shared library and a TOML file; collective
call sites do not change.

Its C++17 plugin model supports fixed-size and variable-length codecs. COCCL
handles pipeline overlap, automatic algorithm selection, and training-aware
policies for data, tensor, and pipeline parallel traffic.

Bundled compressors include
[SDP4Bit](https://github.com/ByteDance-Seed/SDP4Bit),
[TACO](src/coccl-extend/extensions/compressor_plugin/taco),
[ZFP](https://github.com/LLNL/zfp), and
[dietGPU](https://github.com/facebookresearch/dietgpu).

Copyright (C) 2025 Institute of Computing Technology, Chinese Academy of
Sciences. See [LICENSE.txt](LICENSE.txt).

Developers: Xingchen Liu, Haoran Kong, Man Liu, Xingjian Tian, Daran Sun,
Zheng Wei, Liyang Zhao, Yufan Wang, and Jingwu Yang.

Advisors: [Dingwen Tao](https://www.dingwentao.com/),
[Guangming Tan](https://tanniu.github.io/), and
[Hairui Zhao](https://hairui-zhao.github.io/).

## Quick Start

### Requirements

- Linux with NVIDIA GPUs
- CUDA 12.4 or later
- MPI for multi-process and multi-node tests
- CMake for the bundled ZFP plugin

### Build

```bash
git clone --recurse-submodules https://github.com/hpdps-group/COCCL.git
cd COCCL

CUDA_HOME=/path/to/cuda \
MPI_HOME=/path/to/mpi \
CMAKE_HOME=/path/to/cmake-prefix \
CUDA_ARCH=80 \
bash examples/build_scripts/build.sh
```

This builds COCCL, bundled plugins, the configuration checker, and
MPI-enabled communication tests under `build/`. `CMAKE_HOME` is optional when
`cmake` is already in `PATH`.

For build options, see
[examples/build_scripts/README.md](examples/build_scripts/README.md).

### Run A Benchmark

Single node:

```bash
CUDA_HOME=/path/to/cuda MPI_HOME=/path/to/mpi GPUS_PER_NODE=4 \
bash examples/benchmarks_scripts/communication_benchmarks.sh single
```

Two nodes:

```bash
CUDA_HOME=/path/to/cuda \
MPI_HOME=/path/to/mpi \
HOSTFILE=/path/to/hostfile \
NNODES=2 GPUS_PER_NODE=4 \
bash examples/benchmarks_scripts/communication_benchmarks.sh multi
```

The default matrix compares native NCCL with SDP4Bit and ZFP at pipeline
depths 1, 2, 4, and 8. See
[examples/benchmarks_scripts](examples/benchmarks_scripts) for overrides.

### Use COCCL In An Application

```bash
export COCCL_ROOT=/path/to/COCCL
export LD_LIBRARY_PATH=$COCCL_ROOT/build/lib:${LD_LIBRARY_PATH:-}
export COCCL_ENABLE=1
export COCCL_CONFIG_FILE=$COCCL_ROOT/examples/benchmarks_scripts/configs/sdp4bit.toml

$COCCL_ROOT/build/bin/coccl-config-check "$COCCL_CONFIG_FILE"
your_application
```

Set `COCCL_ENABLE=0` or unset it to use native NCCL.

## Integrating A Compressor

The plugin SDK supports fixed-size, framed variable-length, stateful, and
fused-reduction codecs. New plugins live under
`src/coccl-extend/extensions/compressor_plugin/`; COCCL handles pipeline and
collective scheduling.

See the [compressor integration guide](src/coccl-extend/extensions/compressor_plugin/README.md)
for the SDK contract, a minimal adapter, build rules, validation, and the
Codex integration skill.

## Configuration

Start from a checked-in TOML file:

- [normal-mixed.toml](src/coccl-extend/extensions/configs/normal-mixed.toml):
  normal mode with ZFP and SDP4Bit;
- [training.toml](examples/training_scripts/configs/training.toml): training
  mode with SDP4Bit and autotuning;
- [benchmark configs](examples/benchmarks_scripts/configs): one file per
  bundled compressor.

Validate a file before running:

```bash
build/bin/coccl-config-check path/to/config.toml
```

### Runtime Settings

- `runtime.mode`: `normal` or `training`.
- `runtime.compression_threshold_bytes`: minimum logical message size for
  automatic compression; default 8 MiB.
- `compressor_plugins.compressors`: plugin names used by this file.
- `compressor_plugins.library_path`: directory containing `lib<name>.so`.
  Relative paths are resolved from the TOML file.
- `pipeline.depth`: overlap slices, from 1 to 16.

Standard NCCL calls fall back to native NCCL when the selected compressor does
not support the operation, datatype, or shape. Explicit `coccl*Comp*` calls
bypass the size threshold but still require a matching policy.

### Normal Mode

Normal mode selects policies by operation:

```text
normal.all_gather
normal.reduce_scatter
normal.all_reduce
normal.all_to_all
normal.sendrecv
```

Each operation can define:

- `default`: flat communication and fallback policy;
- `intra`: node-local phases;
- `inter`: cross-node phases.

An explicit `intra` or `inter` policy overrides `default`. Set
`enabled = false` to leave that scope on native NCCL.

This example loads two compressors and assigns one to each operation:

```toml
[runtime]
mode = "normal"
compression_threshold_bytes = 1048576

[compressor_plugins]
compressors = ["zfp", "sdp4bit"]
library_path = "/path/to/COCCL/build/obj/coccl-extend/compressor_plugin/libcompress"

[normal.all_reduce.default]
compressor = "zfp"

[normal.all_reduce.default.config]
rate = 8

[normal.reduce_scatter.default]
compressor = "sdp4bit"

[normal.reduce_scatter.default.config]
groupCount = 128
quantBits = 4
quantType = "Symmetric"
hadamard = true
```

### Training Mode

Training mode routes communication by parallel role:

```text
training.dp.{all_gather,reduce_scatter,all_reduce}
training.tp.{all_gather,reduce_scatter,all_reduce}
training.pp.sendrecv.{forward,backward}
```

Declare the framework topology in `training.classifier`:

- `data_parallel_size`, `tensor_parallel_size`, and
  `pipeline_parallel_size` must match the job;
- `dp_strategy` is `ddp`, `sdp`, or `fsdp`;
- `sequence_parallel = true` when TP uses ReduceScatter and AllGather instead
  of AllReduce.

COCCL uses communicator sizes first, then observes repeated collective
patterns when DP and TP are ambiguous. Unclassified traffic stays on native
NCCL.

Minimal example:

```toml
[runtime]
mode = "training"

[compressor_plugins]
compressors = ["sdp4bit"]
library_path = "/path/to/COCCL/build/obj/coccl-extend/compressor_plugin/libcompress"

[training.classifier]
data_parallel_size = 4
tensor_parallel_size = 2
pipeline_parallel_size = 1
dp_strategy = "sdp"
sequence_parallel = false

[training.dp.reduce_scatter.default]
compressor = "sdp4bit"

[training.dp.reduce_scatter.default.config]
groupCount = 128
quantBits = 4
quantType = "Symmetric"

[autotune]
enabled = true
reduce_scatter_algorithm = "auto"
all_reduce_algorithm = "auto"
```

The runnable Qwen2.5 and Qwen3 example is under
[examples/training_scripts](examples/training_scripts).

### Autotuning

Autotuning selects ReduceScatter and AllReduce algorithms:

- `autotune.enabled`: enable profiling and model-based selection; default
  `true`.
- `profile_min_bytes`, `profile_max_bytes`: profiling range; defaults 256 KiB
  to 8 GiB and is capped by free GPU memory.
- `warmup`, `iterations`: samples per profile point; defaults 3 and 10.
- `reduce_scatter_algorithm`: `auto`, `oneshot`, or `twoshot`.
- `all_reduce_algorithm`: `auto`, `oneshot`, `twoshot`, or `tripleshot`.

If a model is unavailable, COCCL uses its built-in heuristic.

### Logs

COCCL uses the NCCL logger:

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=COCCL_TUNING
```

Available subsystems:

- `COCCL`: all COCCL logs;
- `COCCL_INIT`: configuration and initialization;
- `COCCL_RUNTIME`: policy routing;
- `COCCL_PIPELINE`: pipeline execution;
- `COCCL_COMPRESS`: compressor calls and resources;
- `COCCL_MEMORY`: workspace allocation;
- `COCCL_TUNING`: autotuning and training classification.

## More Examples

- [Build](examples/build_scripts)
- [Communication benchmarks](examples/benchmarks_scripts)
- [Qwen training](examples/training_scripts)
- [COCCL tests](tests/coccl-tests)

## Performance

The following results use 32 H800 GPUs, CUDA 12.6, NCCL 2.21.5, and
InfiniBand. COCCL-SDP4Bit reaches up to 2.60x AllReduce, 2.58x ReduceScatter,
5.66x AllGather, and 4.92x AllToAll speedup over FP32 NCCL.

![Communication performance of COCCL](assets/results/communication_performance.png)

End-to-end training preserves validation loss while improving throughput:

<p align="center">
  <img src="assets/results/e2e_accuracy.png" alt="End-to-end validation loss with COCCL" width="50%">
</p>

<div align="center">
  <table>
    <thead>
      <tr>
        <th align="left">Model</th>
        <th align="right">Size</th>
        <th align="right">NCCL TFLOPS</th>
        <th align="right">COCCL TFLOPS</th>
        <th align="right">Speedup</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>GPT</td>
        <td align="right">2.7B</td>
        <td align="right">88.5</td>
        <td align="right">94.5</td>
        <td align="right">1.06x</td>
      </tr>
      <tr>
        <td>GPT</td>
        <td align="right">6.7B</td>
        <td align="right">148.6</td>
        <td align="right">163.2</td>
        <td align="right">1.10x</td>
      </tr>
      <tr>
        <td>GPT</td>
        <td align="right">13B</td>
        <td align="right">158.8</td>
        <td align="right">197.1</td>
        <td align="right">1.24x</td>
      </tr>
      <tr>
        <td>Qwen2.5</td>
        <td align="right">7B</td>
        <td align="right">222.3</td>
        <td align="right">234.2</td>
        <td align="right">1.05x</td>
      </tr>
    </tbody>
  </table>
</div>

## Citation

```bibtex
@inproceedings{liu2026coccl,
  title={COCCL: A Collective Communication Library Supporting Easy Integration and Configuration of Customized Compression for Scalable LLM Training},
  author={Liu, Xingchen and Kong, Haoran and Zhao, Hairui and Lyu, Shengkai and Wei, Zheng and Liu, Man and Tian, Xingjian and Zhao, Liyang and Chen, Zhuohan and Wang, Fakang and others},
  booktitle={Proceedings of the 31st ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming},
  pages={384--397},
  year={2026}
}
@misc{liu2026taco,
  title={TACO: Efficient Communication Compression of Intermediate Tensors for Scalable Tensor-Parallel LLM Training},
  author={Man Liu and Xingchen Liu and Xingjian Tian and Bing Lu and Shengkay Lyu and Shengquan Yin and Wenjing Huang and Zheng Wei and Hairui Zhao and Guangming Tan and Dingwen Tao},
  year={2026},
  eprint={2604.24088},
  archivePrefix={arXiv},
  primaryClass={cs.DC},
  url={https://arxiv.org/abs/2604.24088}
}
```
