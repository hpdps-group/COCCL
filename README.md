# COCCL

English | [简体中文](README_zh-CN.md)

COCCL is a compression-aware GPU collective communication library that makes
customized compression easy to integrate and configure. Built upon NCCL
2.31.2, it preserves NCCL-compatible APIs and provides compression-aware
pipelines for AllGather, ReduceScatter, AllReduce, AllToAll, Send, and Recv.
Its unified C++17 plugin model supports fixed-layout and framed
variable-length compression operators and includes
[SDP4Bit](https://github.com/ByteDance-Seed/SDP4Bit),
[TACO](src/coccl-extend/extensions/compressor_plugin/taco),
[ZFP](https://github.com/LLNL/zfp), and
[dietGPU](https://github.com/facebookresearch/dietgpu) by default.
COCCL also provides pipeline overlap, automatic algorithm selection, and
training-aware policies for data, tensor, and pipeline parallel traffic.

For example, with SDP4Bit, COCCL achieves up to 2.60x, 2.58x, 5.66x, and
4.92x speedups on AllReduce, ReduceScatter, AllGather, and AllToAll,
respectively, compared with FP32 communication. In end-to-end 3D-parallel
training, COCCL improves throughput by up to 1.24x while maintaining model
accuracy.

(C) 2025 by Institute of Computing Technology, Chinese Academy of Sciences. See [COPYRIGHT](LICENSE.txt) in the top-level directory.


- Developers: Xingchen Liu, Haoran Kong, Man Liu, Xingjian Tian, Daran Sun, Zheng Wei, Liyang Zhao, Jingwu Yang

- Advisors: [Dingwen Tao](https://www.dingwentao.com/), [Guangming Tan](https://tanniu.github.io/), [Hairui Zhao](https://hairui-zhao.github.io/)

## Current Release

- The backend is NCCL 2.31.2. Release builds and A800 validation use CUDA
  12.8.
- Runtime validation covers four A800 GPUs on one node and eight A800 GPUs
  across two nodes. Native `sm_90` and `sm_100` builds are ready; Hopper and
  Blackwell runtime validation is pending target hardware. The A800 matrix
  covers depth 1/2/4/8, fixed SDP4Bit/ZFP, and inter-only framed dietGPU.
- Registered pipeline arenas use one physical allocation and can be used by
  NCCL symmetric or Host RMA windows. Unregistered raw staging arenas may grow
  with multiple physical segments.
- Fixed communication stages use NCCL's Host Alltoall and per-collective
  Config APIs. Framed payloads select internal AllGatherV, grouped P2P, or
  opt-in Host RMA according to the available backend capability.
- NCCL Zero-CTA/Copy Engine execution is available as an opt-in backend
  policy. The default remains NCCL `DEFAULT` because the end-to-end result is
  operation and compressor dependent.
- NCCL still chooses its internal Ring or PAT algorithm for native stages.
  COCCL autotuning selects only the high-level OneShot, TwoShot, or TripleShot
  compressed recipe. Codec profiling runs on one coordinator rank and the
  steady-state Host selection path is below one microsecond on the validated
  A800 system.

## Quick Start

### Requirements

- CUDA 12.8 (release-tested toolchain)
- A C++17-capable host compiler
- MPI for multi-process and multi-node tests
- CMake for the bundled ZFP and dietGPU plugins

### Build

To build the COCCL library:

```shell
git clone https://github.com/hpdps-group/COCCL.git
chmod 777 -R COCCL
cd COCCL
make -j src.build
```

If CUDA is not installed in the default `/usr/local/cuda` path, specify it
when building:

```shell
make -j src.build CUDA_HOME=<path to CUDA>
```

By default, COCCL is compiled for all supported GPU architectures. To build
only for the target architecture, set `NVCC_GENCODE`:

```shell
make -j src.build CUDA_HOME=<path to CUDA> \
  NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"
```

`src.build` builds the COCCL library and bundled compressor plugins. Build the
configuration checker separately when using the manual path:

```shell
make -f src/coccl-extend/Makefile coccl-config-check \
  BUILDDIR="$PWD/build" NCCLDIR="$PWD" \
  COCCL_ROOT=src/coccl-extend CUDA_HOME=<path to CUDA>
```

To clean the build:

```shell
make clean
```

For the automated build script, see
[examples/build_scripts](examples/build_scripts).

### After Build

COCCL is installed in `build/` unless `BUILDDIR` is set. Point applications
to the generated library and headers:

```shell
export COCCL_ROOT=/path/to/COCCL
export NCCL_HOME=$COCCL_ROOT/build
export LIBRARY_PATH=$NCCL_HOME/lib:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=$NCCL_HOME/lib:${LD_LIBRARY_PATH:-}
export C_INCLUDE_PATH=$NCCL_HOME/include:${C_INCLUDE_PATH:-}
export CPLUS_INCLUDE_PATH=$NCCL_HOME/include:${CPLUS_INCLUDE_PATH:-}

export COCCL_ENABLE=1
export COCCL_CONFIG_FILE=$COCCL_ROOT/examples/benchmarks_scripts/configs/sdp4bit.toml

$NCCL_HOME/bin/coccl-config-check "$COCCL_CONFIG_FILE"
your_application
```

Set `COCCL_ENABLE=0` or unset it to use native NCCL.

### Tests

To build [COCCL tests](tests/coccl-tests):

```shell
cd tests/coccl-tests
make -j MPI=1 MPI_HOME=<path to MPI> CUDA_HOME=<path to CUDA> \
  NCCL_HOME=$NCCL_HOME NVCC_GENCODE="$NVCC_GENCODE"
```

The test binaries are generated under `build/`. For example:

```shell
./build/alltoall_comp_perf -b 1M -e 1G -f 2 -t <number of GPUs> -g 1
```

For the provided single-node and multi-node test scripts, see
[examples/benchmarks_scripts](examples/benchmarks_scripts).

## Integrating A Compressor

The plugin SDK supports fixed-size, framed variable-length, stateful, and
fused-reduction codecs. New plugins live under
`src/coccl-extend/extensions/compressor_plugin/`; COCCL handles pipeline and
collective scheduling.

The current compressor ABI is v9. Rebuild external plugins when upgrading
from an older COCCL release. User-owned collective recipes, plugins, and
configs live under `src/coccl-extend/extensions/`; scheduling, memory,
compression runtime, tuning, and training classification live under
`src/coccl-extend/core/`.

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

The global threshold can be overridden by
`normal.<operation>.threshold_bytes`,
`training.{dp,tp}.<operation>.threshold_bytes`, or
`training.pp.sendrecv.{forward,backward}.threshold_bytes`.

Standard NCCL calls fall back to native NCCL when the selected compressor does
not support the operation, datatype, or shape. Explicit `coccl*Comp*` calls
bypass the size threshold but still require a matching policy.

The generated `nccl.h` exposes these explicit interfaces for testing, direct
compressed-path invocation, or manual algorithm selection:

- `cocclAllGatherComp` and `cocclAllToAllComp`;
- `cocclReduceScatterCompOneShot` and `cocclReduceScatterCompTwoShot`;
- `cocclAllReduceCompOneShot`, `cocclAllReduceCompTwoShot`, and
  `cocclAllReduceCompTripleShot`;
- `cocclSendComp` and `cocclRecvDecomp`.

Compressed Send/Recv processes one complete message serially. Grouped calls
batch their metadata and payload exchanges so pipeline-parallel traffic can
use the same fixed or framed protocol without creating per-direction streams.

### NCCL Backend Policy

COCCL leaves NCCL's CTA policy at `DEFAULT`. To benchmark Copy Engine or
hierarchical Copy Engine communication on a supported NCCL 2.31 system, set:

```bash
export NCCL_CTA_POLICY=ZERO
```

Eligible fixed stages then use registered symmetric windows; unsupported
stages keep NCCL's normal fallback. Zero-CTA can reduce SM and workspace use,
but it is intentionally opt-in because its end-to-end benefit depends on the
collective and compressor.

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

Each scope describes only the encoding produced by that scope. In a
hierarchical reduction, COCCL carries the previous scope's compressor
descriptor forward for decoding and uses the current scope's config for
recompression. An `inter` config therefore does not repeat the `intra` input
parameters.

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
patterns of at least 1 MiB when DP and TP are ambiguous. A uniquely matching
configured parallel size is classified immediately; ambiguous traffic stays
on native NCCL during the observation window and switches only after the role
is committed for that communicator. `training.observation_iterations`
defaults to 5.

Minimal example:

```toml
[runtime]
mode = "training"

[training]
observation_iterations = 5

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

If a model is unavailable, COCCL uses its built-in heuristic. Profiling is
lazy and process-wide. The model compares COCCL recipes; NCCL continues to
select the transport algorithm used inside every native stage.

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

The following published results use 32 H800 GPUs, CUDA 12.6, NCCL 2.21.5, and
InfiniBand. They are retained as the paper baseline; the current NCCL 2.31.2
release passed separate single-node and two-node A800 regression validation.
COCCL-SDP4Bit reaches up to 2.60x AllReduce, 2.58x ReduceScatter, 5.66x
AllGather, and 4.92x AllToAll speedup over FP32 NCCL.

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
