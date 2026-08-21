# COCCL

A collective communication library supporting easy integration and configuration of customized compression.

## Introduction

COCCL is a compression-aware GPU collective communication library built upon NCCL 2.21.5. It systematically integrates compression support into NCCL and provides NCCL-compatible APIs with a suite of collective communication pipelines optimized by high-performance GPU compression techniques. COCCL is designed to be extensible, supporting customized compression operators through a unified compression programming model, with [SDP4Bit](https://github.com/ByteDance-Seed/SDP4Bit), [TACO](src/coccl-extend/extensions/compressor_plugin/taco), [ZFP](https://github.com/LLNL/zfp), and [dietGPU](https://github.com/facebookresearch/dietgpu) included by default. COCCL also introduces automatic algorithm selection and a two-level runtime overlap mechanism to hide compression overhead. We acknowledge that while some existing works support collective communication with compression, such as [1]-[5], COCCL is the first NCCL-based collective communication library to deeply integrate compression operators and co-design compression-aware algorithms for multiple collective primitives within GPU clusters.


For example, by utilizing SDP4Bit compression, COCCL achieves 2.60x, 2.58x, 5.66x, and 4.92x speedups on AllReduce, ReduceScatter, AllGather, and AlltoAll, respectively, compared to the original FP32-based communication. In end-to-end 3D-parallel training, the tuned COCCL-3D configuration improves GPT and Qwen2.5 training throughput by up to 1.24x while maintaining model accuracy. COCCL is particularly beneficial for applications requiring intensive collective communication, including large-scale model training, inference systems, and scientific computing.


Moving forward, we plan to incorporate NCCL device API support and integrate selected optimization mechanisms to further enhance COCCL's communication performance.

(C) 2025 by Institute of Computing Technology, Chinese Academy of Sciences. See [COPYRIGHT](LICENSE.txt) in the top-level directory.


- Developers: Xingchen Liu, Haoran Kong, Man Liu, Xingjian Tian, Daran Sun, Zheng Wei, Liyang Zhao, Yufan Wang, Jingwu Yang

- Advisors: [Dingwen Tao](https://www.dingwentao.com/), [Guangming Tan](https://tanniu.github.io/), [Hairui Zhao](https://hairui-zhao.github.io/)

## Build

Clone COCCL and initialize the optional training-framework submodule:

```bash
git clone --recurse-submodules https://github.com/hpdps-group/COCCL.git
cd COCCL
```

The user-facing build script compiles COCCL, all bundled compressor plugins,
the configuration checker, and MPI-enabled communication tests:

```bash
CUDA_HOME=/path/to/cuda \
MPI_HOME=/path/to/mpi \
CMAKE_HOME=/path/to/cmake-prefix \
CUDA_ARCH=80 \
bash examples/build_scripts/build.sh
```

`COCCL_ROOT`, `BUILD_JOBS`, and `NVCC_GENCODE` can override the inferred
repository, parallelism, and generated GPU architecture. See
[examples/build_scripts/README.md](examples/build_scripts/README.md) for the
complete build-script interface.

To build only the library manually:

```bash
make -j src.build \
  CUDA_HOME=/path/to/cuda \
  NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"
```

Use `make clean` to clean the build. COCCL is installed under `build/` unless
`BUILDDIR` is set. The current pipeline layout kernels require CUDA 12.4 or
later.

After building, make the COCCL NCCL-compatible library and plugins visible to
the application:

```bash
export COCCL_ROOT=/path/to/COCCL
export NCCL_HOME=$COCCL_ROOT/build
export LD_LIBRARY_PATH=$NCCL_HOME/lib:${LD_LIBRARY_PATH:-}
export C_INCLUDE_PATH=$NCCL_HOME/include:${C_INCLUDE_PATH:-}
export CPLUS_INCLUDE_PATH=$NCCL_HOME/include:${CPLUS_INCLUDE_PATH:-}
```

## Tests

The build script above compiles [tests/coccl-tests](tests/coccl-tests). A
focused manual build is also available:

```bash
cd tests/coccl-tests
make -j MPI=1 MPI_HOME=/path/to/mpi CUDA_HOME=/path/to/cuda \
  NCCL_HOME=$NCCL_HOME \
  NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"
./build/alltoall_comp_perf -b 1M -e 1G -f 2 -t 4 -g 1
```

Ready-to-run single-node and multi-node benchmark workflows are documented in
[examples/benchmarks_scripts](examples/benchmarks_scripts).

## Integrating a compressor

Source ownership, the compressor SDK contract, a minimal plugin, built-in
parameters, build validation, and the Codex integration skill are documented
in one place:
[src/coccl-extend/extensions/compressor_plugin/README.md](src/coccl-extend/extensions/compressor_plugin/README.md).

## Configuration

### Enable COCCL

COCCL keeps NCCL's environment interface. Enable routing and select one TOML
file for the process:

```bash
export COCCL_ENABLE=1
export COCCL_CONFIG_FILE=/path/to/coccl.toml
```

Unset `COCCL_ENABLE`, or set it to `0`, to use native NCCL. Configuration is
loaded once per process. Relative plugin paths are resolved from the TOML
file, and an invalid configuration disables COCCL with a parser error.

Standard NCCL settings such as `NCCL_IB_HCA`, `NCCL_SOCKET_IFNAME`, and
`NCCL_BUFFSIZE` continue to work. COCCL logs use NCCL's logger:

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=COCCL
```

`COCCL` enables every COCCL subsystem. Select individual areas with
`COCCL_INIT` for configuration and initialization, `COCCL_RUNTIME` for policy
routing, `COCCL_PIPELINE` for pipeline execution, `COCCL_COMPRESS` for codec
resources, `COCCL_MEMORY` for workspace allocation, or `COCCL_TUNING` for
autotuning and training-role classification.

### Common TOML Settings

Every configuration starts with `schema_version = 3` and a runtime mode:

- `runtime.mode` is `normal` or `training`.
- `runtime.compression_threshold_bytes` defaults to 8 MiB. Standard NCCL calls
  are compressed only when the logical message is larger than this threshold.
  Explicit `coccl*Comp*` APIs bypass the threshold but still require a policy.
- `compressor_plugins.compressors` lists the plugin names used by policies.
- `compressor_plugins.library_path` names the directory containing
  `lib<name>.so`. This is how COCCL finds plugins; the directory does not need
  to be added to `LD_LIBRARY_PATH`.
- `pipeline.depth` is the number of overlap slices and accepts `1..16`.

Unsupported operations, datatypes, shapes, or policies fall back to native
NCCL when called through the standard NCCL APIs.

### Normal Mode

Normal mode provides one policy for each operation:

```text
normal.all_gather
normal.reduce_scatter
normal.all_reduce
normal.all_to_all
normal.sendrecv
```

Each policy can override the global threshold with `threshold_bytes` and can
define three compressor scopes. `default` handles flat communication and is
the fallback for unspecified scopes. `intra` handles node-local phases;
`inter` handles cross-node phases. A scope names a `compressor` and an
optional plugin-specific `config` table. Set `enabled = false` to keep that
scope on native NCCL.

An explicit `intra` or `inter` scope overrides `default`. ReduceScatter
TwoShot executes `intra` then `inter`; AllReduce TripleShot additionally uses
`default` for its final global phase. This example leaves the local phase
native and compresses only cross-node ReduceScatter traffic:

```toml
schema_version = 3

[runtime]
mode = "normal"
compression_threshold_bytes = 1048576

[compressor_plugins]
compressors = ["zfp"]
library_path = "/path/to/COCCL/build/obj/coccl-extend/compressor_plugin/libcompress"

[normal.reduce_scatter]
threshold_bytes = 4194304

[normal.reduce_scatter.intra]
enabled = false

[normal.reduce_scatter.inter]
compressor = "zfp"

[normal.reduce_scatter.inter.config]
rate = 8

[pipeline]
depth = 4

[autotune]
enabled = false
reduce_scatter_algorithm = "twoshot"
```

Ready-to-edit normal-mode examples for SDP4Bit, ZFP, and dietGPU are under
`src/coccl-extend/extensions/configs/`.

### Training Mode

Training mode classifies communicators as data parallel (DP), tensor parallel
(TP), pipeline parallel (PP), or unknown, then applies role-specific policies:

```text
training.dp.{all_gather,reduce_scatter,all_reduce}
training.tp.{all_gather,reduce_scatter,all_reduce}
training.pp.sendrecv.{forward,backward}
```

`training.classifier.data_parallel_size`, `tensor_parallel_size`, and
`pipeline_parallel_size` are required. Set `dp_strategy` to `ddp`, `sdp`, or
`fsdp` to match the framework optimizer. Set `sequence_parallel = true` when
TP uses ReduceScatter/AllGather instead of the usual AllReduce pattern.
`training.observation_iterations` defaults to `5` and accepts `2..100`.

When DP and TP sizes differ, topology identifies them immediately. Otherwise
COCCL observes repeated AllGather, ReduceScatter, AllReduce, Send, and Recv
schedules. Ambiguous messages smaller than 1 MiB are ignored, and traffic
stays on native NCCL until a role is committed. PP direction is inferred from
the local rank and peer.

The following minimal training configuration compresses DP AllGather with
SDP4Bit while leaving TP and PP on native NCCL:

```toml
schema_version = 3

[runtime]
mode = "training"
compression_threshold_bytes = 67108864

[compressor_plugins]
compressors = ["sdp4bit"]
library_path = "/path/to/COCCL/build/obj/coccl-extend/compressor_plugin/libcompress"

[training]
observation_iterations = 5

[training.classifier]
data_parallel_size = 4
tensor_parallel_size = 2
pipeline_parallel_size = 1
dp_strategy = "sdp"
sequence_parallel = false

[training.dp.all_gather.default]
compressor = "sdp4bit"

[training.dp.all_gather.default.config]
groupCount = 2048
quantBits = 8
quantType = "Symmetric"
subAdd = true
pipelineSize = 1

[pipeline]
depth = 1
```

Use `NCCL_DEBUG_SUBSYS=COCCL_TUNING` to inspect classification and algorithm
selection. The complete validated example is
[training.toml](src/coccl-extend/extensions/configs/training.toml), and Qwen
launch instructions are in
[examples/training_scripts/README.md](examples/training_scripts/README.md).

### Algorithm And Memory Controls

Autotuning applies to ReduceScatter and AllReduce:

- `autotune.enabled` defaults to `true`. Missing models fall back to the
  built-in heuristic.
- `profile_min_bytes` and `profile_max_bytes` delimit profiling sizes; the
  defaults are 256 KiB and 8 GiB. Runtime also caps the maximum to one quarter
  of free device memory.
- `warmup` and `iterations` default to `3` and `10`.
- `reduce_scatter_algorithm` accepts `auto`, `oneshot`, or `twoshot`.
- `all_reduce_algorithm` accepts `auto`, `oneshot`, `twoshot`, or
  `tripleshot`.

The buffer pool normally needs no tuning. `buffer.legacy_block_bytes = 0`
allocates legacy blocks to the request size. `buffer.physical_chunk_bytes`
controls CUDA VMM growth and defaults to 8 MiB.

### Validate a Configuration

Validate parsing, plugin loading, plugin parameters, and inherited scopes
without starting a communicator:

```bash
build/bin/coccl-config-check path/to/config.toml
```

## Deploying COCCL in LLM frameworks

COCCL is loaded as an NCCL-compatible shared library; training frameworks do
not need COCCL-specific collective calls. Dynamic-link verification,
Pai-Megatron-Patch setup, Qwen2.5/Qwen3 launch commands, and training-role
detection diagnostics are maintained with the runnable scripts in
[examples/training_scripts](examples/training_scripts).


## Performance

**Setup.** Experiments are conducted on a 4-node H800 cluster, with 8 NVIDIA H800 SXM5 80 GB GPUs per node, 8 InfiniBand links, CUDA 12.6, NVIDIA driver 550.90.07, NCCL 2.21.5, PyTorch 2.5.1, and Megatron-LM. Communication benchmarks use `nccl-tests`; end-to-end training uses 3D parallelism with compression-aware collective policies.

- **Communication performance**

![Communication performance of COCCL](assets/results/communication_performance.png)

At 32 GPUs, COCCL-SDP4Bit achieves 2.60x, 2.58x, 5.66x, and 4.92x speedups on AllReduce, ReduceScatter, AllGather, and AlltoAll, respectively, compared with NCCL 2.21.5.

- **End-to-end training**

<p align="center">
  <img src="assets/results/e2e_accuracy.png" alt="End-to-end validation loss with COCCL-3D" width="50%">
</p>

<div align="center">
  <table>
    <thead>
      <tr>
        <th align="left">Model</th>
        <th align="right">Size</th>
        <th align="right">Baseline TFLOPS</th>
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

## Examples

Build, communication benchmark, and Qwen training scripts are documented in
[examples/README.md](examples/README.md).

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
