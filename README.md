# COCCL

A collective communication library supporting easy integration and configuration of customized compression.

## Introduction

COCCL is a compression-aware GPU collective communication library built upon NCCL 2.21.5. It systematically integrates compression support into NCCL and provides NCCL-compatible APIs with a suite of collective communication pipelines optimized by high-performance GPU compression techniques. COCCL is designed to be extensible, supporting customized compression operators through a unified compression programming model, with [SDP4Bit](https://github.com/ByteDance-Seed/SDP4Bit), [TACO](src/coccl-extend/extensions/compressor_plugin/taco), TAHQuant, ZFP, and dietGPU included by default. COCCL also introduces automatic algorithm selection and a two-level runtime overlap mechanism to hide compression overhead. We acknowledge that while some existing works support collective communication with compression, such as [1]-[5], COCCL is the first NCCL-based collective communication library to deeply integrate compression operators and co-design compression-aware algorithms for multiple collective primitives within GPU clusters.


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
export LD_LIBRARY_PATH=$NCCL_HOME/lib:$NCCL_HOME/obj/coccl-extend/compressor_plugin/libcompress:${LD_LIBRARY_PATH:-}
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

For source ownership and extension points, see
[src/coccl-extend/README.md](src/coccl-extend/README.md). Compressor integration
instructions are in
[src/coccl-extend/extensions/compressor_plugin/README.md](src/coccl-extend/extensions/compressor_plugin/README.md).

## Configuration

COCCL is NCCL-compatible, so standard NCCL environment variables such as
`NCCL_DEBUG`, `NCCL_IB_HCA`, and `NCCL_SOCKET_IFNAME` remain available. COCCL
itself reads only two environment variables:

| Variable | Meaning |
| --- | --- |
| `COCCL_ENABLE` | `1` enables COCCL routing; unset or `0` keeps native NCCL behavior. |
| `COCCL_CONFIG_FILE` | Path to one process-wide schema-v3 TOML file. Required when `COCCL_ENABLE=1`. |

Configuration is loaded once per process. An invalid file disables COCCL for
that process and reports the parser error through NCCL logging. Relative plugin
paths are resolved relative to the TOML file.

### Global settings

| TOML key | Default or accepted value | Effect |
| --- | --- | --- |
| `schema_version` | Required: `3` | Selects the configuration schema. |
| `runtime.mode` | Required: `normal` or `training` | Selects operation-based policies or automatically classified DP/TP/PP policies. |
| `runtime.compression_threshold_bytes` | `8388608` | Default minimum logical message size for automatic routing. COCCL compresses only when bytes are greater than the threshold. |
| `compressor_plugins.compressors` | `[]` | Names of plugin DSOs to load, for example `sdp4bit` or `zfp`. Every compressor referenced by a policy must be listed. |
| `compressor_plugins.library_path` | Empty | Directory containing `lib<name>.so`; required when the catalog is nonempty. |
| `pipeline.depth` | `1`, range `1..16` | Number of slices in compression/communication overlap. |

Explicit `coccl*Comp*` APIs still require a configured compressor policy, but
they bypass the compression threshold. Calls through standard NCCL APIs use
the threshold and fall back to native NCCL when the operation, datatype,
shape, policy, or compressor is unsupported.

Buffer settings control the communicator-owned temporary-memory pool:

| TOML key | Default | Effect |
| --- | ---: | --- |
| `buffer.legacy_block_bytes` | `0` | Minimum block size for the legacy `ncclMemAlloc` backend; `0` allocates to the current request size. |
| `buffer.physical_chunk_bytes` | `8388608` | Physical mapping growth unit for the CUDA VMM backend. It must be greater than zero. |
| `buffer.pool_limit_bytes` | `0` | Reserved configuration field. The current allocator does not enforce this limit; leave it at `0`. |

Autotuning applies to ReduceScatter and AllReduce algorithm selection:

| TOML key | Default or accepted value | Effect |
| --- | --- | --- |
| `autotune.enabled` | `true` | Profiles communication and codec cost and scores legal algorithms. If a required model is unavailable, selection uses the built-in heuristic. |
| `autotune.profile_min_bytes` | `262144` | Smallest profiling sample. Must be greater than zero. |
| `autotune.profile_max_bytes` | `8589934592` | Largest profiling sample; runtime also caps it to one quarter of free device memory. |
| `autotune.warmup` | `3` | Warmup iterations per profiling point. |
| `autotune.iterations` | `10` | Timed iterations per profiling point. |
| `autotune.reduce_scatter_algorithm` | `auto` | `auto`, `oneshot`, or `twoshot`. A forced unavailable TwoShot falls back to OneShot. |
| `autotune.all_reduce_algorithm` | `auto` | `auto`, `oneshot`, `twoshot`, or `tripleshot`. A forced unavailable TripleShot falls back to TwoShot. |

### Normal mode

Normal mode selects a policy by collective type:

```text
normal.all_gather
normal.reduce_scatter
normal.all_reduce
normal.all_to_all
normal.sendrecv
```

Each policy accepts `threshold_bytes` and three compressor scopes:

| Scope | Meaning |
| --- | --- |
| `default` | Flat/global communication and the fallback for an unspecified `intra` or `inter` scope. |
| `intra` | Node-local phase, or a Send/Recv whose peer is on the same node. |
| `inter` | Cross-node phase, or a Send/Recv whose peer is on another node. |

Each configured scope contains:

| TOML key | Default | Effect |
| --- | --- | --- |
| `enabled` | `true` when the scope table exists | Enables or explicitly disables compression for this scope. A disabled scope cannot also name a compressor or config. |
| `compressor` | Required when enabled | Plugin name from `compressor_plugins.compressors`. |
| `config` | Empty table | Scalar parameters passed to that plugin's `configure()` method; accepted keys are plugin-specific. |

An explicitly configured `intra` or `inter` scope overrides `default`.
`enabled = false` explicitly runs that phase without compression; an omitted
scope inherits `default`, and an omitted policy has compression disabled.
Hierarchical ReduceScatter TwoShot uses `intra` then `inter`; AllReduce
TripleShot uses `intra`, `inter`, and `default` for its final global phase.

This example keeps node-local ReduceScatter native and compresses only the
cross-node phase:

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

### Training mode

Training mode first identifies each communicator as data parallel (DP), tensor
parallel (TP), pipeline parallel (PP), or unknown, then selects the matching
policy. The classifier settings are:

| TOML key | Default or accepted value | Effect |
| --- | --- | --- |
| `training.observation_iterations` | `5`, range `2..100` | Repeated iterations used when topology alone cannot distinguish communicator roles. |
| `training.max_events` | `65536` | Maximum retained trace events; the effective runtime range is `256..1048576`. |
| `training.classifier.data_parallel_size` | Required in training mode | Expected DP communicator size. |
| `training.classifier.tensor_parallel_size` | Required in training mode | Expected TP communicator size. |
| `training.classifier.pipeline_parallel_size` | Required in training mode | Expected PP communicator size. |
| `training.classifier.dp_strategy` | `sdp` | `ddp` expects AllReduce buckets, `sdp` uses sharded optimizer communication patterns, and `fsdp` expects recurring AllGather/ReduceScatter flows. |
| `training.classifier.sequence_parallel` | `false` | Tells the classifier whether TP may use ReduceScatter/AllGather instead of the usual dense AllReduce pattern. |
| `training.classifier.context_parallel` | `false` | Records whether the job uses context parallelism. It currently does not select a separate role or policy. |

Training policies use the same `threshold_bytes` and
`default`/`intra`/`inter` scope grammar as normal mode:

```text
training.dp.{all_gather,reduce_scatter,all_reduce}
training.tp.{all_gather,reduce_scatter,all_reduce}
training.pp.sendrecv.{forward,backward}
```

When DP and TP communicator sizes are different, COCCL classifies them
immediately from the configured sizes. Otherwise it observes AllGather,
ReduceScatter, AllReduce, Send, and Recv calls, ignores ambiguous messages
smaller than 1 MiB, detects repeated iteration schedules, and combines
operation shape, DP strategy, topology, ordering, and PP boundaries. Traffic
remains on native NCCL until a role is committed; unknown communicators stay
native. PP direction is derived locally from rank and peer, allowing separate
forward and backward compressors.

Enable classification logs with:

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=TUNING
```

The complete training example is
[src/coccl-extend/extensions/configs/training.toml](src/coccl-extend/extensions/configs/training.toml).
It enables the validated SDP4Bit DP policies; TP and PP remain native until
the user adds corresponding policies.
Framework loading, submodule setup, and Qwen launch instructions are in
[examples/training_scripts/README.md](examples/training_scripts/README.md).

### Validate a configuration

Examples are under `src/coccl-extend/extensions/configs/`. Validate parsing,
plugin loading, plugin parameters, and the effective inherited scopes without
starting a communicator:

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

**Setup.** Experiments are conducted on a 4-node H800 cluster, with 8 NVIDIA H800 SXM5 80 GB GPUs per node, 8 InfiniBand links, CUDA 12.6, NVIDIA driver 550.90.07, NCCL 2.21.5, PyTorch 2.5.1, and Megatron-LM. Communication benchmarks use `nccl-tests`; end-to-end training uses 3D parallelism with SDP4Bit, TAHQuant, and cuZFP.

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
