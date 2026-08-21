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

To build the library:

```shell
git clone https://github.com/hpdps-group/coccl.git
chmod 777 -R coccl
cd coccl
make -j src.build
```

If CUDA is not installed in the default `/usr/local/cuda` path, specify the CUDA path with:

```shell
$ make src.build CUDA_HOME=<path to cuda install>
```

By default, COCCL is compiled for all supported architectures. To accelerate compilation and reduce binary size, redefine `NVCC_GENCODE` (defined in `makefiles/common.mk`) to include only the architecture of the target platform:

```shell
$ make -j src.build NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"
```

To clean the build, use:

```shell
$ make clean
```

instead of deleting the build directory manually.

COCCL is compiled and installed in `build/` unless `BUILDDIR` is set.

After building COCCL, set the required environment variables:

```shell
export COCCL_PATH=<path to coccl>
export NCCL_HOME=$COCCL_PATH/build
export LIBRARY_PATH=$NCCL_HOME/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$NCCL_HOME/lib:$LD_LIBRARY_PATH
export C_INCLUDE_PATH=$NCCL_HOME/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$NCCL_HOME/include:$CPLUS_INCLUDE_PATH
```

The current compression kernel implementation requires CUDA 12.2 or later to fully support the half type.

## Tests

To build [tests](tests/coccl-tests):


```shell
cd tests/coccl-tests
make -j MPI=1 MPI_HOME=<path to mpi install> CUDA_HOME=<path to cuda install> NCCL_HOME=$NCCL_HOME NVCC_GENCODE=$NVCC_GENCODE
./build/alltoall_comp_perf -b 1M -e 1G -f 2 -t <ngpus> -g 1
```

All tests are listed as binary files in the build directory. For more examples, see [benchmark examples](examples/benchmarks_scripts).

## Integrating a compressor

For source ownership and extension points, see
[src/coccl-extend/README.md](src/coccl-extend/README.md). Compressor integration
instructions are in
[src/coccl-extend/extensions/compressor_plugin/README.md](src/coccl-extend/extensions/compressor_plugin/README.md).

## Environment variables

COCCL is NCCL-compatible, so standard NCCL environment variables such as
`NCCL_DEBUG`, `NCCL_IB_HCA`, and `NCCL_SOCKET_IFNAME` remain available. COCCL
itself uses two environment variables:

- `COCCL_ENABLE=1` enables compression-aware routing. It is disabled when
  unset or set to `0`.
- `COCCL_CONFIG_FILE=/absolute/path/to/config.toml` selects the process-wide
  schema-v3 configuration.

The TOML file owns plugin loading, per-operation `default`/`intra`/`inter`
policies, thresholds, pipeline depth, autotuning, and training-mode routing.
Examples are under `src/coccl-extend/extensions/configs/`. Validate a file
without starting a communicator with:

```shell
build/bin/coccl-config-check path/to/config.toml
```

## Deploying COCCL in LLM frameworks

Install the necessary environment dependencies for training frameworks such as [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) or [PyTorch](https://github.com/pytorch/pytorch).

To switch from the NCCL library to the COCCL library, follow the steps below:

1. Confirm whether the NCCL library used by PyTorch is a dynamic library.

   - Confirm the location of the PyTorch library.
     If PyTorch is installed in a specific directory, search directly within that directory. For example, after confirming that PyTorch resides in `/usr/local/lib`, the following command locates the `libtorch.so` file:

     ```bash
     find /usr/local/lib -name "libtorch*"
     # Example output:
     /usr/local/lib/python3.10/dist-packages/torch/lib/libtorchcuda.so
     /usr/local/lib/python3.10/dist-packages/torch/lib/libtorch.so
     /usr/local/lib/python3.10/dist-packages/torch/lib/libtorchbindtest.so
     ```

   - Use the `ldd` command to inspect the PyTorch library’s dependency on the NCCL library.

     ```bash
     ldd libtorch.so | grep nccl
     ```

     If the command returns results in the following format, it indicates that PyTorch depends on NCCL as a dynamic library. You can then configure COCCL according to the subsequent steps.

     ```bash
     libnccl.so.2=>/usr/lib/x86_64-linux-gnu/libnccl.so.2(0x00007feab3b27000)
     ```

     If `ldd` returns no results, this indicates that PyTorch relies on NCCL as a static (non-dynamic) library and therefore cannot be switched to COCCL. To proceed with COCCL configuration, use a PyTorch version that depends on the NCCL dynamic library.

2. Install and replace the backend with the COCCL library.

   After confirming that the PyTorch library depends on NCCL dynamically, replace the NCCL backend loaded by PyTorch with COCCL.

   - Build and initialize the COCCL runtime environment.

     Follow the Build section to build COCCL, then configure the required environment variables, such as `NCCL_HOME` and `LD_LIBRARY_PATH`, so that applications load the COCCL library instead of the original NCCL library at runtime.

   - Verify whether PyTorch now resolves NCCL to the COCCL library.
   
     ```bash
     ldd <path to libtorch.so> | grep nccl
     ```

     If the output points to the COCCL build directory, the replacement has succeeded. For example:
   
     ```bash
     libnccl.so.2 => path/to/coccl/build/lib/libnccl.so.2
     ```
   
   - Replace the original NCCL library if `LD_LIBRARY_PATH` does not take effect.
   
     If `libnccl.so.2` is still resolved to the original NCCL installation, replace that shared library with the COCCL-provided version. Back up the original file before overwriting it:
   
     ```bash
     ORIG_NCCL=/usr/lib/x86_64-linux-gnu/libnccl.so.2
     COCCL_NCCL=$COCCL_PATH/build/lib/libnccl.so.2
     
     cp -a $ORIG_NCCL ${ORIG_NCCL}.bak
     rm -f $ORIG_NCCL
     cp -a $COCCL_NCCL $ORIG_NCCL
     ```
     
   - Check the PyTorch dependency again after replacement.
   
     ```bash
     ldd <path to libtorch.so> | grep nccl
     ldd <path to libtorch_cuda.so> | grep nccl
     ```
   
     The output should now resolve `libnccl.so.2` to the COCCL-provided library or to the original path that has been replaced by the COCCL library.
   
3. For running scripts, see [training examples](examples/training_scripts) for details.


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

Example scripts are provided for communication benchmarks and end-to-end training:

- Communication benchmark examples: `examples/benchmarks_scripts/`
- End-to-end training examples: `examples/training_scripts/`

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
