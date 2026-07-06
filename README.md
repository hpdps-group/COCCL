# COCCL

A collective communication library supporting easy integration and configuration of customized compression.

---

## Introduction

COCCL is a compression-aware GPU collective communication library built upon NCCL 2.21.5. It systematically integrates compression support into NCCL and provides NCCL-compatible APIs with a suite of collective communication pipelines optimized by high-performance GPU compression techniques.

COCCL is designed to be **extensible**, supporting customized compression operators through a unified compression programming model. Built-in compressors include:

* SDP4Bit
* TAHQuant
* cuZFP
* **TACO (Tensor-parallel Adaptive COmmunication compression)**

COCCL further introduces:

* Compression-aware collective algorithms
* Automatic algorithm selection
* Two-level runtime overlap to hide compression overhead

Compared with prior works, COCCL is the first NCCL-based system that **deeply integrates compression into collective primitives** and co-designs algorithms for GPU clusters.

---

## ✨ TACO: Tensor-parallel Adaptive COmmunication Compression

TACO is a **high-performance communication compression operator** designed for tensor-parallel LLM training and fully integrated into COCCL.

### Motivation

In tensor-parallel training, intermediate activations exhibit:

* Dense distributions
* Strong near-zero concentration
* High communication frequency

These properties make them highly compressible but sensitive to quantization error, requiring **structure-aware compression design**.

---

### Key Techniques

TACO introduces a set of techniques for high-fidelity and efficient compression:

* **FP8-based quantization**
  Efficient low-bit representation tailored for communication

* **Adaptive scaling**
  Dynamically adjusts scaling factors across tensor groups

* **Hadamard transform (optional)**
  Improves value distribution for more stable quantization

* **Dual-scale quantization**
  Enhances robustness under extreme values

* **Fused GPU kernels**
  Reduces kernel launch overhead and latency

* **Seamless 3D parallel integration**
  Compatible with TP / PP / DP in Megatron-LM

---

## Build

```bash
git clone https://github.com/hpdps-group/coccl.git
chmod 777 -R coccl
cd coccl
make -j src.build
```

If CUDA is not installed in the default path:

```bash
make src.build CUDA_HOME=<path to cuda install>
```

Optional: restrict GPU architectures:

```bash
make -j src.build NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"
```

Clean build:

```bash
make clean
```

---

## Environment Setup

```bash
export COCCL_PATH=<path to coccl>
export NCCL_HOME=$COCCL_PATH/build
export LIBRARY_PATH=$NCCL_HOME/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$NCCL_HOME/lib:$LD_LIBRARY_PATH
export C_INCLUDE_PATH=$NCCL_HOME/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$NCCL_HOME/include:$CPLUS_INCLUDE_PATH
```

> CUDA ≥ 12.2 is required.

---

## Tests

```bash
cd tests/coccl-tests
make -j MPI=1 MPI_HOME=<path to mpi> CUDA_HOME=<path to cuda> NCCL_HOME=$NCCL_HOME
./build/alltoall_comp_perf -b 1M -e 1G -f 2 -t <ngpus> -g 1
```

---

## Using Compression

### Enable Runtime

```bash
export NCCL_ENABLE_COMPRESS=1
```

### Load Compressors

```bash
export NCCL_COMPRESSORS=taco
```

Multiple compressors:

```bash
export NCCL_COMPRESSORS=sdp4bit,tahquant,taco
```

---

## TACO Configuration

Configuration directory:

```bash
src/device/compress/configs/taco
```

Example:

```yaml
fp8_format: 1        # 0=E4M3, 1=E5M2
saturation: 1
use_scale: 1
group_size: 256
target_range: 57734.0f
safe_mode: 0
# Hadamard Transform
use_as_hadamard: 1
lambda: 1e-6f
fp8_max_val: 57734.0f
pivotSwap: 0
```

---

## TACO Implementation

Source:

```bash
src/device/compress/taco
```

Compile only TACO:

```bash
cd src/device/compress/taco
make
```

---

## Deploying COCCL in LLM Frameworks

COCCL can replace NCCL backend in frameworks such as Megatron-LM and PyTorch.

### Step 1: Check NCCL linkage

```bash
ldd libtorch.so | grep nccl
```

### Step 2: Replace NCCL with COCCL

```bash
export NCCL_HOME=$COCCL_PATH/build
export LD_LIBRARY_PATH=$NCCL_HOME/lib:$LD_LIBRARY_PATH
```

Verify:

```bash
ldd libtorch.so | grep nccl
```

---

## Training Examples

```bash
examples/training_scripts/GPT2/train_gpt_coccl.sh
examples/training_scripts/Qwen2.5/train_qwen_coccl.sh
```

Enable TACO:

```bash
export NCCL_COMPRESSORS=taco
```

---

## Performance

### TACO End-to-End Training

* Up to **1.87× training speedup**
* Near-lossless accuracy vs FP32 baseline
* Best performance under **TP ≥ 4**

---

### TACO Ablation

| Method   | Val Loss ↓ | Test Loss ↓ | Val Deg. ↓ | Test Deg. ↓ |
| -------- | ---------- | ----------- | ---------- | ----------- |
| Baseline | 2.389899   | 2.344701    | –          | –           |
| TahQuant | 2.458742   | 2.413642    | +2.88%     | +2.94%      |
| **TACO** | 2.395784   | 2.351210    | **+0.25%** | **+0.28%**  |

---

## 📈 Comparison Results

![Comparison](Comparison.png)


## Integrating a Compressor

See:

```bash
src/device/compress/README.md
```

TACO serves as a reference implementation for custom compressor integration.

---

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

---

## Notes

* Fully NCCL-compatible
* Designed for large-scale GPU clusters
* Works with Megatron-LM / PyTorch
* Supports TP / PP / DP (3D parallelism)

---
