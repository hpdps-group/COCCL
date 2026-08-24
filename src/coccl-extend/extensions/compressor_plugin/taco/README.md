# TACO

English | [简体中文](README_zh-CN.md)

TACO compresses tensor-parallel communication with FP8 quantization. This
directory contains its CUDA kernels and COCCL adapter. See the
[compressor plugin guide](../README.md) for the common SDK and integration
workflow.

![TACO overview](TACO_Overview.png)

## Configure

TACO supports `E4M3` and `E5M2`, optional finite saturation, and group sizes
32, 64, 128, 256, or 512.

```toml
[training.tp.all_reduce.default]
compressor = "taco"

[training.tp.all_reduce.default.config]
fp8Format = "E4M3"
saturate = true
groupSize = 128
targetRange = 448.0
lambda = 1e-6
```

Set `fp8MaxValue` only when overriding the selected FP8 format's default
maximum.

## Build

Build COCCL first, then this plugin:

```bash
make -C src/coccl-extend/extensions/compressor_plugin/taco \
  CUDA_HOME=/path/to/cuda \
  NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"
```

The shared library is written to COCCL's compressor plugin output directory.

## Results

The reported TACO evaluation reaches up to 1.87x end-to-end training speedup.
The validation and test losses in the ablation were within 0.3% of baseline:

| Method | Validation loss | Test loss | Validation change | Test change |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 2.389899 | 2.344701 | - | - |
| TACO | 2.395784 | 2.351210 | +0.25% | +0.28% |

![TACO ablation](Ablation.png)

![TACO comparison](Comparison.png)

## Citation

```bibtex
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

Paper: <https://arxiv.org/abs/2604.24088>
