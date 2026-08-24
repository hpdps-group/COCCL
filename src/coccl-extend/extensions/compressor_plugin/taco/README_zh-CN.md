# TACO

[English](README.md) | 简体中文

TACO 使用 FP8 量化压缩张量并行通信。本目录包含其 CUDA kernel 和 COCCL 适配器。通用 SDK 与集成流程见[压缩器插件指南](../README_zh-CN.md)。

![TACO 概览](TACO_Overview.png)

## 配置

TACO 支持 `E4M3` 和 `E5M2`、可选的有限值饱和处理，以及 32、64、128、256 或 512 的 group size。

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

仅在需要覆盖所选 FP8 格式的默认最大值时设置 `fp8MaxValue`。

## 构建

先构建 COCCL，再构建该插件：

```bash
make -C src/coccl-extend/extensions/compressor_plugin/taco \
  CUDA_HOME=/path/to/cuda \
  NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"
```

共享库会写入 COCCL 的压缩器插件输出目录。

## 结果

TACO 报告的评估结果显示，端到端训练最高可获得 1.87 倍加速。消融实验中的验证集和测试集损失相对 baseline 均在 0.3% 以内：

| 方法 | 验证集损失 | 测试集损失 | 验证集变化 | 测试集变化 |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 2.389899 | 2.344701 | - | - |
| TACO | 2.395784 | 2.351210 | +0.25% | +0.28% |

![TACO 消融实验](Ablation.png)

![TACO 对比](Comparison.png)

## 引用

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

论文：<https://arxiv.org/abs/2604.24088>
