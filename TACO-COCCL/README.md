# TACO: Efficient Communication Compression for Tensor-Parallel LLM Training

This repository provides the implementation of **TACO**, a communication compression framework designed for efficient large-scale tensor-parallel LLM training.

---

## 📌 Overview

Communication overhead is a critical bottleneck in large-scale tensor-parallel training, especially due to the **dense and near-zero distributions** of intermediate tensors.

To address this, we propose **TACO (Tensor-parallel Adaptive COmmunication compression)**:

* ✅ FP8-based compression for intermediate tensors
* ✅ Adaptive Scale + Hadamard Transform for high-fidelity quantization
* ✅ Dual-scale quantization for numerical stability
* ✅ Fused compression kernels for low overhead
* ✅ Seamless integration with TP + PP + DP (3D parallelism)

### 🚀 Performance

* Up to **1.87× end-to-end training speedup**
* **Near-lossless accuracy**

---

## 🖼️ Framework Overview

![TACO Overview](TACO_Overview.png)

---

## ⚙️ Build COCCL Communication Library

```bash
cd coccl-acc
bash build.sh
```

---

## 🧩 TACO Compression Operator

Source directory:

```bash
coccl-acc/src/device/compress/taco
```

### 🔧 Compile Operator Only

```bash
cd coccl-acc/src/device/compress/taco
make
```

---

## ⚙️ Compression Configuration

Configuration path:

```bash
coccl-acc/src/device/compress/configs/taco
```

### Key Parameters

```bash
fp8_format: 1        # 0=E4M3, 1=E5M2
saturation: 1        # Enable saturation
use_scale: 1         # Enable scaling
group_size: 256      # Group size
target_range: 57734.0f

safe_mode: 0         # Disable safe mode

# Hadamard Transform
use_as_hadamard: 1   # Enable Hadamard transform
lambda: 1e-6f
fp8_max_val: 57734.0f
pivotSwap: 0
```

---

## 📊 Ablation Results

| Method   | Val Loss ↓ | Test Loss ↓ | Val Deg. ↓ | Test Deg. ↓ |
| -------- | ---------- | ----------- | ---------- | ----------- |
| Baseline | 2.389899   | 2.344701    | –          | –           |
| TahQuant | 2.458742   | 2.413642    | +2.88%     | +2.94%      |
| **TACO** | 2.395784   | 2.351210    | **+0.25%** | **+0.28%**  |

---

![Ablation](Ablation.png)


## 📂 Training Entry

Training scripts for Qwen2.5 are located at:

```bash
coccl_scripts/qwen2.5/run_qwen_taco.sh
```

## 📈 Comparison Results

![Comparison](Comparison.png)

---


---

## 📎 Notes

* Designed for large-scale distributed training (Megatron-LM based)
* Supports integration with existing TP/PP/DP pipelines
* Requires CUDA + NCCL + MPI environment

---

## 📖 Citation

If you find this work useful, please cite:

```
TACO: Efficient Communication Compression of Intermediate Tensors for Scalable Tensor-Parallel LLM Training
```

---
