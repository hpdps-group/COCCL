# 使用 COCCL 训练 Qwen3

[English](README.md) | 简体中文

本示例使用 NVIDIA Megatron-LM 和 COCCL 运行 Qwen3-0.6B Base。仓库中的子模块固定为已验证的 Megatron `core_v0.15.3` 版本。

## 准备

如果克隆仓库时没有拉取 Megatron-LM，请初始化该子模块：

```bash
git submodule update --init examples/training_scripts/Megatron-LM
```

按 [Megatron-LM 依赖说明](https://github.com/NVIDIA/Megatron-LM)创建并激活环境，然后编辑 [`training_envs.sh`](training_envs.sh)：

- 设置 `CUDA_HOME`，必要时覆盖 `MEGATRON_ROOT`；
- 将 `QWEN3_DATA_PREFIX` 设为 Megatron MMAP 数据 `.bin`/`.idx` 文件的共同前缀；
- 将 `QWEN3_TOKENIZER_DIR` 设为 Hugging Face Qwen3-0.6B tokenizer 目录；
- 如需继续训练，设置 Megatron Core checkpoint 目录 `QWEN3_LOAD_DIR`；
- 设置 `TRAIN_OUTPUT_ROOT` 和 `TRAIN_LOG_ROOT`。

启动前先编译 COCCL。脚本会使用 `$COCCL_ROOT/build/lib/libnccl.so.2` 替换 NCCL。请确认 PyTorch 动态链接 NCCL：

```bash
TORCH_CUDA_LIB=$(python -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent / "lib/libtorch_cuda.so")')
ldd "$TORCH_CUDA_LIB" | grep libnccl
```

如果命令没有输出，说明 PyTorch 静态嵌入了 NCCL，无法通过 `LD_PRELOAD` 替换。

## 配置 COCCL

[`configs/training.toml`](configs/training.toml) 默认为 DP AllGather 和 ReduceScatter 启用 SDP4Bit，开启自动调优，TP 和 PP 保持使用原生 NCCL。默认拓扑为两节点、每节点四张 GPU：

```text
TP=2, PP=2, DP=2, 开启 sequence parallel
```

启动拓扑变化时，必须同步修改 `training.classifier`。脚本使用 Megatron distributed optimizer，因此保持 `dp_strategy = "sdp"`。`TP_SIZE > 1` 时脚本会启用 sequence parallel。

## 启动

在每个节点上执行相同命令，只修改 `node_rank`：

```bash
# Node 0
bash train_qwen3_coccl.sh 2 4 0 10.0.0.1

# Node 1
bash train_qwen3_coccl.sh 2 4 1 10.0.0.1
```

脚本参数为：

```text
train_qwen3_coccl.sh <nnodes> <gpus_per_node> [node_rank]
                     [master_addr] [master_port]
```

常用训练参数通过环境变量覆盖：

```bash
TP_SIZE=2 PP_SIZE=2 TRAIN_ITERS=200 DP_OVERLAP=on \
  bash train_qwen3_coccl.sh 2 4 0 10.0.0.1
```

- `DP_OVERLAP=on|off` 控制 Megatron 的 gradient-reduce 和 parameter-gather overlap。
- `MICRO_BATCH_SIZE`、`GLOBAL_BATCH_SIZE`、`SEQUENCE_LENGTH`、`EVAL_INTERVAL` 和 `EVAL_ITERS` 控制训练。
- `TRANSFORMER_IMPL=auto|local|transformer_engine` 选择 Transformer 实现。
- `COCCL_ROOT` 和 `COCCL_CONFIG_FILE` 用于覆盖脚本推导的仓库和配置路径。

脚本不保存 checkpoint；`QWEN3_LOAD_DIR` 仅用于加载初始 Megatron Core 模型。

## TACO TP 和 PP

[`configs/training-taco.toml`](configs/training-taco.toml) 为 TP 的 AllGather/ReduceScatter
和 PP 的前后向通信启用 TACO E4M3/groupSize=128，DP 不压缩。
算法自动调优和 `depth = "auto"` 均开启。

配置好上述路径后，在两个节点分别执行，只修改节点编号：

```bash
COCCL_CONFIG_FILE="$PWD/configs/training-taco.toml" \
TP_SIZE=2 PP_SIZE=2 MICRO_BATCH_SIZE=32 GLOBAL_BATCH_SIZE=128 \
GRAD_REDUCE_DTYPE=bf16 TRAIN_ITERS=1000 EVAL_INTERVAL=100 EVAL_ITERS=1 \
DP_OVERLAP=off TRAIN_RUN_NAME=taco-tp-pp-mbs32 \
bash train_qwen3_coccl.sh 2 4 0 10.0.0.1
```

- TP=4/DP=1/PP=2 时，设置 `TP_SIZE=4`，并同步修改 TOML 中的分类器并行度。
- 仅压缩 TP 或 PP 时，复制一份 TOML，删除另一角色的策略段。
- 原生 NCCL 对照保持训练参数一致，设置 `COCCL_ENABLE=0` 和不同的
  `TRAIN_RUN_NAME`，在同一库中绕过压缩。
- `GRAD_REDUCE_DTYPE=fp32|bf16` 默认 FP32，模型权重保持 BF16。
  `TRAIN_RUN_NAME` 用于分开保存不同实验的输出与日志。

这些参数对应已完成的 1000 轮 TACO 测试。压缩不保证训练加速，需要同时比较 loss 和迭代时间。

## 检查自动路由

训练代码继续调用标准 NCCL API。COCCL 会分类每个 communicator，并应用匹配的 DP、TP 或 PP 策略。可以通过以下命令查看判断：

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COCCL_TUNING \
  bash train_qwen3_coccl.sh 2 4 0 10.0.0.1
```

为新拓扑启用压缩前，请确认日志中的 `role=DP|TP|PP` 和 COCCL 算法选择符合预期。
