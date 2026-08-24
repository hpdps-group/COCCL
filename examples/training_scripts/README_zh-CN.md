# 使用 COCCL 训练 Qwen

[English](README.md) | 简体中文

本示例通过 [Pai-Megatron-Patch](Pai-Megatron-Patch) 使用 COCCL 运行 Qwen2.5 或 Qwen3。训练框架以 Git 子模块形式固定版本。

## 运行前准备

如果克隆仓库时未拉取子模块，请初始化训练框架：

```bash
git submodule update --init --recursive \
  examples/training_scripts/Pai-Megatron-Patch
```

激活 Pai-Megatron-Patch 使用的 Python 环境，然后编辑 `training_envs.sh`。需要设置：

- `CUDA_HOME`。
- Qwen2.5 或 Qwen3 的数据前缀和模型路径。
- `TRAIN_OUTPUT_ROOT` 和 `TRAIN_LOG_ROOT`。
- 仅在需要覆盖默认值时设置 `COCCL_ROOT`、`COCCL_CONFIG_FILE` 或 `PAI_MEGATRON_ROOT`。

COCCL 会替代 NCCL 加载。请确认 PyTorch 使用动态链接的 NCCL 库：

```bash
TORCH_CUDA_LIB=$(python -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent / "lib/libtorch_cuda.so")')
ldd "$TORCH_CUDA_LIB" | grep libnccl
```

如果命令没有输出，说明 PyTorch 静态嵌入了 NCCL，无法通过 `LD_PRELOAD` 替换。请使用动态链接 NCCL 的 PyTorch。启动脚本会预加载 COCCL 的 `libnccl.so.2`；压缩器库路径来自 TOML 文件，不应加入 `LD_LIBRARY_PATH`。

## 配置 COCCL

默认的 [`configs/training.toml`](configs/training.toml) 会为已验证的 DP 策略加载 SDP4Bit，TP 和 PP 保持使用原生 NCCL。

以下分类器配置必须与训练命令一致：

- `data_parallel_size`、`tensor_parallel_size` 和 `pipeline_parallel_size` 必须描述实际任务拓扑。
- `dp_strategy` 必须与 DDP、分片式 distributed optimizer（`sdp`）或 FSDP 一致。
- `sequence_parallel` 必须与 TP 通信模式一致。

启动脚本按照 `DP = world_size / (TP * PP * CP)` 计算 DP 大小，并将该结果写入 `data_parallel_size`。`NCCL_SOCKET_IFNAME`、`NCCL_IB_HCA` 和 `CUDA_VISIBLE_DEVICES` 等机器相关设置应放在任务环境中。

## 启动

在每个节点运行相同命令，只修改节点 rank。以下示例使用两个节点，每个节点四张 GPU：

```bash
# Node 0
MASTER_ADDR=10.0.0.1 MASTER_PORT=29501 \
bash train_qwen_coccl.sh qwen25 2 4 0

# Node 1
MASTER_ADDR=10.0.0.1 MASTER_PORT=29501 \
bash train_qwen_coccl.sh qwen25 2 4 1
```

使用 `qwen3` 可运行 Qwen3。通过环境变量覆盖训练参数：

```bash
TP_SIZE=2 PP_SIZE=2 CP_SIZE=1 \
TRAIN_ITERS=100 DP_OVERLAP=off \
bash train_qwen_coccl.sh qwen3 2 4 0
```

`DP_OVERLAP` 可设为 `on` 或 `off`。启动脚本还支持 `MICRO_BATCH_SIZE`、`GLOBAL_BATCH_SIZE`、`SEQUENCE_LENGTH`、`WARMUP_ITERS`、`EVAL_INTERVAL` 和 `EVAL_ITERS`。

默认不保存 checkpoint。设置 `SAVE_CHECKPOINTS=on`，并可选设置 `SAVE_INTERVAL`，即可启用保存。

## 检查自动路由

训练代码继续调用标准 NCCL API。COCCL 会对每个 communicator 分类，然后选择对应的 DP、TP 或 PP 策略：

- 配置中唯一的 DP、TP 或 PP 大小会被立即分类。
- 存在歧义的 DP 和 TP communicator 会根据重复通信模式、拓扑、DP 策略和 sequence parallelism 进行识别。
- 小于 1 MiB 的消息不作为分类依据。
- 未知 communicator 保持使用原生 NCCL。
- PP 方向根据本地 rank 和 peer 推导。

验证新拓扑时启用路由日志：

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COCCL_TUNING \
bash train_qwen_coccl.sh qwen3 2 4 0
```

为对应角色启用压缩器前，请确认日志中出现 `role=DP|TP|PP` 和 `COCCL route`。
