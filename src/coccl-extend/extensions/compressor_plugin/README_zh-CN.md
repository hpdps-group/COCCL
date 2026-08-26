# 向 COCCL 集成压缩器

[English](README.md) | 简体中文

COCCL 压缩器插件使用公开的 C++17 SDK，并构建为独立共享库。插件负责实现编码，slice 划分、Pack/Unpack、通信、工作区和调度由 COCCL 负责。

```cpp
#include "compressor_plugin/coccl_compressor_plugin.h"
```

当前压缩器 ABI 版本为 v9。插件只能包含公开 SDK 和自身源码，不能包含 `core/` 头文件。

## 选择编码布局

- **固定布局：** 每个逻辑 chunk 的编码大小可预测。编码后调用 `Output::commitBytes()`。
- **分帧布局：** 每个 chunk 的 payload 大小可以不同。设置 `static constexpr bool kFramed = true`，为每个 chunk 写入一条 metadata，然后调用 `Output::commitFrames()`。

只有真正的变长编码才应使用分帧布局。COCCL 会交换 frame 大小，并且只发送每个 frame 的实际 payload。

## 添加插件

1. 创建带有 `Makefile` 的 `compressor_plugin/mycodec/` 目录。
2. 保留压缩算法已有的 CUDA 实现以及 Make 或 CMake 构建方式。
3. 添加一个 SDK 适配器，其中包含强类型 `Config`、`compress()` 和 `decompress()`。
4. 只有压缩算法自身实现相应操作时，才添加融合的 `decompressReduce()` 或 `decompressReduceCompress()`；否则使用 COCCL 提供的通用路径。
5. 使用 `COCCL_REGISTER_COMPRESSOR("mycodec", MyCompressor)` 注册一次。
6. 生成 `libmycodec.so`，在 TOML 中列出插件，并绑定到策略。
7. 验证导出符号、输出边界、passthrough 和 GPU 编解码闭环。

父级 Makefile 会自动发现包含 `Makefile` 的直接子目录。

## SDK 约定

- `coccl::Input` 提供只读数据、shape、datatype、chunks 和可选的 frame metadata。
- `coccl::Output` 提供可写容量，压缩器必须提交实际输出。
- `coccl::Context` 提供 CUDA stream、拓扑、强类型配置和按需资源。
- `scratch()` 是临时设备工作区；`persistent()` 和 `instance()` 用于跨调用保留设备或 Host 状态。

所有 CUDA 工作都必须提交到 `Context::stream()`。普通 codec 回调内部不得申请 CUDA 内存、注册 NCCL 内存、发起通信或同步 Host。

如果固定布局的编码结果不小于输入，请调用 `output.passthrough(input, context.stream())`。对于分帧压缩器，应将每个 frame 标记为 `Encoded` 或 `Raw`，并设置实际的 `payloadBytes`。

当 shape 和配置可以给出安全上界时，实现可选的纯 Host 接口 `encodedSizeBound()`。该接口应复用执行路径的布局公式。如果无法给出上界，COCCL 会按照原始数据大小规划。

面向字节的无损压缩器可以声明：

```cpp
static constexpr bool kBytewiseLossless = true;
```

这会启用 Int8、Int32 和 Int64 路由。不要对有损或依赖特定 datatype 的格式使用该能力。

## 最小固定布局插件

下面的适配器将 CUDA kernel 留给压缩算法实现，只展示面向 COCCL 的代码：

```cpp
#include "compressor_plugin/coccl_compressor_plugin.h"
#include <cuda_runtime.h>

namespace {

struct MyConfig {
  int bits = 4;
};

template <typename Shape>
bool encodedBytes(const Shape& shape, int bits, size_t* bytes) {
  size_t chunkBits = 0;
  size_t paddedBits = 0;
  return coccl::checkedMultiply(shape.elementsPerChunk(), (size_t)bits,
                                &chunkBits) &&
      coccl::checkedAdd(chunkBits, 7, &paddedBits) &&
      coccl::checkedMultiply(paddedBits / 8, shape.chunks(), bytes);
}

void launchMyEncode(const float* input, void* output, size_t elements,
                    int bits, cudaStream_t stream);
void launchMyDecode(const void* input, float* output, size_t elements,
                    int bits, cudaStream_t stream);

struct MyCompressor {
  using Config = MyConfig;

  static coccl::Status configure(coccl::ConfigReader& reader, Config& config,
                                 const coccl::ConfigContext&) {
    coccl::Status result = reader.get("bits", config.bits, 4, 8).finish();
    if (result != ncclSuccess) return result;
    return config.bits == 4 || config.bits == 8
        ? ncclSuccess : ncclInvalidArgument;
  }

  static coccl::Status encodedSizeBound(
      const coccl::Shape& shape, size_t* bytes,
      const coccl::SizeContext& context) {
    if (context.operation() != cocclCompressorOperationCompress) {
      return ncclInvalidUsage;
    }
    return encodedBytes(shape, context.config<Config>().bits, bytes)
        ? ncclSuccess : ncclInvalidArgument;
  }

  static coccl::Status compress(const coccl::Input& input,
                                coccl::Output& output,
                                coccl::Context& context) {
    if (input.datatype() != ncclFloat32) return ncclInvalidArgument;
    const Config& config = context.config<Config>();
    size_t bytes = 0;
    if (!encodedBytes(input, config.bits, &bytes)) return ncclInvalidArgument;
    if (coccl::shouldPassthrough(input, bytes)) {
      return output.passthrough(input, context.stream());
    }
    if (bytes > output.capacityBytes()) return ncclInvalidArgument;
    launchMyEncode(input.dataAs<float>(), output.data(), input.elements(),
                   config.bits, context.stream());
    coccl::Status result = coccl::fromCuda(cudaGetLastError());
    return result == ncclSuccess
        ? output.commitBytes(bytes, input.chunks()) : result;
  }

  static coccl::Status decompress(const coccl::Input& input,
                                  coccl::Output& output,
                                  coccl::Context& context) {
    if (output.datatype() != ncclFloat32) return ncclInvalidArgument;
    launchMyDecode(input.data(), output.dataAs<float>(), output.elements(),
                   context.config<Config>().bits, context.stream());
    return coccl::fromCuda(cudaGetLastError());
  }
};

}  // namespace

COCCL_REGISTER_COMPRESSOR("mycodec", MyCompressor)
```

将共享库绑定到集合通信：

```toml
[compressor_plugins]
compressors = ["mycodec"]
library_path = "/path/to/COCCL/build/obj/coccl-extend/compressor_plugin/libcompress"

[normal.all_gather.default]
compressor = "mycodec"

[normal.all_gather.default.config]
bits = 4
```

## 构建与检查

构建 COCCL、插件和配置检查器：

```bash
make src.build CUDA_HOME=/path/to/cuda \
  NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"

make -C src/coccl-extend/extensions/compressor_plugin/mycodec \
  CUDA_HOME=/path/to/cuda \
  NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"

make -f src/coccl-extend/Makefile coccl-config-check \
  BUILDDIR="$PWD/build" NCCLDIR="$PWD" \
  COCCL_ROOT=src/coccl-extend CUDA_HOME=/path/to/cuda

build/bin/coccl-config-check path/to/mycodec.toml
nm -D build/obj/coccl-extend/compressor_plugin/libcompress/libmycodec.so \
  | grep cocclGetCompressorPlugin
```

随后应针对每种支持的 datatype、具有代表性的 chunk 和 partial-group shape、raw passthrough、输出 guard bytes 以及编解码闭环运行 GPU 测试。

不同类型的实现示例可参考：使用 NVCC 构建的小型插件 [TACO](taco/README_zh-CN.md)、使用 CMake bridge 的 [ZFP](zfp)、采用分帧无损编码的 [dietGPU](dietgpu)，以及包含融合归约与 persistent state 的 [SDP4Bit](sdp4bit)。

## 内置插件参数

- **SDP4Bit：** `groupCount`、`quantBits = 4|8`、`quantType = "Symmetric"|"Asymmetric"`、`hadamard` 和 `subAdd`。`pipelineSize` 控制 subAdd 状态槽位，而不是 COCCL pipeline depth。
- **TACO：** `fp8Format = "E4M3"|"E5M2"`、`saturate`、`groupSize = 32|64|128|256|512`、`targetRange`、`lambda` 和可选的 `fp8MaxValue`。
- **ZFP：** `rate = 1..64` bits/value。
- **dietGPU：** `probBits = 9|10|11`。该插件采用分帧、面向字节的无损编码，支持浮点数据以及 Int8、Int32 和 Int64。

未知参数会被拒绝。[`../configs`](../configs) 中的示例是权威策略配置。

## 使用 Codex Skill

项目内置的 [`coccl-integrate-compressor` skill](skills/coccl-integrate-compressor/SKILL.md) 可以检查压缩算法，并生成适配器、构建 bridge、配置和有针对性的测试。

示例请求：

```text
Use $coccl-integrate-compressor to integrate /data/code/mycodec as a fixed-size
plugin named mycodec. Preserve its CMake build, support FP32, expose bits=4|8,
add an AllGather config, and validate only sm_80 targets.
```

请提供压缩算法路径、插件名称、固定或分帧布局、支持的 datatype、构建系统、参数以及目标 GPU 架构。
