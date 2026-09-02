# Add A Compressor To COCCL

English | [简体中文](README_zh-CN.md)

COCCL compressor plugins use the public C++17 SDK and build as independent
shared libraries. A plugin implements encoding; COCCL owns slicing, Pack and
Unpack, communication, workspace, and scheduling.

```cpp
#include "compressor_plugin/coccl_compressor_plugin.h"
```

The current compressor ABI is v9. Plugins include only the public SDK and
their own sources, never `core/` headers.

## Pick A Layout

- **Fixed layout:** every logical chunk has a predictable encoded size. Call
  `Output::commitBytes()` after encoding.
- **Framed layout:** each chunk may have a different payload size. Set
  `static constexpr bool kFramed = true`, write one metadata entry per chunk,
  and call `Output::commitFrames()`.

Use framed layout only for genuinely variable-length codecs. COCCL exchanges
frame sizes and sends only each frame's actual payload.

## Add The Plugin

1. Create `compressor_plugin/mycodec/` with a `Makefile`.
2. Keep the codec's CUDA implementation and existing Make or CMake build.
3. Add one SDK adapter with a typed `Config`, `compress()`, and
   `decompress()`.
4. Add fused `decompressReduce()` or `decompressReduceCompress()` only when
   the codec implements that operation. COCCL supplies the generic path.
5. Register once with
   `COCCL_REGISTER_COMPRESSOR("mycodec", MyCompressor)`.
6. Emit `libmycodec.so`, list it in TOML, and bind it to a policy.
7. Validate the exported symbol, output bounds, passthrough, and GPU round
   trip.

The parent Makefile discovers immediate child directories that contain a
`Makefile`.

## Know The SDK Contract

- `coccl::Input` exposes read-only data, shape, datatype, chunks, and optional
  frame metadata.
- `coccl::Output` exposes writable capacity. A compressor must commit its
  actual output.
- `coccl::Context` provides the CUDA stream, topology, typed config, and lazy
  resources.
- `scratch()` is temporary device workspace. `persistent()` and `instance()`
  retain device or Host state across calls.

Launch all CUDA work on `Context::stream()`. Do not allocate CUDA memory,
register NCCL memory, communicate, or synchronize the Host inside a normal
codec callback.

If fixed-layout encoding is not smaller than the input, use
`output.passthrough(input, context.stream())`. For framed codecs, mark each
frame `Encoded` or `Raw` and set its actual `payloadBytes`.

Implement the optional Host-only `encodedSizeBound()` when shape and config
give a safe upper bound. Reuse the execution layout formula. If no bound is
available, COCCL plans the raw size.

A lossless byte-oriented codec may declare:

```cpp
static constexpr bool kBytewiseLossless = true;
```

This enables Int8, Int32, and Int64 routing. Do not use it for lossy or
datatype-specific formats.

## Minimal Fixed Plugin

This adapter leaves the CUDA kernels to the codec and shows the COCCL-facing
code:

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

Bind the shared library to a collective:

```toml
[compressor_plugins]
compressors = ["mycodec"]
library_path = "/path/to/COCCL/build/obj/coccl-extend/compressor_plugin/libcompress"

[normal.all_gather.default]
compressor = "mycodec"

[normal.all_gather.default.config]
bits = 4
```

## Build And Check

Build COCCL, the plugin, and the config checker:

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

Then run GPU tests for every supported datatype, representative chunk and
partial-group shapes, raw passthrough, output guard bytes, and encode/decode
round trips.

See [TACO](taco) for a small NVCC plugin, [ZFP](zfp) for a CMake bridge,
[dietGPU](dietgpu) for framed lossless encoding, and [SDP4Bit](sdp4bit) for
fused reductions and persistent state.

## Built-In Parameters

- **SDP4Bit:** `groupCount`, `quantBits = 4|8`,
  `quantType = "Symmetric"|"Asymmetric"`, `hadamard`, and `subAdd`.
  `pipelineSize` controls subAdd state slots, not COCCL pipeline depth.
- **TACO:** `fp8Format = "E4M3"|"E5M2"`, `saturate`,
  `groupSize = 32|64|128|256|512`, `targetRange`, `lambda`, and optional
  `fp8MaxValue`.
- **ZFP:** `rate = 1..64` bits per value.
- **dietGPU:** `probBits = 9|10|11`. It uses dietGPU's floating-point codec
  for FP16, BF16, and FP32, and bytewise ANS for Int8, Int32, and Int64.
  Both paths are framed and lossless.

Unknown parameters are rejected. The examples in
[`../configs`](../configs) are the authoritative policy configurations.

## Use The Codex Skill

The bundled
[`coccl-integrate-compressor` skill](skills/coccl-integrate-compressor/SKILL.md)
can inspect a codec and generate its adapter, build bridge, config, and focused
tests.

Example request:

```text
Use $coccl-integrate-compressor to integrate /data/code/mycodec as a fixed-size
plugin named mycodec. Preserve its CMake build, support FP32, expose bits=4|8,
add an AllGather config, and validate only sm_80 targets.
```

Provide the codec path, plugin name, fixed or framed layout, supported
datatypes, build system, parameters, and target GPU architecture.
