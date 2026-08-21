# Integrating a COCCL Compressor

This directory is the user-owned extension boundary for compressor plugins.
A plugin may include the public C++ SDK and its own sources, but must not
include `core/` headers or depend on pipeline internals.

```cpp
#include "compressor_plugin/coccl_compressor_plugin.h"
#include <cuda_runtime.h>
```

COCCL currently accepts compressor ABI v8. The SDK generates the ABI
descriptor, configuration lifecycle, operation dispatch, and fixed entry
symbol. Plugin authors implement typed algorithm code rather than the raw ABI.

## Choose The Encoding Model

| Model | Use it when | Plugin contract |
| --- | --- | --- |
| Fixed layout | Shape and configuration determine one equal encoded size per logical chunk. | Implement `compress()` and `decompress()`, then commit exact bytes with `Output::commitBytes()`. |
| Framed layout | Each logical chunk can produce a different encoded size. | Declare `static constexpr bool kFramed = true`, emit one fixed-capacity slot and one metadata entry per chunk, then call `Output::commitFrames()`. |

Pack/Unpack, pipeline slicing, rank layout, metadata exchange, and NCCL
communication are Core responsibilities. A normal plugin receives contiguous
logical chunks and does not need to know how the collective is scheduled.

## Integration Steps

1. Create an immediate child directory, for example `mycodec/`, containing a
   `Makefile`. The root plugin Makefile discovers only immediate children with
   this file.
2. Keep the codec's existing kernels and Makefile or CMake project. Add one
   C++17/PIC adapter translation unit that includes only the public SDK.
3. Add a typed `Config` and `configure()` only when the codec has parameters.
4. Implement required `compress()` and `decompress()` methods. Define optional
   `decompressReduce()` or `decompressReduceCompress()` only when the codec has
   a real fused implementation; otherwise Core supplies the generic fallback.
5. Register the plugin exactly once with
   `COCCL_REGISTER_COMPRESSOR("mycodec", MyCompressor)`.
6. Build `$(SUBOBJDIR)/libcompress/libmycodec.so`, add `mycodec` to the TOML
   plugin catalog, and bind it to one or more policies.
7. Validate the configuration, exported symbol, encoded layout, output guard
   bytes, and GPU round trip.

## SDK Responsibilities

- `coccl::Input` is read-only and exposes data, bytes, elements, datatype,
  chunks, and optional frame metadata.
- `coccl::Output` owns the writable buffer and capacity. Compression must
  commit its output; successful decompression is committed to the planned raw
  shape by the SDK adapter.
- `coccl::Context` provides the call stream, rank/topology, typed config, and
  lazy resources.
- `Context::scratch()` is callback-local device workspace.
- `Context::persistent()` is device memory retained for a compressor handle
  and slot.
- `Context::instance()` creates typed Host state only when state must survive
  across calls.

Do not call `cudaMalloc`, `cudaFree`, NCCL registration, or communication APIs
inside normal compressor callbacks. Launch all work on `Context::stream()`.

For a fixed layout, calculate bytes before launching the encoder. If encoding
would be larger than the raw input, call
`output.passthrough(input, context.stream())`; Core transports and copies that
raw representation without calling the decoder. Check capacity before every
encoded write.

For a framed layout, each metadata entry contains an actual `payloadBytes`, an
`Encoded` or `Raw` tag, and `reserved = 0`. A Raw payload is the raw bytes of
that frame and can be smaller than an alignment-padded `frameStrideBytes`.
COCCL exchanges metadata and sends only `payloadBytes`; the plugin must not do
Host synchronization or communication.

Implement the optional Host-only `encodedSizeBound()` when shape and config
provide a safe upper bound. It must share byte-layout arithmetic with
execution and must not launch kernels or acquire resources. If omitted or if
it returns `ncclInvalidUsage`, COCCL plans the raw size. A DRC size query
describes the reduced raw output that will be recompressed.

Lossless byte-oriented codecs may declare:

```cpp
static constexpr bool kBytewiseLossless = true;
```

This allows automatic routing of `ncclInt8`, `ncclInt32`, and `ncclInt64` in
addition to the floating-point types supported by the normal compression
path. Do not declare it for a lossy or datatype-specific encoding.

## Minimal Fixed-Layout Example

The following adapter shows the complete SDK shape. `launchMyEncode` and
`launchMyDecode` stand for the codec's existing CUDA launchers.

```cpp
#include "compressor_plugin/coccl_compressor_plugin.h"
#include <cuda_runtime.h>

namespace {

struct MyConfig {
  int bits = 4;
};

template <typename Shape>
bool encodedBytes(const Shape& shape, int bits, size_t* bytes) {
  if (shape.chunks() == 0 || shape.elements() % shape.chunks() != 0 ||
      (bits != 4 && bits != 8)) {
    return false;
  }
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
    size_t bytes = 0;
    const Config& config = context.config<Config>();
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
    if (output.datatype() != ncclFloat32 ||
        input.chunks() != output.chunks()) {
      return ncclInvalidArgument;
    }
    launchMyDecode(input.data(), output.dataAs<float>(), output.elements(),
                   context.config<Config>().bits, context.stream());
    return coccl::fromCuda(cudaGetLastError());
  }
};

}  // namespace

COCCL_REGISTER_COMPRESSOR("mycodec", MyCompressor)
```

Bind the resulting DSO in TOML:

```toml
[compressor_plugins]
compressors = ["mycodec"]
library_path = "/path/to/COCCL/build/obj/coccl-extend/compressor_plugin/libcompress"

[normal.all_gather.default]
compressor = "mycodec"

[normal.all_gather.default.config]
bits = 4
```

The directory Makefile must honor `NCCLDIR`, `COCCL_ROOT`, `BUILDDIR`,
`SUBOBJDIR`, and `NVCC_GENCODE`, and emit
`$(SUBOBJDIR)/libcompress/libmycodec.so`. See
[taco/Makefile](taco/Makefile) for a small NVCC build and
[zfp/Makefile](zfp/Makefile) for a Make-to-CMake bridge. Do not replace an
existing CMake project; invoke it from the bridge.

## Built-In Plugin Parameters

| Plugin | Key and default | Effect and constraints |
| --- | --- | --- |
| `sdp4bit` | `groupCount = 2048` | Values per quantization group. Hadamard kernels cap their effective group at 128. |
| `sdp4bit` | `quantBits = 8` | Quantized payload width; only `4` and `8` are accepted. |
| `sdp4bit` | `quantType = "Symmetric"` | `Symmetric` or `Asymmetric` metadata and quantization. |
| `sdp4bit` | `hadamard = false` | Enables the Hadamard transform path. |
| `sdp4bit` | `pipelineSize = 1` | Number of persistent subAdd state slots. It is not the COCCL pipeline depth. |
| `sdp4bit` | `subAdd = false` | Enables stateful delta compression; it cannot be combined with Hadamard. |
| `tahquant` | `groupCount = 2048` | Values per quantization group; the encoded shape must contain complete groups. |
| `tahquant` | `quantBits = 8` | Quantized payload width; only `4` and `8` are accepted. |
| `tahquant` | `quantType = "Symmetric"` | `Symmetric` or `Asymmetric` quantization. |
| `tahquant` | `hadamard = false` | Enables the Hadamard path and caps the effective group at 128. |
| `tahquant` | `pivotSwap = false` | Emits pivot metadata; valid only with Hadamard enabled. |
| `taco` | `fp8Format = "E4M3"` | Selects `E4M3` or `E5M2`. |
| `taco` | `saturate = true` | Uses finite saturation during FP8 conversion. |
| `taco` | `groupSize = 128` | Accepted groups are `32`, `64`, `128`, `256`, and `512`. |
| `taco` | `targetRange = 448.0` | Target range used to derive the quantization scale. |
| `taco` | `lambda = 1e-6` | Stabilizer added to the variance estimate. |
| `taco` | `fp8MaxValue = 0.0` | Optional positive FP8 maximum override; `0` uses the format default. |
| `zfp` | `rate = 4` | Fixed rate in `1..64` bits per value. |
| `dietgpu` | `probBits = 10` | ANS probability precision; only `9`, `10`, and `11` are accepted. The codec is framed and bytewise lossless. |

Unknown keys are rejected by `ConfigReader::finish()`. The examples in
`../configs/` are the authoritative policy examples.

## Build And Validate

Build the host library first, then the focused plugin:

```bash
make -j src.build CUDA_HOME=/path/to/cuda \
  NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"

make -C src/coccl-extend/extensions/compressor_plugin/mycodec -j8 \
  CUDA_HOME=/path/to/cuda \
  NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"

make -f src/coccl-extend/Makefile coccl-config-check \
  BUILDDIR="$PWD/build" NCCLDIR="$PWD" \
  COCCL_ROOT=src/coccl-extend CUDA_HOME=/path/to/cuda

build/bin/coccl-config-check path/to/mycodec.toml
nm -D build/obj/coccl-extend/compressor_plugin/libcompress/libmycodec.so \
  | grep cocclGetCompressorPlugin
```

Build and config parsing are not numerical validation. On a GPU, test every
supported datatype, representative chunk count, partial group, raw
passthrough, capacity guard, and compress/decompress round trip.

Built-in reference points:

- [taco](taco): small configured fixed-layout plugin.
- [zfp](zfp): existing CMake project behind a Makefile bridge.
- [dietgpu](dietgpu): framed variable-length lossless codec.
- [tahquant](tahquant): optional scratch without persistent state.
- [sdp4bit](sdp4bit): fused reductions plus lazy state and persistent memory.

## Integrate With The Codex Skill

This repository includes
[`$coccl-integrate-compressor`](../../../../.codex/skills/coccl-integrate-compressor/SKILL.md).
Start Codex from the COCCL repository and provide the codec source path,
plugin name, fixed or framed layout, supported datatypes, build system, desired
TOML parameters, and target GPU architecture.

Example request:

```text
Use $coccl-integrate-compressor to integrate /data/code/mycodec as a fixed-size
plugin named mycodec. Preserve its CMake build, support FP32, expose bits=4|8,
add a normal AllGather TOML example, and validate for sm_80 without running a
top-level full rebuild.
```

The skill reads the current ABI v8 SDK, preserves the codec implementation and
build ownership, generates only the required adapter/bridge/config changes,
and reports which Host, build, symbol, and GPU checks actually ran.
