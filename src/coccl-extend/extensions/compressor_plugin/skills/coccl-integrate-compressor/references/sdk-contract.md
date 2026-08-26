# COCCL Compressor SDK Contract

## Required Shape

Include only:

```cpp
#include "compressor_plugin/coccl_compressor_plugin.h"

struct MyCompressor {
  static coccl::Status compress(
      const coccl::Input&, coccl::Output&, coccl::Context&);
  static coccl::Status decompress(
      const coccl::Input&, coccl::Output&, coccl::Context&);
};

COCCL_REGISTER_COMPRESSOR("mycompressor", MyCompressor);
```

`compress()` must call `output.commit(elements, datatype, chunks)` or
`output.commitBytes(bytes, chunks)`. Before launching a compression kernel,
compare the calculated byte count with `input.bytes()`. If it is larger, call
`output.passthrough(input, context.stream())` instead. Successful decompression
is committed to the already planned output shape by the SDK adapter.

## Typed Configuration

Add `using Config = MyConfig` and `configure()` only when parameters exist:

```cpp
struct MyConfig {
  int bits = 4;
  bool stochastic = false;
};

static coccl::Status configure(coccl::ConfigReader& reader,
                               MyConfig& config,
                               const coccl::ConfigContext&) {
  return reader.get("bits", config.bits, 1, 8)
      .get("stochastic", config.stochastic)
      .finish();
}
```

Use `getEnum()` with an inline initializer list. Validate cross-field or
discrete constraints after `finish()`. Store derived values as small accessors
on `Config` instead of creating parameter-transfer structs.

## Data Views

- ABI v9 uses one `cocclCompressorView` for both directions. The SDK keeps
  `Input` read-only and `Output` writable; plugins should use these facades
  instead of accessing the ABI view directly.
- `Input`: `data()`, `bytes()`, `elements()`, `chunks()`, `datatype()`, and
  `elementsPerChunk()`.
- `Output`: `data()`, `capacityBytes()`, planned shape accessors, commit
  methods, and raw `passthrough()` support.
- `Context`: `stream()`, `rank()`, topology, typed config, `reduceChunks()`,
  and lazy resources. Fused DRC reads the input codec with
  `inputConfig<T>()` and the recompress codec with `config<T>()`.
- Use `coccl::checkedAdd()` and `coccl::checkedMultiply()` for byte-layout
  arithmetic.

Compressed bytes are authoritative for transport. Datatype and elements remain
available because kernel dispatch and typed compressed formats may require
them. Keep `chunks` meaningful: collective stages assume equal-sized logical
chunks.

The plugin receives contiguous logical chunks. Pipeline Pack/Unpack, swizzle,
rank placement, metadata collectives, and payload exchange stay in Core.

## Framed Variable-Length Output

Declare a variable-length compressor with:

```cpp
static constexpr bool kFramed = true;
```

One logical chunk is one independent frame. COCCL supplies a fixed-capacity
payload slot per frame plus device metadata. Use `Input::frameMetadata()`,
`Input::frameStrideBytes()`, and the matching `Output` accessors. Every emitted
metadata entry must satisfy:

```text
reserved == 0
0 < payloadBytes <= frameStrideBytes
encoding == Encoded or Raw
```

For `Raw`, `payloadBytes` is the raw byte count for that logical frame. It can
be smaller than an alignment-padded `frameStrideBytes`. Choose `Raw`
independently for any frame whose encoding is not smaller, copy the raw frame
into its slot, and call `output.commitFrames()` once the whole batch is queued.
`bytes` remains the total fixed slot capacity; transport reads actual wire
lengths from metadata. Compression and decompression should process all frames
as a batch, using masked decode or a separate Raw copy path when a batch mixes
encodings.

Framed plugins do not call `output.passthrough()`: that marker describes one
fixed whole-edge layout and cannot represent mixed frames. They also do not
perform metadata collectives, Host synchronization, or Send/Recv; those belong
to the COCCL pipeline.

## Optional Capabilities

Define only operations the compressor implements natively:

```cpp
decompressReduce(...)
decompressReduceCompress(...)
```

C++17 traits expose the corresponding capability automatically. Omit them to
use COCCL's fallback implementation. Do not add forwarding methods that only
call the fallback yourself.

A lossless codec that accepts arbitrary byte representations may declare:

```cpp
static constexpr bool kBytewiseLossless = true;
```

This enables Int8, Int32, and Int64 routing. Do not use it for a lossy or
datatype-specific codec.

## Optional Encoded-Size Bound

When encoded size is predictable from shape and config, expose the same
checked layout arithmetic to workspace planning:

```cpp
static coccl::Status encodedSizeBound(
    const coccl::Shape& shape, size_t* encodedBytes,
    const coccl::SizeContext& context);
```

Return a nonzero guaranteed upper bound for `Compress` and, when implemented,
`DecompressReduceCompress`. This callback is Host-only: it must not launch a
kernel, synchronize a stream, request scratch/persistent memory, or access
mutable state. Return `ncclInvalidUsage` for an operation that cannot be
estimated. Omitting the method is equivalent and makes COCCL use the
uncompressed output size.

`shape` always describes raw values about to be encoded. For `Compress` it is
the input shape. For `DecompressReduceCompress` it is the reduced raw output
shape, so calculate the bound with recompress bit width, group size, and
metadata layout. Input compressed bytes and reduction topology are not part of
size estimation.

An unpredictable framed codec should return a safe raw-slot bound. Metadata
still carries the smaller runtime payload length.

## Lazy Resources

- `context.scratch(bytes, &buffer)`: callback-local device workspace, released
  on the callback stream.
- `context.persistent(slot, bytes, &buffer)`: device allocation retained for
  this compressor handle and slot.
- `context.instance(&state)`: lazily constructs one typed Host state.

Call these inside the algorithm branch that needs them. A normal stateless
compressor must not create a `State` merely to satisfy an interface.

## Error Contract

- Reject unsupported datatypes and shapes before launching a kernel.
- Select raw passthrough before the capacity check when a fixed encoding would
  exceed input bytes; framed compressors select Raw independently per frame.
- Check output capacity before writing.
- Fused DRC must compare its exact recompress size with `capacityBytes()` before
  launching the kernel.
- If a compressor implements fused DRC but cannot fuse a particular input and
  output configuration pair, return `ncclInvalidUsage` before launching work;
  Core will run the generic decode/reduce/encode path.
- Check launch errors without adding a device synchronization.
- Convert CUDA errors with `coccl::fromCuda()`.
- Avoid exceptions across the plugin boundary.
- Keep `encodedSizeBound()` and execution on one shared byte-layout helper.
- Validate real codec constraints such as alignment, batch limits, and
  per-chunk layout before launch; unsupported framed shapes should emit Raw
  without writing outside their slots.
