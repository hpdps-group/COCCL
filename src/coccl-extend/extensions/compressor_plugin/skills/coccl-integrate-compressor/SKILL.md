---
name: coccl-integrate-compressor
description: Integrate or migrate fixed-size or framed variable-length CUDA compression algorithms into the COCCL C++17 compressor SDK and ABI v9 while preserving the plugin's existing Makefile or CMake build. Use when Codex must adapt an external compressor, generate Config/Compressor glue, replace legacy descriptors, connect scratch/persistent/state resources, add TOML policy parameters, vendor an offline codec, or validate a compressor shared library with COCCL.
---

# Integrate a COCCL Compressor

Integrate from the plugin author's perspective. Preserve algorithm kernels and
the plugin's build system; add only the adapter and build bridge COCCL needs.

## Workflow

1. Read `src/coccl-extend/include/compressor_plugin/coccl_compressor_plugin.h`
   and the source plugin before editing. Identify supported datatypes, output
   layout, configuration actually consumed, optional fused operations,
   temporary memory, and state that must survive across calls. Decide whether
   each logical chunk has a fixed encoding or needs an independent framed slot.
2. Read [references/sdk-contract.md](references/sdk-contract.md). Choose the
   smallest capability set that represents the algorithm.
3. Implement one `Config` only when parameters exist and one `Compressor` with
   required `compress()` and `decompress()`. Add `State` only for genuine
   cross-call state. Do not recreate ABI descriptors or lifecycle wrappers.
4. Keep algorithm-specific kernel launch helpers when they remove real
   dispatch duplication. Remove wrappers that merely rename an SDK callback.
5. Preserve the existing build system. Read
   [references/build-contract.md](references/build-contract.md), then add the
   plugin-directory `Makefile` bridge expected by COCCL.
6. Add the plugin to a focused TOML example. Keep catalog loading separate
   from primitive policy selection and include only parameters consumed by
   `configure()`.
7. Build the changed plugin and host objects incrementally, run
   `coccl-config-check`, inspect the fixed entry symbol, and run GPU round-trip
   tests when a CUDA device is available. Generate only the requested GPU
   architecture during focused validation.
8. Review with [references/review-checklist.md](references/review-checklist.md)
   before reporting completion.

## Design Rules

- Treat `Config + Compressor` as the normal plugin shape. A stateless,
  configuration-free plugin needs only `Compressor`.
- Use `COCCL_REGISTER_COMPRESSOR("name", Compressor)` as the only registration
  code. Never expose ABI fields, `void*` configuration lifecycle, or operation
  switches to the plugin author.
- Use the output buffer supplied by COCCL. Do not call `cudaMalloc`,
  `cudaFree`, or NCCL registration APIs inside normal plugin callbacks.
- Use `Context::scratch()` only for callback-local workspace,
  `Context::persistent()` only for device memory retained across calls, and
  `Context::instance()` only when the selected algorithm path needs Host state.
- Request lazy resources inside the branch that consumes them, never during
  plugin loading or configuration.
- Preserve logical chunk count and equal per-chunk layout. A plugin receives
  contiguous chunks and does not implement pipeline Pack/Unpack, rank
  placement, metadata exchange, or communication.
- Compute the encoded byte count before launching a fixed-layout compression
  kernel. If `coccl::shouldPassthrough(input, encodedBytes)` is true, return
  `output.passthrough(input, context.stream())` before checking encoded output
  capacity. Use `commitBytes()` for normal encoded payloads; `ncclUint8` is
  reserved by COCCL for the raw passthrough marker.
- For variable-length output, declare `static constexpr bool kFramed = true`.
  Encode one independent frame per logical chunk into its fixed-capacity slot,
  write valid device metadata, and finish with `output.commitFrames()`.
  Select `Raw` per frame when encoding is not smaller; do not use whole-edge
  passthrough for a framed compressor.
- Declare `static constexpr bool kBytewiseLossless = true` only for a codec
  that preserves arbitrary bytes. This is what enables automatic Int8, Int32,
  and Int64 routing.
- Implement optional `encodedSizeBound()` when the encoded layout is
  predictable from shape and config. Return a guaranteed Host-side upper
  bound, share its arithmetic with execution, and do not launch kernels or
  acquire SDK resources. Omit it when no safe bound is available; COCCL then
  plans the corresponding uncompressed size. For DRC, the supplied shape is
  the reduced raw output and the estimate must use recompress parameters.
- A fused DRC implementation reads the previous stage's typed configuration
  with `context.inputConfig<Config>()` and the recompress configuration with
  `context.config<Config>()`. Return `ncclInvalidUsage` before launching work
  when the compressor supports DRC but that configuration pair cannot be
  fused; Core then uses the generic fallback.
- Return `ncclInvalidArgument` for unsupported datatypes, shapes, or parameter
  combinations. Return CUDA failures through `coccl::fromCuda()`.
- Keep config defaults in the typed `Config`; let `ConfigReader::finish()`
  reject unknown keys. Do not retain obsolete or unused options.
- Do not force an external plugin to adopt COCCL's build implementation. Its
  directory only needs to honor the common output and target contract.
- For an offline vendored codec, pin exact upstream commits, retain every
  applicable license, document local format-preserving patches, and exclude
  unrelated language bindings, benchmarks, and tests from the plugin target.

## Reference Implementations

- Minimal configured Makefile plugin:
  `src/coccl-extend/extensions/compressor_plugin/taco/`
- Existing CMake project behind a Makefile bridge:
  `src/coccl-extend/extensions/compressor_plugin/zfp/`
- Framed variable-length codec with an offline CMake bridge:
  `src/coccl-extend/extensions/compressor_plugin/dietgpu/`
- Lazy instance and persistent memory:
  `src/coccl-extend/extensions/compressor_plugin/sdp4bit/`

Use these as decision examples, not as templates to copy wholesale.
