# Integration Review Checklist

## User Surface

- Plugin code contains only algorithm data structures that are genuinely
  required.
- Normal shape is `Config + Compressor`; `State` has a documented cross-call
  purpose and is created lazily.
- Plugin author does not implement an ABI descriptor, operation switch,
  config allocation/destruction, or buffer-manager handle lifecycle.
- Required callbacks read like algorithm code rather than runtime plumbing.

## Correctness

- Supported datatypes are explicit; unsupported types fail before launch.
- Input/output byte calculations are overflow checked.
- Fixed encodings larger than input select raw passthrough before kernel
  launch.
- Framed plugins declare `kFramed`, preserve one frame per logical chunk, use
  fixed-capacity slots, and commit valid per-frame device metadata.
- Framed Raw fallback is selected independently per frame; payload bytes
  describe the raw frame rather than padded slot capacity.
- Output capacity is checked and compression commits exact metadata.
- Chunk count and equal per-chunk fixed layout are preserved.
- Optional fused operations match their reported output shape.
- Optional encoded-size estimates are nonzero safe upper bounds and share
  checked arithmetic with execution.
- DRC estimates use reduced raw output shape and actual recompress parameters;
  exact output capacity is checked before launch.
- Size estimation is Host-only and does not consume scratch, persistent state,
  streams, or kernels; an unavailable estimate falls back to raw size.
- Scratch, persistent memory, and state lifetimes match actual use.
- `kBytewiseLossless` is declared only when arbitrary byte representations are
  preserved.

## Build And Configuration

- Existing Makefile or CMake remains authoritative.
- Directory Makefile honors the COCCL bridge contract.
- DSO is named `lib<descriptor-name>.so` and exposes the ABI v8 fixed entry.
- Vendored sources record exact commits, retain licenses, and build offline
  without unrelated frameworks, benchmarks, or tests.
- TOML catalog includes the plugin; a policy independently selects it.
- `configure()` consumes every documented key and rejects unknown keys.
- Focused build and `coccl-config-check` pass.

## Report

State which capabilities were implemented, which resources are used, which
datatypes and chunk layouts are supported, which build path was preserved, and
whether GPU numerical validation was run.
