# COCCL Compressor Plugins

This directory contains the built-in ABI v8 compressor plugins. Plugin code is
an extension boundary: it may include `compressor_plugin/coccl_compressor_plugin.h`
and its own sources, but must not include headers from `core/`.

Each plugin exports `cocclGetCompressorPlugin`, implements its config parser and
supported operations through the SDK, and builds one shared library under:

```text
build/obj/coccl-extend/compressor_plugin/libcompress/
```

The root Makefile discovers immediate child directories containing a Makefile.
Make-based plugins use the propagated `NCCLDIR`, `COCCL_ROOT`, `BUILDDIR`,
`SUBOBJDIR`, and `NVCC_GENCODE` values. ZFP and dietGPU retain their CMake build
and are invoked through their local Makefile bridges.

User-editable TOML examples live in `../configs/`. Configuration binds a plugin
to an operation and scope (`default`, `intra`, or `inter`); codec parameters are
owned by that scope and passed unchanged to the plugin parser.

Built-in plugins:

- `sdp4bit`: fixed-size 4/8-bit quantization and supported fused reductions.
- `tahquant`: fixed-size quantization with optional pivot metadata.
- `taco`: fixed-size FP8 encoding.
- `zfp`: fixed-rate CUDA ZFP encoding.
- `dietgpu`: framed variable-length ANS compression with raw fallback.

See `compressor_plugin/coccl_compressor_plugin.h` for the C++ SDK contract.
