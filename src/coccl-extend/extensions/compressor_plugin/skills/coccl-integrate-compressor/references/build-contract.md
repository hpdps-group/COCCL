# Plugin Build Contract

COCCL discovers immediate directories under
`src/coccl-extend/extensions/compressor_plugin/` that contain a `Makefile`. The
plugin may use any internal build system if that Makefile exposes the common
bridge.

## Required Contract

- Default target: build the plugin.
- `clean` target: remove only that plugin's outputs.
- Output: `$(SUBOBJDIR)/libcompress/lib<plugin-name>.so`.
- Consume caller values when provided: `NCCLDIR`, `COCCL_ROOT`, `BUILDDIR`,
  `SUBOBJDIR`, and `NVCC_GENCODE`.
- Compile the SDK adapter as C++17 with PIC.
- Include `$(COCCL_ROOT)/include` and `$(BUILDDIR)/include`.
- Export only the SDK-generated `cocclGetCompressorPlugin` entry when
  practical.

## Existing Makefile Or NVCC Project

Keep its source graph and flags. Add the SDK include path, C++17, PIC, standard
output path, and `all/clean` aliases if missing. See
`extensions/compressor_plugin/taco/Makefile`.

Do not copy another compressor's long Makefile unless its source graph actually
requires the same RDC or dependency-generation behavior.

## Existing CMake Project

Keep CMake as the source-of-truth build. Add a small adapter target to its
existing CMake graph, then make the directory-level Makefile run:

```text
cmake -S . -B <plugin-object-dir> <COCCL include/output options>
cmake --build <plugin-object-dir> --target <adapter-target> --parallel <jobs>
```

Prefer statically linking the third-party core into the plugin DSO so
deployment still consists of one `lib<name>.so`. Ensure all static objects use
PIC. See `extensions/compressor_plugin/zfp/Makefile` and
`extensions/compressor_plugin/zfp/coccl_interface/CMakeLists.txt`.

Do not replace the external project's CMake files with a copied COCCL
Makefile.

For an offline vendored dependency, pin the exact source commit in a short
vendor manifest, retain upstream licenses, and build only codec sources and
utilities required by the adapter. Keep local upstream patches narrow and
document whether they alter the encoded format. Static third-party libraries
must use PIC and be linked into the single plugin DSO.

## Validation Commands

Build COCCL once so generated `nccl.h` is available, then build only the
changed plugin:

```bash
make -j src.build CUDA_HOME=/path/to/cuda \
  NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"

make -C src/coccl-extend/extensions/compressor_plugin/<name> -j8 \
  CUDA_HOME=/path/to/cuda \
  NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"

make -f src/coccl-extend/Makefile coccl-config-check \
  BUILDDIR="$PWD/build" NCCLDIR="$PWD" \
  COCCL_ROOT=src/coccl-extend CUDA_HOME=/path/to/cuda

build/bin/coccl-config-check path/to/plugin-example.toml
nm -D build/obj/coccl-extend/compressor_plugin/libcompress/lib<name>.so \
  | grep cocclGetCompressorPlugin
```

When a GPU is available, add datatype/size/chunk round-trip tests and check
output guards. Build success and config parsing do not prove numerical kernel
correctness.
