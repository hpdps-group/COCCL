# COCCL Source Layout

COCCL separates user-owned extensions from the runtime that plans and executes
them.

## Extensions

`extensions/` is the supported source area for application-specific changes:

- `primitives/` defines collective recipes, explicit `coccl*Comp*` entry
  points, and pipeline stage graphs.
- `compressor_plugin/` contains built-in compressor plugins and their vendor
  dependencies. Plugins may include only the public compressor SDK and their
  own sources.
- `configs/` contains editable runtime and compressor policy examples.

Primitive implementations use Core's narrow prepared-call, communicator, and
pipeline interfaces. Adding a primitive must not expose its implementation
headers to the rest of Core.

## Core

`core/` owns behavior that extension authors do not need to modify:

- `runtime/`: routing, prepared calls, group replay, communicator state, and
  primitive dispatch.
- `pipeline/`: shape planning, workspace layout, stream scheduling, stage
  execution, and frame exchange.
- `compression/`: compressor loading/runtime adapters and generic reduction
  fallback.
- `memory/`: registered and unregistered workspace pools.
- `config/`: TOML parsing, normalization, formatting, and logging.
- `tuning/`: profiling, cost models, and algorithm selection.
- `training/`: communication trace classification and training policy roles.
- `device/`: COCCL-owned CUDA kernels.

Core never includes headers from `extensions/`. Runtime invokes primitive
implementations only through `core/runtime/coccl_primitive_dispatch.h`.

## Public Includes

`include/runtime/` contains the NCCL integration hooks used by NCCL source
files. `include/compressor_plugin/` is the stable compressor ABI and C++ SDK.
All other headers are private to their owning Core module and are included from
the COCCL source root.

The dependency direction is:

```text
NCCL integration -> Core runtime -> Core modules
                              ^
                              |
                 Extensions primitives

Extensions compressor plugins -> Public compressor SDK only
```

Build products remain under `build/obj/coccl-extend/`; moving source ownership
does not change exported NCCL or COCCL symbols.
