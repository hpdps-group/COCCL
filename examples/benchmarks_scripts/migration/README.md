# M0 benchmark harness

The M0 performance matrix matches the cases in
`communication_benchmarks_single_node.sh`, but runs every message size in an
independent process and repeats each point twice. Each variant runs 1080
processes with `-w 10 -n 20`: native collectives plus SDP4Bit and cuZFP at
pipeline depths 1, 2, 4, and 8. ReduceScatter TwoShot, AllReduce TripleShot,
TahQuant, and the retired Ring experiments are outside this single-node
matrix. The runner waits two seconds after every completed 8 GiB process.

The parser compares the median of the two runs. Performance differences
below 512 MiB are informational because short kernels are sensitive to launch
and scheduling noise. Only paired points at or above 512 MiB use the +/-3% M0
hard gate. Memory sampling remains isolated by process and runs twice.

Any initial hard-gate or memory-noise candidate is rerun with a balanced order:
original run 1, migrate-copy run 1, migrate-copy run 2, original run 2. The
initial result remains in the report as a candidate; the balanced two-run
confirmation determines the final gate.

All commands are intended to run on `d1n41a14g01`.

```bash
bash examples/benchmarks_scripts/migration/m0_build.sh \
  /data/home/scyb672/run/lxc/COCCL-migrate

bash examples/benchmarks_scripts/migration/m0_matrix.sh \
  migrate-copy /data/home/scyb672/run/lxc/COCCL-migrate smoke

bash examples/benchmarks_scripts/migration/m0_matrix.sh \
  migrate-copy /data/home/scyb672/run/lxc/COCCL-migrate full

bash examples/benchmarks_scripts/migration/m0_matrix.sh \
  migrate-copy /data/home/scyb672/run/lxc/COCCL-migrate memory

bash examples/benchmarks_scripts/migration/m0_confirm_performance.sh performance
bash examples/benchmarks_scripts/migration/m0_confirm_performance.sh memory
```

Set `M0_FORCE=1` to rerun completed work. `M0_PERFORMANCE_RUNS` changes the
default two performance samples; `M0_MEMORY_RUNS` changes the default two
memory samples per case.
